"""GraphQL agent using declarative queries (GraphQL + DuckDB SQL)."""

import json
import logging
import re
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from agents import Agent, MaxTurnsExceeded, Runner, function_tool

from ..config import settings
from ..context import RequestContext
from ..executor import (
    execute_sql,
    extract_tables_from_response,
    truncate_for_context,
)
from ..graphql import execute_query as graphql_fetch
from ..recipe import (
    RECIPE_STORE,
    _return_directly_flag,
    _set_return_directly,
    _tools_to_final_output,
    build_api_id,
    build_partial_result,
    build_recipe_docstring,
    create_params_model,
    deduplicate_tool_name,
    execute_recipe_steps,
    format_recipe_response,
    maybe_extract_and_save_recipe,
    render_text_template,
    search_recipes,
    validate_and_prepare_recipe,
    validate_recipe_params,
)
from ..tracing import trace_metadata
from .contextvar_utils import safe_append_contextvar_list, safe_get_contextvar
from .model import get_run_config, model
from .progress import get_turn_context, reset_progress
from .prompts import (
    CONTEXT_SECTION,
    DECISION_GUIDANCE,
    EFFECTIVE_PATTERNS,
    GRAPHQL_SCHEMA_NOTATION,
    OPTIONAL_PARAMS_SPEC,
    PERSISTENCE_SPEC,
    SEARCH_TOOL_DESC,
    SQL_RULES,
    SQL_TOOL_DESC,
    TOOL_USAGE_RULES,
    UNCERTAINTY_SPEC,
)
from .schema_search import create_search_schema_tool

logger = logging.getLogger(__name__)


def _log(msg: str) -> None:
    """Log agent activity only in debug mode."""
    if settings.DEBUG:
        logger.info(f"[GQL] {msg}")


# Context-local storage (isolated per async request)
# NOTE: Use mutable containers for values that need to be modified by tool functions,
# because ContextVar.set() in child tasks (task groups) doesn't propagate to parent.
_graphql_queries: ContextVar[list[str]] = ContextVar("graphql_queries")
_recipe_steps: ContextVar[list[dict[str, Any]]] = ContextVar("recipe_steps")
_query_results: ContextVar[dict[str, Any]] = ContextVar("query_results")
_last_result: ContextVar[list] = ContextVar("last_result")  # Mutable container: [result_value]
_raw_schema: ContextVar[str] = ContextVar("raw_schema")  # Raw introspection JSON for search
_sql_steps: ContextVar[list[str]] = ContextVar("sql_steps")


def _format_type(t: dict | None) -> str:
    """Convert introspection type to compact notation: [User!]!"""
    if not t:
        return "?"
    kind = t.get("kind")
    name = t.get("name")
    inner = t.get("ofType")

    if kind == "NON_NULL":
        return f"{_format_type(inner)}!"
    if kind == "LIST":
        return f"[{_format_type(inner)}]"
    return name or "?"


_INTROSPECTION_QUERY = """{
  __schema {
    queryType {
      fields { name description args { name type { ...TypeRef } defaultValue } type { ...TypeRef } }
    }
    types {
      name kind description
      fields { name description args { name type { ...TypeRef } defaultValue } type { ...TypeRef } }
      enumValues { name description }
      inputFields { name type { ...TypeRef } defaultValue }
      interfaces { name }
      possibleTypes { name }
    }
  }
}
fragment TypeRef on __Type {
  name kind ofType { name kind ofType { name kind ofType { name } } }
}"""

# Shallow introspection for APIs with strict depth limits
_INTROSPECTION_QUERY_SHALLOW = """{
  __schema {
    queryType { fields { name args { name } type { name kind } } }
    types {
      name kind
      fields { name type { name kind } }
      inputFields { name }
      enumValues { name }
    }
  }
}"""


def _is_required(type_def: dict | None) -> bool:
    """Check if GraphQL type is required (NON_NULL wrapper)."""
    return type_def.get("kind") == "NON_NULL" if type_def else False


def _format_arg(a: dict) -> str:
    """Format argument with optional default value."""
    type_str = _format_type(a["type"])
    default = a.get("defaultValue")
    if default is not None:
        return f"{a['name']}: {type_str} = {default}"
    return f"{a['name']}: {type_str}"


def _filter_required_args(args: list[dict]) -> list[dict]:
    """Filter to only required arguments (NON_NULL type)."""
    return [a for a in args if _is_required(a.get("type"))]


def _format_field(fld: dict) -> str:
    """Format a field with optional args."""
    args = fld.get("args", [])
    if args:
        arg_str = "(" + ", ".join(_format_arg(a) for a in args) + ")"
    else:
        arg_str = ""
    desc = f" # {fld['description']}" if fld.get("description") else ""
    return f"  {fld['name']}{arg_str}: {_format_type(fld['type'])}{desc}"


def _build_schema_context(schema: dict) -> str:
    """Build compact SDL context from introspection schema."""
    queries = schema.get("queryType", {}).get("fields", [])
    all_types = [t for t in schema.get("types", []) if not t["name"].startswith("__")]

    objects = [
        t
        for t in all_types
        if t["kind"] == "OBJECT" and t["name"] not in ("Query", "Mutation", "Subscription")
    ]
    enums = [t for t in all_types if t["kind"] == "ENUM"]
    inputs = [t for t in all_types if t["kind"] == "INPUT_OBJECT"]
    interfaces = [t for t in all_types if t["kind"] == "INTERFACE"]
    unions = [t for t in all_types if t["kind"] == "UNION"]

    lines = ["<queries>"]
    for f in queries:
        desc = f" # {f['description']}" if f.get("description") else ""
        # Only show required args
        required_args = _filter_required_args(f.get("args", []))
        args = ", ".join(_format_arg(a) for a in required_args)
        lines.append(f"{f['name']}({args}) -> {_format_type(f['type'])}{desc}")

    if interfaces:
        lines.append("\n<interfaces>")
        for t in interfaces:
            impl = [p["name"] for p in t.get("possibleTypes", []) or []]
            impl_str = f" # implemented by: {', '.join(impl)}" if impl else ""
            fields = [_format_field(fld) for fld in t.get("fields", []) or []]
            lines.append(f"{t['name']} {{{impl_str}\n" + "\n".join(fields) + "\n}")

    if unions:
        lines.append("\n<unions>")
        for t in unions:
            types = [p["name"] for p in t.get("possibleTypes", []) or []]
            lines.append(f"{t['name']}: {' | '.join(types)}")

    lines.append("\n<types>")
    for t in objects:
        impl = [i["name"] for i in t.get("interfaces", []) or []]
        impl_str = f" implements {', '.join(impl)}" if impl else ""
        fields = [_format_field(fld) for fld in t.get("fields", []) or []]
        lines.append(f"{t['name']}{impl_str} {{\n" + "\n".join(fields) + "\n}")

    lines.append("\n<enums>")
    for e in enums:
        vals = " | ".join(v["name"] for v in e.get("enumValues", []))
        lines.append(f"{e['name']}: {vals}")

    lines.append("\n<inputs>")
    for inp in inputs:
        # Only show required input fields
        required_fields = [
            f for f in (inp.get("inputFields", []) or []) if _is_required(f.get("type"))
        ]
        fields = ", ".join(f"{f['name']}: {_format_type(f['type'])}" for f in required_fields)
        lines.append(f"{inp['name']} {{ {fields} }}")

    return "\n".join(lines)


def _strip_descriptions(context: str) -> str:
    """Strip # comments from SDL context."""
    return re.sub(r" #[^\n]*", "", context)


def _is_depth_limit_error(result: dict) -> bool:
    """Check if error is due to query depth limit (413)."""
    error = result.get("error", "")
    if isinstance(error, str):
        return "413" in error or "depth" in error.lower()
    if isinstance(error, list):
        return any("depth" in str(e).lower() for e in error)
    return False


async def _fetch_schema_context(endpoint: str, headers: dict[str, str] | None) -> str:
    """Fetch schema in compact SDL format. Falls back to shallow query on depth limit."""
    result = await graphql_fetch(_INTROSPECTION_QUERY, None, endpoint, headers)

    # Retry with shallow introspection if depth limit exceeded
    if not result.get("success") and _is_depth_limit_error(result):
        logger.info("Full introspection failed (depth limit), retrying with shallow query")
        result = await graphql_fetch(_INTROSPECTION_QUERY_SHALLOW, None, endpoint, headers)

    if not result.get("success") or not result.get("data"):
        return ""

    schema = result["data"]["__schema"]

    # Store raw introspection JSON for grep-like search (preserves all info)
    _raw_schema.set(json.dumps(schema, indent=2))

    # Build DSL for LLM context
    context = _build_schema_context(schema)

    if len(context) > settings.MAX_SCHEMA_CHARS:
        context = _strip_descriptions(context)
        if len(context) > settings.MAX_SCHEMA_CHARS:
            context = (
                context[: settings.MAX_SCHEMA_CHARS]
                + "\n[SCHEMA TRUNCATED - use search_schema() to explore]"
            )

    return context


async def fetch_graphql_schema_raw(endpoint: str, headers: dict[str, str] | None) -> str:
    """Fetch raw GraphQL schema JSON for matching and validation."""
    result = await graphql_fetch(_INTROSPECTION_QUERY, None, endpoint, headers)

    if not result.get("success") and _is_depth_limit_error(result):
        logger.info("Full introspection failed (depth limit), retrying with shallow query")
        result = await graphql_fetch(_INTROSPECTION_QUERY_SHALLOW, None, endpoint, headers)

    if not result.get("success") or not result.get("data"):
        return ""

    schema = result["data"]["__schema"]
    return json.dumps(schema, indent=2)


def _build_system_prompt(recipe_context: str = "") -> str:
    """Build system prompt for GraphQL agent."""
    current_date = datetime.now().strftime("%Y-%m-%d")

    workflow_start = "1"

    return f"""You are a GraphQL API agent that answers questions by querying APIs and returning data.

{SQL_RULES}

## GraphQL-Specific
- Use inline values, never $variables

<tools>
graphql_query(query, name?, return_directly?)
  Execute GraphQL query. Result stored as DuckDB table.
  - return_directly: Skip LLM analysis, return raw data directly to user

{SQL_TOOL_DESC}

{SEARCH_TOOL_DESC}
</tools>
<workflow>
{workflow_start}. Read <queries> and <types> provided below
{int(workflow_start) + 1}. Execute graphql_query with needed fields
{int(workflow_start) + 2}. If user needs filtering/aggregation → sql_query, else return data
</workflow>

{CONTEXT_SECTION.format(current_date=current_date, max_turns=settings.MAX_AGENT_TURNS)}

{recipe_context}

{DECISION_GUIDANCE}

{GRAPHQL_SCHEMA_NOTATION}

{UNCERTAINTY_SPEC}

{OPTIONAL_PARAMS_SPEC}

{PERSISTENCE_SPEC.format(max_turns=settings.MAX_AGENT_TURNS)}

{EFFECTIVE_PATTERNS}

{TOOL_USAGE_RULES}

<examples>
Simple: graphql_query('{{ users(limit: 10) {{ id name }} }}')
Aggregation: graphql_query('{{ posts {{ authorId views }} }}'); sql_query('SELECT authorId, SUM(views) as total FROM data GROUP BY authorId')
Join: graphql_query('{{ users {{ id name }} }}', name='u'); graphql_query('{{ posts {{ authorId title }} }}', name='p'); sql_query('SELECT u.name, p.title FROM u JOIN p ON u.id = p.authorId')
</examples>
"""


def _create_graphql_query_tool(ctx: RequestContext):
    """Create graphql_query tool with bound context."""

    @function_tool
    async def graphql_query(query: str, name: str = "data", return_directly: bool = False) -> str:
        """Execute GraphQL query and store result for sql_query.

        Args:
            query: GraphQL query string
            name: Table name for sql_query (default: "data")
            return_directly: Skip LLM processing, return data directly to client.
                            Only applies on success. Errors still processed by LLM.

        Returns:
            JSON string with query results
        """
        result = await graphql_fetch(query, None, ctx.target_url, ctx.target_headers)

        schema_info = None
        stored_data = None
        if result.get("success"):
            try:
                results = _query_results.get()
                data = result.get("data", {})
                tables, schema_info = extract_tables_from_response(data, name)
                results.update(tables)
                _query_results.set(results)
                # Store full data for final response (the extracted list)
                # Mutate in-place so changes propagate from task group child
                stored_data = tables.get(name)
                if stored_data is not None:
                    _last_result.get()[0] = stored_data
            except LookupError:
                pass

            # Track successful step for recipe extraction
            safe_append_contextvar_list(
                _recipe_steps, {"kind": "graphql", "query": query, "name": name}
            )

        safe_append_contextvar_list(_graphql_queries, query)

        _log(f"RESULT {json.dumps(result)[:200]}")

        if return_directly and result.get("success"):
            _set_return_directly()

        # Smart context optimization - cap by chars for LLM safety
        if result.get("success") and stored_data:
            # Wrapped dict (1-row) → return schema info
            if schema_info:
                return json.dumps(
                    {"success": True, "table": name, **schema_info},
                    indent=2,
                )

            # Apply char-based truncation (normalized format)
            if isinstance(stored_data, list):
                return json.dumps(
                    {"success": True, **truncate_for_context(stored_data, name)},
                    indent=2,
                )

        if not result.get("success"):
            result["hint"] = (
                "Use search_schema to find valid field names, enum values, or required args"
            )

        return json.dumps(result, indent=2)

    return graphql_query


# Create search_schema tool bound to GraphQL schema context var
search_schema = create_search_schema_tool(_raw_schema)


@function_tool
def sql_query(sql: str, return_directly: bool = False) -> str:
    """Run DuckDB SQL on stored GraphQL results.

    Args:
        sql: DuckDB SQL query
        return_directly: Skip LLM processing, return results directly to client

    Returns:
        JSON string with query results
    """
    try:
        data = _query_results.get()
    except LookupError:
        return json.dumps({"success": False, "error": "No data. Call graphql_query first."})

    if not data:
        return json.dumps({"success": False, "error": "No data. Call graphql_query first."})

    result = execute_sql(data, sql)

    _log(f"SQL {json.dumps(result)[:200]}")

    # Store full result for final response + apply char truncation for LLM
    if result.get("success"):
        rows = result.get("result", [])
        # Mutate in-place so changes propagate from task group child
        try:
            _last_result.get()[0] = rows
        except LookupError:
            pass

        # Track successful SQL for recipe extraction
        safe_append_contextvar_list(_sql_steps, sql)

        if return_directly:
            _set_return_directly()

        if isinstance(rows, list):
            return json.dumps(
                {"success": True, **truncate_for_context(rows, "sql_result")},
                indent=2,
            )

    return json.dumps(result, indent=2)


def _create_individual_recipe_tools(
    ctx: RequestContext,
    suggestions: list[dict[str, Any]],
) -> list:
    """Generate one function_tool per recipe suggestion."""
    tools = []
    seen_names: set[str] = set()

    for s in suggestions:
        recipe = RECIPE_STORE.get_recipe(s["recipe_id"])
        if not recipe:
            continue

        tool_name = deduplicate_tool_name(s.get("tool_name", "unknown_recipe"), seen_names)
        params_spec = recipe.get("params", {})
        docstring = build_recipe_docstring(
            s["question"],
            recipe.get("steps", []),
            recipe.get("sql_steps", []),
            "graphql",
            params_spec=params_spec,
        )

        def make_tool(rid: str, pspec: dict[str, Any], doc: str, tname: str):
            ParamsModel = create_params_model(pspec, tname)

            async def dynamic_recipe_tool(
                params: ParamsModel,
                return_directly: bool = True,
            ) -> str:
                kwargs = params.model_dump()
                validated_params, error = validate_recipe_params(pspec, kwargs)
                if error:
                    return error

                recipe, validated_params, error = validate_and_prepare_recipe(
                    rid, json.dumps(kwargs), _raw_schema
                )
                if error:
                    return error

                async def graphql_step_executor(step_idx, step, params, results):
                    if not isinstance(step, dict) or step.get("kind") != "graphql":
                        return (
                            False,
                            None,
                            json.dumps(
                                {"success": False, "error": "invalid recipe step"}, indent=2
                            ),
                            None,
                        )

                    name = step.get("name") or "data"
                    tmpl = step.get("query_template")
                    if not isinstance(tmpl, str):
                        return (
                            False,
                            None,
                            json.dumps(
                                {"success": False, "error": "missing query_template"}, indent=2
                            ),
                            None,
                        )

                    query = render_text_template(tmpl, params)
                    res = await graphql_fetch(query, None, ctx.target_url, ctx.target_headers)
                    if not res.get("success"):
                        return (
                            False,
                            None,
                            json.dumps(
                                {"success": False, "error": res.get("error", "query failed")},
                                indent=2,
                            ),
                            None,
                        )

                    data = res.get("data", {})
                    tables, _ = extract_tables_from_response(data, str(name))
                    results.update(tables)
                    _query_results.set(results)
                    safe_append_contextvar_list(_graphql_queries, query)
                    return True, tables.get(str(name)), "", query

                executed_queries: list[str] = []
                if recipe is None or validated_params is None:
                    return json.dumps({"success": False, "error": "recipe validation failed"})
                success, last_data, executed_sql, error = await execute_recipe_steps(
                    recipe,
                    validated_params,
                    _query_results,
                    _last_result,
                    graphql_step_executor,
                    executed_queries,
                )
                if not success:
                    return error

                # Track executed SQL for tracing
                for sql in executed_sql:
                    safe_append_contextvar_list(_sql_steps, sql)

                if return_directly:
                    _set_return_directly()

                return format_recipe_response(
                    _last_result,
                    executed_queries,
                    executed_sql,
                    "executed_queries",
                )

            dynamic_recipe_tool.__name__ = tname
            dynamic_recipe_tool.__doc__ = doc
            return function_tool(dynamic_recipe_tool)

        tools.append(make_tool(s["recipe_id"], params_spec, docstring, tool_name))

    return tools


async def process_query(question: str, ctx: RequestContext) -> dict[str, Any]:
    """Process natural language query against GraphQL API.

    Args:
        question: Natural language question
        ctx: Request context with target_url and target_headers
    """
    try:
        _log(f"QUERY {question[:80]}")

        # Reset per-request storage
        # Use mutable containers so tool functions can modify in-place
        # (ContextVar.set() in child tasks doesn't propagate to parent)
        _graphql_queries.set([])
        _recipe_steps.set([])
        _sql_steps.set([])
        _query_results.set({})
        _last_result.set([None])  # Mutable list: [result_value]
        _return_directly_flag.set([])  # Reset direct return flag
        reset_progress()  # Reset turn counter

        # Fetch schema with dynamic endpoint
        schema_ctx = await _fetch_schema_context(ctx.target_url, ctx.target_headers)

        # Pre-flight recipe search
        suggestions, recipe_context = [], ""
        if settings.ENABLE_RECIPES:
            raw_schema = safe_get_contextvar(_raw_schema, "")
            api_id = build_api_id(ctx, "graphql")
            suggestions, recipe_context = search_recipes(api_id, raw_schema, question)
            if suggestions:
                _log(
                    f"PRE-FLIGHT found={len(suggestions)} ids={[s['recipe_id'] for s in suggestions]}"
                )
            elif raw_schema:
                _log(f"PRE-FLIGHT no matches for api_id={api_id[:50]}")

        # Create tools with bound context
        gql_tool = _create_graphql_query_tool(ctx)
        tools = [gql_tool, sql_query, search_schema]
        if suggestions:  # Create individual recipe tools for each suggestion
            recipe_tools = _create_individual_recipe_tools(ctx, suggestions)
            tools = [*recipe_tools, *tools]

        # Create fresh agent with dynamic tools
        agent = Agent(
            name="graphql-agent",
            model=model,
            instructions=_build_system_prompt(recipe_context),
            tools=tools,
            tool_use_behavior=_tools_to_final_output,
        )

        # Inject schema into query
        augmented_query = f"{schema_ctx}\n\nQuestion: {question}" if schema_ctx else question

        # Run agent with MaxTurnsExceeded handling for partial results
        queries = []
        last_data = None
        turn_info = ""
        try:
            with trace_metadata({"mcp_name": settings.MCP_SLUG, "agent_type": "graphql"}):
                result = await Runner.run(
                    agent,
                    augmented_query,
                    max_turns=settings.MAX_AGENT_TURNS,
                    run_config=get_run_config(),
                )

            queries = _graphql_queries.get()
            last_data = _last_result.get()[0]
            turn_info = get_turn_context(settings.MAX_AGENT_TURNS)

        except MaxTurnsExceeded:
            # Return partial results when turn limit exceeded
            queries = _graphql_queries.get()
            last_data = _last_result.get()[0]
            turn_info = get_turn_context(settings.MAX_AGENT_TURNS)
            return build_partial_result(last_data, queries, turn_info, "queries")

        # Check if tool requested direct return (detected by marker)
        is_direct_return = False
        try:
            is_direct_return = result.final_output == "__DIRECT_RETURN__" or bool(
                _return_directly_flag.get()
            )
        except LookupError:
            pass

        # Early return for error cases (no extraction needed)
        if not result.final_output and not is_direct_return:
            if last_data:
                return {
                    "ok": True,
                    "data": f"[Partial - {turn_info}] Data retrieved but agent didn't complete.",
                    "result": last_data,
                    "queries": queries,
                    "error": None,
                }
            return {
                "ok": False,
                "data": None,
                "result": None,
                "queries": queries,
                "error": f"No output ({turn_info})",
            }

        # Build result for success paths
        if is_direct_return:
            agent_output = None
        else:
            agent_output = str(result.final_output)
            _log(f"DONE queries={len(queries)} output={agent_output[:100]}")

        await maybe_extract_and_save_recipe(
            api_type="graphql",
            api_id=build_api_id(ctx, "graphql"),
            question=question,
            steps=safe_get_contextvar(_recipe_steps, []),
            sql_steps=safe_get_contextvar(_sql_steps, []),
            raw_schema=safe_get_contextvar(_raw_schema, ""),
        )

        return {
            "ok": True,
            "data": agent_output,
            "result": last_data,
            "queries": queries,
            "error": None,
        }

    except Exception as e:
        logger.exception("Agent error")
        return {
            "ok": False,
            "data": None,
            "queries": [],
            "error": str(e),
        }
