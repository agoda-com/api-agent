"""GraphQL agent using declarative queries (GraphQL + DuckDB SQL)."""

import json
import logging
import re
from collections.abc import Callable
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from agents import function_tool

from ..config import settings
from ..context import RequestContext
from ..executor import (
    execute_sql,
    extract_tables_from_response,
    truncate_for_context,
)
from ..graphql import execute_query as graphql_fetch
from ..graphql.schema_context import build_schema_context
from ..recipe.contracts import (
    get_recipe_description,
    get_recipe_tool_args,
    get_recipe_tool_name,
    resolve_recipe_values,
)
from ..recipe.execution import (
    collect_step_rows,
    error_json,
    execute_recipe_steps,
    format_recipe_response,
    render_graphql_query_sets,
    store_step_rows,
    validate_recipe_params,
)
from ..recipe.learning import async_validate_and_prepare_recipe
from ..recipe.search import build_api_id
from ..recipe.state import _set_return_directly, mark_recipe_tool_used
from ..recipe.tooling import build_recipe_docstring, create_params_model, deduplicate_tool_name
from .contextvar_utils import safe_append_contextvar_list, safe_get_contextvar
from .prompts import (
    CONTEXT_SECTION,
    DECISION_GUIDANCE,
    EFFECTIVE_PATTERNS,
    GRAPHQL_SCHEMA_NOTATION,
    OPTIONAL_PARAMS_SPEC,
    PERSISTENCE_SPEC,
    REASONING_GUIDANCE,
    SEARCH_TOOL_DESC,
    SQL_RULES,
    SQL_TOOL_DESC,
    TOOL_USAGE_RULES,
    UNCERTAINTY_SPEC,
)
from .runtime import AgentRuntimeConfig, AgentRuntimeState, LoadedSchema, run_agent_query
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
    context = build_schema_context(schema)

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

{REASONING_GUIDANCE}

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
                _recipe_steps,
                {"kind": "graphql", "query": query, "name": name, "result": stored_data},
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
        safe_append_contextvar_list(_recipe_steps, {"kind": "sql", "query": sql, "result": rows})

        if return_directly:
            _set_return_directly()

        if isinstance(rows, list):
            return json.dumps(
                {"success": True, **truncate_for_context(rows, "sql_result")},
                indent=2,
            )

    return json.dumps(result, indent=2)


async def _execute_graphql_recipe_step(
    ctx: RequestContext,
    step: Any,
    params: dict[str, Any],
    results: dict[str, Any],
    *,
    record_query: Callable[[str], None] | None = None,
    pretty_errors: bool = False,
) -> tuple[bool, Any, str, list[str] | None]:
    if not isinstance(step, dict) or step.get("kind") != "graphql":
        return False, None, error_json("invalid recipe step", pretty=pretty_errors), None

    rendered_queries, render_error = render_graphql_query_sets(step, params, results)
    if render_error:
        return False, None, error_json(render_error, pretty=pretty_errors), None

    combined_rows: list[Any] = []
    queries: list[str] = []
    for rendered in rendered_queries:
        query = rendered.query
        res = await graphql_fetch(query, None, ctx.target_url, ctx.target_headers)
        if not res.get("success"):
            return (
                False,
                None,
                error_json(res.get("error", "query failed"), pretty=pretty_errors),
                None,
            )

        combined_rows.extend(collect_step_rows(res.get("data", {}), step, rendered.binding))
        queries.append(query)
        if record_query:
            record_query(query)

    store_step_rows(results, step, combined_rows)
    return True, combined_rows, "", queries


def _create_individual_recipe_tools(
    ctx: RequestContext,
    suggestions: list[dict[str, Any]],
) -> list:
    """Generate one function_tool per recipe suggestion."""
    tools = []
    seen_names: set[str] = set()

    for s in suggestions:
        recipe = s.get("recipe")
        if not isinstance(recipe, dict):
            continue

        tool_name = deduplicate_tool_name(get_recipe_tool_name(recipe), seen_names)
        params_spec = get_recipe_tool_args(recipe)
        docstring = build_recipe_docstring(
            s["question"],
            [],
            api_type="graphql",
            params_spec=params_spec,
            description=get_recipe_description(recipe),
        )

        def make_tool(rid: str, pspec: dict[str, Any], doc: str, tname: str):
            ParamsModel = create_params_model(pspec, tname)

            async def dynamic_recipe_tool(
                params: ParamsModel,  # ty: ignore[invalid-type-form]
                return_directly: bool = True,
            ) -> str:
                mark_recipe_tool_used(rid)
                kwargs = params.model_dump()
                validated_params, error = validate_recipe_params(pspec, kwargs)
                if error:
                    return error

                recipe, validated_params, error = await async_validate_and_prepare_recipe(
                    rid, json.dumps(kwargs), _raw_schema
                )
                if error:
                    return error
                assert recipe is not None
                execution_params, error = resolve_recipe_values(recipe, validated_params or {})
                if error:
                    return error_json(error, pretty=False)

                async def graphql_step_executor(step_idx, step, params, results):
                    _ = step_idx
                    success, data, step_error, queries = await _execute_graphql_recipe_step(
                        ctx,
                        step,
                        params,
                        results,
                        record_query=lambda query: safe_append_contextvar_list(
                            _graphql_queries, query
                        ),
                        pretty_errors=True,
                    )
                    _query_results.set(results)
                    return success, data, step_error, queries

                executed_queries: list[str] = []
                success, _last_data, executed_sql, error = await execute_recipe_steps(
                    recipe,
                    execution_params or {},
                    _query_results,
                    _last_result,
                    graphql_step_executor,
                    executed_queries,
                )
                if not success:
                    return error

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


async def _load_graphql_schema(ctx: RequestContext) -> LoadedSchema:
    schema_ctx = await _fetch_schema_context(ctx.target_url, ctx.target_headers)
    return LoadedSchema(
        schema_context=schema_ctx,
        raw_schema=safe_get_contextvar(_raw_schema, ""),
    )


def _build_graphql_tools(ctx: RequestContext, state: AgentRuntimeState) -> list[Any]:
    tools = [_create_graphql_query_tool(ctx), sql_query, search_schema]
    if state.suggestions:
        return [*_create_individual_recipe_tools(ctx, state.suggestions), *tools]
    return tools


async def _validate_graphql_recipe_candidate(
    ctx: RequestContext,
    _state: AgentRuntimeState,
    recipe: dict[str, Any],
    tool_args: dict[str, Any],
) -> Any:
    execution_params, error = resolve_recipe_values(recipe, tool_args)
    if error:
        return None

    query_results_var: ContextVar[dict[str, Any]] = ContextVar("graphql_recipe_validation_results")
    last_result_var: ContextVar[list[Any]] = ContextVar("graphql_recipe_validation_last")
    query_results_var.set({})
    last_result_var.set([None])

    async def graphql_step_executor(step_idx, step, params, results):
        _ = step_idx
        success, data, step_error, queries = await _execute_graphql_recipe_step(
            ctx,
            step,
            params,
            results,
        )
        query_results_var.set(results)
        return success, data, step_error, queries

    success, last_data, _executed_sql, _error = await execute_recipe_steps(
        recipe,
        execution_params or {},
        query_results_var,
        last_result_var,
        graphql_step_executor,
        [],
    )
    return last_data if success else None


def _build_graphql_prompt(_ctx: RequestContext, state: AgentRuntimeState) -> str:
    return _build_system_prompt(state.recipe_context)


def _graphql_api_id(ctx: RequestContext, _state: AgentRuntimeState) -> str:
    return build_api_id(ctx, "graphql")


_GRAPHQL_RUNTIME = AgentRuntimeConfig(
    agent_name="graphql-agent",
    agent_type="graphql",
    call_key="queries",
    calls_var=_graphql_queries,
    recipe_steps_var=_recipe_steps,
    query_results_var=_query_results,
    last_result_var=_last_result,
    raw_schema_var=_raw_schema,
    load_schema=_load_graphql_schema,
    build_tools=_build_graphql_tools,
    build_prompt=_build_graphql_prompt,
    build_api_id=_graphql_api_id,
    log=_log,
    done_log_label="queries",
    exception_message="Agent error",
    validate_recipe_candidate=_validate_graphql_recipe_candidate,
)


async def process_query(question: str, ctx: RequestContext) -> dict[str, Any]:
    """Process natural language query against GraphQL API."""
    return await run_agent_query(question, ctx, _GRAPHQL_RUNTIME)
