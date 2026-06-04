"""REST agent using declarative queries (REST API + DuckDB SQL)."""

import json
import logging
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
from ..recipe.contracts import (
    get_recipe_description,
    get_recipe_tool_args,
    get_recipe_tool_name,
    resolve_recipe_values,
)
from ..recipe.execution import (
    build_rest_call_record,
    collect_step_rows,
    error_json,
    execute_recipe_steps,
    format_recipe_response,
    get_rest_step_call,
    render_rest_call_sets,
    store_step_rows,
    validate_recipe_params,
)
from ..recipe.learning import async_validate_and_prepare_recipe
from ..recipe.search import build_api_id
from ..recipe.state import _set_return_directly, mark_recipe_tool_used
from ..recipe.tooling import build_recipe_docstring, create_params_model, deduplicate_tool_name
from ..rest.client import execute_request
from ..rest.polling import create_poll_tool
from ..rest.schema_loader import fetch_schema_context
from .contextvar_utils import safe_append_contextvar_list, safe_get_contextvar
from .prompts import (
    CONTEXT_SECTION,
    DECISION_GUIDANCE,
    EFFECTIVE_PATTERNS,
    OPTIONAL_PARAMS_SPEC,
    PERSISTENCE_SPEC,
    REASONING_GUIDANCE,
    REST_SCHEMA_NOTATION,
    REST_TOOL_DESC,
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
        logger.info(f"[REST] {msg}")


# Context-local storage (isolated per async request)
# NOTE: Use mutable containers for values that need to be modified by tool functions,
# because ContextVar.set() in child tasks (task groups) doesn't propagate to parent.
_rest_calls: ContextVar[list[dict[str, Any]]] = ContextVar("rest_calls")
_recipe_steps: ContextVar[list[dict[str, Any]]] = ContextVar("recipe_steps")
_query_results: ContextVar[dict[str, Any]] = ContextVar("query_results")
_last_result: ContextVar[list] = ContextVar("last_result")  # Mutable container: [result_value]
_raw_schema: ContextVar[str] = ContextVar("raw_schema")  # Raw OpenAPI JSON for search


def _build_system_prompt(poll_paths: tuple[str, ...] = (), recipe_context: str = "") -> str:
    """Build system prompt for REST agent.

    Args:
        poll_paths: Paths that require polling (empty = no polling support)
        recipe_context: Pre-computed recipe suggestions to inject
    """
    current_date = datetime.now().strftime("%Y-%m-%d")

    poll_tool_desc = ""
    poll_rules = ""
    if poll_paths:
        paths_str = ", ".join(poll_paths)
        poll_tool_desc = f"""
poll_until_done(method, path, done_field, done_value, body?, name?, delay_ms?)
  Poll async API until done_field equals done_value.
  - done_field: dot-path (e.g., "status", "data.0.complete", "trips.0.isCompleted")
  - done_value: target value as string ("true", "COMPLETED")
  - delay_ms: ms between polls (default: {settings.DEFAULT_POLL_DELAY_MS}ms, max: {settings.MAX_POLL_DELAY_MS}ms)
  - Auto-increments polling.count if present in body
  Max {settings.MAX_POLLS} polls. Polling paths: {paths_str}
"""
        poll_rules = f"""
<polling-required>
IMPORTANT: These paths are ASYNC and REQUIRE polling: {paths_str}
- You MUST use poll_until_done (NOT rest_call) for these paths
- rest_call will fail or return incomplete data for polling paths
- Check schema for the completion field (e.g., isCompleted, status, done)
</polling-required>
"""

    # Conditionally add polling example
    poll_example = ""
    if poll_paths:
        poll_example = f"""Polling: poll_until_done("POST", "{poll_paths[0]}", done_field="isCompleted", done_value="true", body='{{...}}')
"""

    workflow_start = "1"

    return f"""You are a REST API agent that answers questions by querying APIs and returning data.

{SQL_RULES}

<tools>
{REST_TOOL_DESC}
{poll_tool_desc}
{SQL_TOOL_DESC}

{SEARCH_TOOL_DESC}
</tools>
<workflow>
{workflow_start}. Read <endpoints> and <schemas> below
{int(workflow_start) + 1}. Check if endpoint is in polling paths - if yes, use poll_until_done; otherwise use rest_call
{int(workflow_start) + 2}. Use sql_query to filter/aggregate results
</workflow>

{CONTEXT_SECTION.format(current_date=current_date, max_turns=settings.MAX_AGENT_TURNS)}

{REASONING_GUIDANCE}

{recipe_context}

{DECISION_GUIDANCE}

{REST_SCHEMA_NOTATION}
{poll_rules}
{UNCERTAINTY_SPEC}

{OPTIONAL_PARAMS_SPEC}

{PERSISTENCE_SPEC.format(max_turns=settings.MAX_AGENT_TURNS)}

{EFFECTIVE_PATTERNS}

{TOOL_USAGE_RULES}

<examples>
GET: rest_call("GET", "/users", query_params='{{"limit": 10}}')
Path param: rest_call("GET", "/users/{{{{id}}}}", path_params='{{"id": "123"}}')
{poll_example}Join: rest_call("GET", "/users", name="u"); rest_call("GET", "/posts", name="p"); sql_query('SELECT u.name, p.title FROM u JOIN p ON u.id = p.userId')
</examples>
"""


def _create_rest_call_tool(ctx: RequestContext, base_url: str):
    """Create rest_call tool with bound context."""

    @function_tool
    async def rest_call(
        method: str,
        path: str,
        path_params: str = "",
        query_params: str = "",
        body: str = "",
        name: str = "data",
        return_directly: bool = False,
    ) -> str:
        """Execute REST API call and store result for sql_query.

        Args:
            method: HTTP method (GET recommended, others may be blocked)
            path: API path (e.g., /users/{id})
            path_params: JSON string for path values (e.g., '{"id": "123"}')
            query_params: JSON string for query params (e.g., '{"limit": 10}')
            body: JSON string for request body (e.g., '{"name": "John"}')
            name: Table name for sql_query (default: "data")
            return_directly: Skip LLM processing, return data directly to client.
                            Only applies on success. Errors still processed by LLM.

        Returns:
            JSON string with API response
        """
        # Parse JSON params
        pp = json.loads(path_params) if path_params else None
        qp = json.loads(query_params) if query_params else None
        bd = json.loads(body) if body else None

        result = await execute_request(
            method,
            path,
            pp,
            qp,
            bd,
            base_url=base_url,
            headers=ctx.target_headers,
            allow_unsafe_paths=list(ctx.allow_unsafe_paths),
        )

        # Track call
        safe_append_contextvar_list(
            _rest_calls,
            {
                "method": method,
                "path": path,
                "path_params": path_params,
                "query_params": query_params,
                "body": body,
                "name": name,
                "success": bool(result.get("success")),
            },
        )

        # Store result for sql_query
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

                # Track successful step for recipe extraction
                safe_append_contextvar_list(
                    _recipe_steps,
                    {
                        "kind": "rest",
                        "name": name,
                        "method": method,
                        "path": path,
                        "path_params": pp,
                        "query_params": qp,
                        "body": bd,
                        "result": stored_data,
                    },
                )
            except LookupError:
                pass

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

        # Add hints on failure to guide agent recovery
        if not result.get("success"):
            status = result.get("status_code", 0)
            # HTTP 4xx/5xx errors - suggest schema search for valid values
            if status >= 400:
                result["hint"] = "Use search_schema to find valid enum values or field names"

        return json.dumps(result, indent=2)

    return rest_call


@function_tool
def sql_query(sql: str, return_directly: bool = False) -> str:
    """Run DuckDB SQL on stored REST API results.

    Tables available = names from rest_call calls + auto-extracted top-level keys.

    Args:
        sql: DuckDB SQL query
        return_directly: Skip LLM processing, return results directly to client

    Returns:
        JSON string with query results
    """
    try:
        data = _query_results.get()
    except LookupError:
        return json.dumps({"success": False, "error": "No data. Call rest_call first."})

    if not data:
        return json.dumps({"success": False, "error": "No data. Call rest_call first."})

    result = execute_sql(data, sql)

    _log(f"SQL {json.dumps(result)[:200]}")

    # Store full result for final response + apply char truncation for LLM
    if result.get("success"):
        rows = result.get("result", [])
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


async def _execute_rest_recipe_step(
    ctx: RequestContext,
    base_url: str,
    allow_unsafe_paths: list[str],
    step: Any,
    params: dict[str, Any],
    results: dict[str, Any],
    *,
    record_call: Callable[[dict[str, Any]], None] | None = None,
    pretty_errors: bool = False,
) -> tuple[bool, Any, str, list[dict[str, Any]] | None]:
    if not isinstance(step, dict) or step.get("kind") != "rest":
        return False, None, error_json("invalid recipe step", pretty=pretty_errors), None

    method, path, name = get_rest_step_call(step)
    rendered_sets, render_error = render_rest_call_sets(step, params, results)
    if render_error:
        return False, None, error_json(render_error, pretty=pretty_errors), None

    combined_rows: list[Any] = []
    call_recs: list[dict[str, Any]] = []
    for rendered in rendered_sets:
        res = await execute_request(
            method,
            path,
            rendered.path_params,
            rendered.query_params,
            rendered.body,
            base_url=base_url,
            headers=ctx.target_headers,
            allow_unsafe_paths=allow_unsafe_paths,
        )
        if not res.get("success"):
            return (
                False,
                None,
                error_json(res.get("error", "request failed"), pretty=pretty_errors),
                None,
            )

        combined_rows.extend(collect_step_rows(res.get("data", {}), step, rendered.binding))
        call_rec = build_rest_call_record(method=method, path=path, name=name, rendered=rendered)
        call_recs.append(call_rec)
        if record_call:
            record_call(call_rec)

    store_step_rows(results, step, combined_rows)
    return True, combined_rows, "", call_recs


def _create_individual_recipe_tools(
    ctx: RequestContext,
    base_url: str,
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

                async def rest_step_executor(step_idx, step, params, results):
                    _ = step_idx
                    success, data, step_error, call_recs = await _execute_rest_recipe_step(
                        ctx,
                        base_url,
                        list(ctx.allow_unsafe_paths),
                        step,
                        params,
                        results,
                        record_call=lambda call_rec: safe_append_contextvar_list(
                            _rest_calls, call_rec
                        ),
                        pretty_errors=True,
                    )
                    _query_results.set(results)
                    return success, data, step_error, call_recs

                executed_calls: list[dict[str, Any]] = []
                success, _last_data, executed_sql, error = await execute_recipe_steps(
                    recipe,
                    execution_params or {},
                    _query_results,
                    _last_result,
                    rest_step_executor,
                    executed_calls,
                )
                if not success:
                    return error

                if return_directly:
                    _set_return_directly()

                return format_recipe_response(
                    _last_result,
                    executed_calls,
                    executed_sql,
                    "executed_calls",
                )

            dynamic_recipe_tool.__name__ = tname
            dynamic_recipe_tool.__doc__ = doc
            return function_tool(dynamic_recipe_tool)

        tools.append(make_tool(s["recipe_id"], params_spec, docstring, tool_name))

    return tools


# Create search_schema tool bound to REST schema context var
search_schema = create_search_schema_tool(_raw_schema)


async def _load_rest_schema(ctx: RequestContext) -> LoadedSchema:
    schema_ctx, spec_base_url, raw_spec_json = await fetch_schema_context(
        ctx.target_url, ctx.target_headers
    )
    base_url = ctx.base_url or spec_base_url
    if not base_url:
        return LoadedSchema(
            schema_context=schema_ctx,
            raw_schema=raw_spec_json,
            early_response={
                "ok": False,
                "data": None,
                "api_calls": [],
                "error": "Could not determine base URL. Set X-Base-URL header or ensure spec has 'servers' field.",
            },
        )
    return LoadedSchema(schema_context=schema_ctx, raw_schema=raw_spec_json, base_url=base_url)


def _build_rest_tools(ctx: RequestContext, state: AgentRuntimeState) -> list[Any]:
    tools = [_create_rest_call_tool(ctx, state.base_url), sql_query, search_schema]
    if ctx.poll_paths:
        tools.insert(
            1,
            create_poll_tool(
                ctx,
                state.base_url,
                rest_calls_var=_rest_calls,
                query_results_var=_query_results,
                last_result_var=_last_result,
            ),
        )
    if state.suggestions:
        return [*_create_individual_recipe_tools(ctx, state.base_url, state.suggestions), *tools]
    return tools


async def _validate_rest_recipe_candidate(
    ctx: RequestContext,
    state: AgentRuntimeState,
    recipe: dict[str, Any],
    tool_args: dict[str, Any],
) -> Any:
    execution_params, error = resolve_recipe_values(recipe, tool_args)
    if error:
        return None

    query_results_var: ContextVar[dict[str, Any]] = ContextVar("rest_recipe_validation_results")
    last_result_var: ContextVar[list[Any]] = ContextVar("rest_recipe_validation_last")
    query_results_var.set({})
    last_result_var.set([None])

    async def rest_step_executor(step_idx, step, params, results):
        _ = step_idx
        success, data, step_error, call_recs = await _execute_rest_recipe_step(
            ctx,
            state.base_url,
            [],
            step,
            params,
            results,
        )
        query_results_var.set(results)
        return success, data, step_error, call_recs

    success, last_data, _executed_sql, _error = await execute_recipe_steps(
        recipe,
        execution_params or {},
        query_results_var,
        last_result_var,
        rest_step_executor,
        [],
    )
    return last_data if success else None


def _build_rest_prompt(ctx: RequestContext, state: AgentRuntimeState) -> str:
    return _build_system_prompt(poll_paths=ctx.poll_paths, recipe_context=state.recipe_context)


def _rest_api_id(ctx: RequestContext, state: AgentRuntimeState) -> str:
    return build_api_id(ctx, "rest", state.base_url)


def _skip_polling_recipe(_ctx: RequestContext, _state: AgentRuntimeState) -> bool:
    return any("poll_attempt" in c for c in safe_get_contextvar(_rest_calls, []))


_REST_RUNTIME = AgentRuntimeConfig(
    agent_name="rest-agent",
    agent_type="rest",
    call_key="api_calls",
    calls_var=_rest_calls,
    recipe_steps_var=_recipe_steps,
    query_results_var=_query_results,
    last_result_var=_last_result,
    raw_schema_var=_raw_schema,
    load_schema=_load_rest_schema,
    build_tools=_build_rest_tools,
    build_prompt=_build_rest_prompt,
    build_api_id=_rest_api_id,
    log=_log,
    done_log_label="calls",
    exception_message="REST Agent error",
    skip_recipe=_skip_polling_recipe,
    validate_recipe_candidate=_validate_rest_recipe_candidate,
)


async def process_rest_query(question: str, ctx: RequestContext) -> dict[str, Any]:
    """Process natural language query against REST API."""
    return await run_agent_query(question, ctx, _REST_RUNTIME)
