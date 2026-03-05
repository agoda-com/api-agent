"""REST agent using declarative queries (REST API + DuckDB SQL)."""

import asyncio
import json
import logging
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
    render_param_refs,
    search_recipes,
    validate_and_prepare_recipe,
    validate_recipe_params,
)
from ..rest.client import execute_request
from ..rest.schema_loader import fetch_schema_context
from ..tracing import trace_metadata
from .contextvar_utils import safe_append_contextvar_list, safe_get_contextvar
from .model import get_run_config, model
from .progress import get_turn_context, reset_progress
from .prompts import (
    CONTEXT_SECTION,
    DECISION_GUIDANCE,
    EFFECTIVE_PATTERNS,
    OPTIONAL_PARAMS_SPEC,
    PERSISTENCE_SPEC,
    REST_SCHEMA_NOTATION,
    REST_TOOL_DESC,
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
        logger.info(f"[REST] {msg}")


# Context-local storage (isolated per async request)
# NOTE: Use mutable containers for values that need to be modified by tool functions,
# because ContextVar.set() in child tasks (task groups) doesn't propagate to parent.
_rest_calls: ContextVar[list[dict[str, Any]]] = ContextVar("rest_calls")
_recipe_steps: ContextVar[list[dict[str, Any]]] = ContextVar("recipe_steps")
_query_results: ContextVar[dict[str, Any]] = ContextVar("query_results")
_last_result: ContextVar[list] = ContextVar("last_result")  # Mutable container: [result_value]
_raw_schema: ContextVar[str] = ContextVar("raw_schema")  # Raw OpenAPI JSON for search
_sql_steps: ContextVar[list[str]] = ContextVar("sql_steps")


def _get_nested_value(data: dict | None, path: str) -> Any:
    """Extract value from nested dict/list using dot notation.

    Args:
        data: Dictionary to extract from
        path: Dot-separated path (e.g., "polling.completed", "trips.0.isCompleted")

    Returns:
        Value at path or None if not found
    """
    if not data or not path:
        return None
    keys = path.split(".")
    current: Any = data
    for key in keys:
        if not isinstance(current, (dict, list)):
            return None
        if isinstance(current, list) and key.isdigit():
            idx = int(key)
            if 0 <= idx < len(current):
                current = current[idx]
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(key)
        else:
            return None
        if current is None:
            return None
    return current


def _set_nested_value(data: dict, path: str, value: Any) -> None:
    """Set value in nested dict using dot notation, creating intermediate dicts.

    Args:
        data: Dictionary to modify
        path: Dot-separated path (e.g., "polling.count")
        value: Value to set
    """
    if not path:
        return
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


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
  - delay_ms: ms between polls (default: {settings.DEFAULT_POLL_DELAY_MS}ms)
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


def _create_poll_tool(ctx: RequestContext, base_url: str):
    """Create poll_until_done tool with bound context."""

    @function_tool
    async def poll_until_done(
        method: str,
        path: str,
        done_field: str,
        done_value: str,
        body: str = "",
        path_params: str = "",
        query_params: str = "",
        name: str = "poll_result",
        delay_ms: int = 0,
    ) -> str:
        """Poll endpoint until done_field equals done_value. Auto-increments polling.count if present.

        Args:
            method: HTTP method (POST typically)
            path: API path
            done_field: Dot-path to check (e.g., "status", "polling.completed", "trips.0.isCompleted")
            done_value: Value indicating done (e.g., "true", "0", "COMPLETED", "100")
            body: JSON string request body
            path_params: JSON string for path values
            query_params: JSON string for query params
            name: Table name for sql_query (default: poll_result)
            delay_ms: Delay between polls in ms (default: 3000ms)

        Returns:
            JSON string with final response or error
        """
        pp = json.loads(path_params) if path_params else None
        qp = json.loads(query_params) if query_params else None
        try:
            body_dict = json.loads(body) if body else {}
        except json.JSONDecodeError as e:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Invalid body JSON: {e.msg}",
                }
            )

        # Internal defaults from config
        max_polls = settings.MAX_POLLS
        wait_ms = delay_ms if delay_ms > 0 else settings.DEFAULT_POLL_DELAY_MS
        current = None  # Track last done_field value for error messages

        attempt = 0
        while attempt < max_polls:
            attempt += 1

            result = await execute_request(
                method,
                path,
                pp,
                qp,
                body=body_dict if body_dict else None,
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
                    "body": json.dumps(body_dict) if body_dict else "",
                    "name": name,
                    "poll_attempt": attempt,
                    "success": bool(result.get("success")),
                },
            )

            if not result.get("success"):
                return json.dumps(
                    {
                        "success": False,
                        "error": result.get("error"),
                        "attempt": attempt,
                    }
                )

            data = result.get("data", {})

            # Validate done_field exists on first response
            current = _get_nested_value(data, done_field)
            if current is None and attempt == 1:
                keys = list(data.keys()) if isinstance(data, dict) else []
                return json.dumps(
                    {
                        "success": False,
                        "error": f"done_field '{done_field}' not found in response. Available keys: {keys}",
                    }
                )

            # Check if done_field value matches done_value (string comparison)
            is_done = str(current).lower() == done_value.lower()

            if is_done:
                # Store result for sql_query
                try:
                    results = _query_results.get()
                    tables, _ = extract_tables_from_response(data, name)
                    results.update(tables)
                    stored = tables.get(name)
                    if stored is not None:
                        _last_result.get()[0] = stored
                except LookupError:
                    pass

                return json.dumps(
                    {
                        "success": True,
                        **truncate_for_context(data if isinstance(data, list) else [data], name),
                        "attempts": attempt,
                    },
                    indent=2,
                )

            await asyncio.sleep(wait_ms / 1000)

            # Auto-increment polling.count if present in body
            if body_dict.get("polling", {}).get("count") is not None:
                body_dict["polling"]["count"] += 1

        return json.dumps(
            {
                "success": False,
                "error": f"max_polls ({max_polls}) exceeded. Last {done_field} value: {current} (expected: {done_value})",
                "attempts": attempt,
            }
        )

    return poll_until_done


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
    base_url: str,
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

                async def rest_step_executor(step_idx, step, params, results):
                    if not isinstance(step, dict) or step.get("kind") != "rest":
                        return (
                            False,
                            None,
                            json.dumps(
                                {"success": False, "error": "invalid recipe step"}, indent=2
                            ),
                            None,
                        )

                    method = str(step.get("method", "GET")).upper()
                    path = str(step.get("path", ""))
                    name = str(step.get("name") or "data")

                    pp = render_param_refs(step.get("path_params") or {}, params)
                    qp = render_param_refs(step.get("query_params") or {}, params)
                    bd = render_param_refs(step.get("body") or {}, params)

                    res = await execute_request(
                        method,
                        path,
                        pp if isinstance(pp, dict) else None,
                        qp if isinstance(qp, dict) else None,
                        bd if isinstance(bd, dict) and bd else None,
                        base_url=base_url,
                        headers=ctx.target_headers,
                        allow_unsafe_paths=list(ctx.allow_unsafe_paths),
                    )
                    if not res.get("success"):
                        return (
                            False,
                            None,
                            json.dumps(
                                {"success": False, "error": res.get("error", "request failed")},
                                indent=2,
                            ),
                            None,
                        )

                    data = res.get("data", {})
                    tables, _ = extract_tables_from_response(data, name)
                    results.update(tables)
                    _query_results.set(results)

                    call_rec = {
                        "method": method,
                        "path": path,
                        "path_params": json.dumps(pp) if pp else "",
                        "query_params": json.dumps(qp) if qp else "",
                        "body": json.dumps(bd) if bd else "",
                        "name": name,
                        "success": True,
                    }
                    safe_append_contextvar_list(_rest_calls, call_rec)
                    return True, tables.get(name), "", call_rec

                executed_calls: list[dict[str, Any]] = []
                if recipe is None or validated_params is None:
                    return json.dumps({"success": False, "error": "recipe validation failed"})
                success, last_data, executed_sql, error = await execute_recipe_steps(
                    recipe,
                    validated_params,
                    _query_results,
                    _last_result,
                    rest_step_executor,
                    executed_calls,
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


async def process_rest_query(question: str, ctx: RequestContext) -> dict[str, Any]:
    """Process natural language query against REST API.

    Args:
        question: Natural language question
        ctx: Request context with target_url (OpenAPI spec) and target_headers
    """
    try:
        _log(f"QUERY {question[:80]}")

        # Reset per-request storage
        _rest_calls.set([])
        _recipe_steps.set([])
        _sql_steps.set([])
        _query_results.set({})
        _last_result.set([None])  # Mutable list: [result_value]
        _return_directly_flag.set([])  # Reset direct return flag
        reset_progress()  # Reset turn counter

        # Fetch schema context (target_url = OpenAPI spec URL)
        schema_ctx, spec_base_url, raw_spec_json = await fetch_schema_context(
            ctx.target_url, ctx.target_headers
        )

        # Store raw OpenAPI spec for search_schema tool
        _raw_schema.set(raw_spec_json)

        # Use header override or spec-derived base URL
        base_url = ctx.base_url or spec_base_url
        if not base_url:
            return {
                "ok": False,
                "data": None,
                "api_calls": [],
                "error": "Could not determine base URL. Set X-Base-URL header or ensure spec has 'servers' field.",
            }

        # Pre-flight recipe search
        suggestions, recipe_context = [], ""
        if settings.ENABLE_RECIPES:
            raw_schema = safe_get_contextvar(_raw_schema, "")
            api_id = build_api_id(ctx, "rest", base_url)
            suggestions, recipe_context = search_recipes(api_id, raw_schema, question)
            if suggestions:
                _log(
                    f"PRE-FLIGHT found={len(suggestions)} ids={[s['recipe_id'] for s in suggestions]}"
                )
            elif raw_schema:
                _log(f"PRE-FLIGHT no matches for api_id={api_id[:50]}")

        # Create tools with bound context
        rest_tool = _create_rest_call_tool(ctx, base_url)

        # Only include poll tool if user specified poll_paths header
        include_polling = bool(ctx.poll_paths)
        tools = [rest_tool, sql_query, search_schema]
        if include_polling:
            poll_tool = _create_poll_tool(ctx, base_url)
            tools.insert(1, poll_tool)
        if suggestions:  # Create individual recipe tools for each suggestion
            recipe_tools = _create_individual_recipe_tools(ctx, base_url, suggestions)
            tools = [*recipe_tools, *tools]

        # Create fresh agent with dynamic tools
        agent = Agent(
            name="rest-agent",
            model=model,
            instructions=_build_system_prompt(
                poll_paths=ctx.poll_paths, recipe_context=recipe_context
            ),
            tools=tools,
            tool_use_behavior=_tools_to_final_output,
        )

        # Inject schema into query
        augmented_query = f"{schema_ctx}\n\nQuestion: {question}" if schema_ctx else question

        # Run agent with MaxTurnsExceeded handling for partial results
        api_calls = []
        last_data = None
        turn_info = ""
        try:
            with trace_metadata({"mcp_name": settings.MCP_SLUG, "agent_type": "rest"}):
                result = await Runner.run(
                    agent,
                    augmented_query,
                    max_turns=settings.MAX_AGENT_TURNS,
                    run_config=get_run_config(),
                )

            api_calls = _rest_calls.get()
            last_data = _last_result.get()[0]
            turn_info = get_turn_context(settings.MAX_AGENT_TURNS)

        except MaxTurnsExceeded:
            # Return partial results when turn limit exceeded
            api_calls = _rest_calls.get()
            last_data = _last_result.get()[0]
            turn_info = get_turn_context(settings.MAX_AGENT_TURNS)
            return build_partial_result(last_data, api_calls, turn_info, "api_calls")

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
                    "api_calls": api_calls,
                    "error": None,
                }
            return {
                "ok": False,
                "data": None,
                "result": None,
                "api_calls": api_calls,
                "error": f"No output ({turn_info})",
            }

        # Build result for success paths
        if is_direct_return:
            agent_output = None
        else:
            agent_output = str(result.final_output)
            _log(f"DONE calls={len(api_calls)} output={agent_output[:100]}")

        # Skip polling recipes (v1)
        skip_polling = any("poll_attempt" in c for c in safe_get_contextvar(_rest_calls, []))
        await maybe_extract_and_save_recipe(
            api_type="rest",
            api_id=build_api_id(ctx, "rest", base_url),
            question=question,
            steps=safe_get_contextvar(_recipe_steps, []),
            sql_steps=safe_get_contextvar(_sql_steps, []),
            raw_schema=safe_get_contextvar(_raw_schema, ""),
            skip_condition=skip_polling,
        )

        return {
            "ok": True,
            "data": agent_output,
            "result": last_data,
            "api_calls": api_calls,
            "error": None,
        }

    except Exception as e:
        logger.exception("REST Agent error")
        return {
            "ok": False,
            "data": None,
            "api_calls": [],
            "error": str(e),
        }
