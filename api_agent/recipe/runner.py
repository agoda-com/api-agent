"""Recipe execution outside agent context (for MCP recipe tools)."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from ..agent.graphql_agent import fetch_graphql_schema_raw
from ..context import RequestContext
from ..graphql import execute_query as graphql_execute
from ..rest.client import execute_request
from ..rest.schema_loader import fetch_schema_context
from ..store import ASYNC_API_AGENT_STORE, sha256_hex
from ..utils.csv import to_csv
from .contracts import get_recipe_tool_args, has_recipe_contract, resolve_recipe_values
from .execution import (
    build_rest_call_record,
    collect_step_rows,
    error_json,
    execute_recipe_steps,
    format_recipe_response,
    get_rest_step_call,
    render_graphql_query_sets,
    render_rest_call_sets,
    store_step_rows,
    validate_recipe_params,
)
from .search import build_api_id


async def load_schema_and_base_url(ctx: RequestContext) -> tuple[str, str]:
    """Load raw schema and base URL (REST). Returns (raw_schema, base_url)."""
    if ctx.api_type == "graphql":
        raw_schema = await fetch_graphql_schema_raw(ctx.target_url, ctx.target_headers)
        return raw_schema, ""

    _, spec_base_url, raw_spec_json = await fetch_schema_context(ctx.target_url, ctx.target_headers)
    base_url = ctx.base_url or spec_base_url or ""
    return raw_spec_json, base_url


async def execute_recipe_tool(
    ctx: RequestContext,
    recipe_id: str,
    params: dict[str, Any] | None,
    return_directly: bool = True,
    *,
    raw_schema: str = "",
    base_url: str = "",
) -> str:
    """Execute a recipe by id and return JSON string."""
    if not raw_schema:
        raw_schema, base_url = await load_schema_and_base_url(ctx)
    if not raw_schema:
        return error_json("schema not loaded")

    meta = await ASYNC_API_AGENT_STORE.get_recipe_meta(recipe_id)
    if not meta:
        return error_json(f"recipe not found: {recipe_id}")

    schema_hash = sha256_hex(raw_schema)
    api_id = build_api_id(ctx, ctx.api_type, base_url)
    if meta.get("schema_hash") != schema_hash or meta.get("api_id") != api_id:
        return error_json("recipe does not match current API or schema")

    recipe = meta.get("recipe") or {}
    if not has_recipe_contract(recipe):
        return error_json(f"recipe not found: {recipe_id}")
    params_spec = get_recipe_tool_args(recipe)
    provided = params or {}
    validated_public_args, error = validate_recipe_params(params_spec, provided)
    if error:
        return error
    execution_params, error = resolve_recipe_values(recipe, validated_public_args or {})
    if error:
        return error_json(error)

    # Initialize storage for results
    query_results_var: ContextVar[dict[str, Any]] = ContextVar("recipe_query_results")
    last_result_var: ContextVar[list[Any]] = ContextVar("recipe_last_result")
    query_results_var.set({})
    last_result_var.set([None])

    if ctx.api_type == "graphql":
        executed_queries: list[str] = []

        async def graphql_step_executor(step_idx, step, params, results):
            if not isinstance(step, dict) or step.get("kind") != "graphql":
                return False, None, error_json("invalid recipe step"), None

            rendered_queries, render_error = render_graphql_query_sets(step, params, results)
            if render_error:
                return False, None, error_json(render_error), None

            combined_rows: list[Any] = []
            queries: list[str] = []
            for rendered in rendered_queries:
                query = rendered.query
                res = await graphql_execute(query, None, ctx.target_url, ctx.target_headers)
                if not res.get("success"):
                    return False, None, error_json(res.get("error", "query failed")), None

                combined_rows.extend(collect_step_rows(res.get("data", {}), step, rendered.binding))
                queries.append(query)

            store_step_rows(results, step, combined_rows)
            query_results_var.set(results)
            return True, combined_rows, "", queries

        success, last_data, executed_sql, error = await execute_recipe_steps(
            recipe,
            execution_params or {},
            query_results_var,
            last_result_var,
            graphql_step_executor,
            executed_queries,
        )
        if not success:
            return error

        if return_directly:
            return to_csv(last_data)
        return format_recipe_response(
            last_result_var, executed_queries, executed_sql, "executed_queries"
        )

    # REST execution
    if not base_url:
        return error_json("Could not determine base URL for REST API")

    executed_calls: list[dict[str, Any]] = []

    async def rest_step_executor(step_idx, step, params, results):
        if not isinstance(step, dict) or step.get("kind") != "rest":
            return False, None, error_json("invalid recipe step"), None

        method, path, name = get_rest_step_call(step)

        rendered_sets, render_error = render_rest_call_sets(step, params, results)
        if render_error:
            return False, None, error_json(render_error), None

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
                allow_unsafe_paths=list(ctx.allow_unsafe_paths),
            )
            if not res.get("success"):
                return False, None, error_json(res.get("error", "request failed")), None

            combined_rows.extend(collect_step_rows(res.get("data", {}), step, rendered.binding))

            call_recs.append(
                build_rest_call_record(method=method, path=path, name=name, rendered=rendered)
            )

        store_step_rows(results, step, combined_rows)
        query_results_var.set(results)
        return True, combined_rows, "", call_recs

    success, last_data, executed_sql, error = await execute_recipe_steps(
        recipe,
        execution_params or {},
        query_results_var,
        last_result_var,
        rest_step_executor,
        executed_calls,
    )
    if not success:
        return error

    if return_directly:
        return to_csv(last_data)
    return format_recipe_response(last_result_var, executed_calls, executed_sql, "executed_calls")
