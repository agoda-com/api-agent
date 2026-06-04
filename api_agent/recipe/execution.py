"""Recipe validation and execution helpers."""

from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from ..executor import execute_sql, extract_tables_from_response, truncate_for_context
from .contracts import (
    get_recipe_steps,
    get_step_call,
    get_step_output,
    get_step_output_name,
    resolve_step_input_values,
)
from .templates import render_param_refs, render_text_template
from .tooling import create_params_model


def error_json(msg: Any, *, pretty: bool = True) -> str:
    """Build JSON error response."""
    return json.dumps({"success": False, "error": msg}, indent=2 if pretty else None)


def validate_recipe_params(
    params_spec: dict[str, Any],
    provided: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Validate public recipe tool args."""
    extra = set(provided.keys()) - set(params_spec.keys())
    if extra:
        return None, error_json(f"unexpected params: {', '.join(sorted(extra))}")

    for pname in params_spec:
        if pname not in provided:
            return None, error_json(f"missing required param: {pname}")

    ParamsModel = create_params_model(params_spec, "RecipeParams")
    try:
        model = ParamsModel.model_validate(provided)
    except PydanticValidationError as e:
        err = e.errors(include_url=False)[0] if e.errors() else {}
        loc = err.get("loc") or ("params",)
        pname = str(loc[0]) if loc else "params"
        expected = "value"
        spec = params_spec.get(pname)
        if isinstance(spec, dict):
            expected = str(spec.get("type") or expected)
        return None, error_json(f"invalid param type: {pname} must be {expected}")

    return model.model_dump(), ""


def _get_results_context(query_results_var: ContextVar[dict[str, Any]]) -> dict[str, Any]:
    """Get or create results dict from ContextVar."""
    try:
        return query_results_var.get()
    except LookupError:
        results: dict[str, Any] = {}
        query_results_var.set(results)
        return results


def _execute_sql_step(
    step: dict[str, Any],
    params: dict[str, Any],
    results: dict[str, Any],
    last_result_var: ContextVar[list[Any]],
) -> tuple[bool, str, str]:
    sql_tmpl = step.get("query_template")
    if not isinstance(sql_tmpl, str):
        return False, "", error_json("invalid sql step")

    input_sets, input_error = build_step_input_sets(step, params, results)
    if input_error:
        return False, "", error_json(input_error)
    if not input_sets:
        return False, "", error_json("sql step has no input")

    sql = render_text_template(sql_tmpl, input_sets[0].params)
    res = execute_sql(results, sql)
    if not res.get("success"):
        return False, sql, json.dumps(res, indent=2)

    name = get_step_output_name(step)
    if name:
        result = res.get("result")
        if isinstance(result, list):
            results[name] = result

    _set_last_result(last_result_var, res.get("result", []))
    return True, sql, ""


def _set_last_result(last_result_var: ContextVar[list[Any]], data: Any) -> None:
    try:
        last_result_var.get()[0] = data
    except LookupError:
        pass


@dataclass(frozen=True)
class StepInputSet:
    params: dict[str, Any]
    binding: dict[str, Any]


@dataclass(frozen=True)
class RenderedRestCall:
    path_params: dict[str, Any] | None
    query_params: dict[str, Any] | None
    body: Any
    binding: dict[str, Any]


@dataclass(frozen=True)
class RenderedGraphQLQuery:
    query: str
    binding: dict[str, Any]


def build_step_input_sets(
    step: dict[str, Any],
    params: dict[str, Any],
    results: dict[str, Any],
) -> tuple[list[StepInputSet], str]:
    """Resolve one step's scalar and row bindings."""
    raw_sets, error = resolve_step_input_values(step, params, results)
    if error:
        return [], error
    assert raw_sets is not None
    return [StepInputSet(values, binding) for values, binding in raw_sets], ""


def get_step_result_name(step: dict[str, Any]) -> str:
    return get_step_output_name(step) or "data"


def render_graphql_query_sets(
    step: dict[str, Any],
    params: dict[str, Any],
    results: dict[str, Any],
) -> tuple[list[RenderedGraphQLQuery], str]:
    tmpl = get_step_call(step).get("query_template")
    if not isinstance(tmpl, str):
        return [], "missing query_template"

    input_sets, input_error = build_step_input_sets(step, params, results)
    if input_error:
        return [], input_error

    rendered: list[RenderedGraphQLQuery] = []
    for input_set in input_sets:
        try:
            query = render_text_template(tmpl, input_set.params)
        except KeyError as e:
            return [], str(e)
        rendered.append(RenderedGraphQLQuery(query=query, binding=input_set.binding))
    return rendered, ""


def render_rest_call_sets(
    step: dict[str, Any],
    params: dict[str, Any],
    results: dict[str, Any],
) -> tuple[list[RenderedRestCall], str]:
    """Render REST call params from explicit step bindings."""
    input_sets, input_error = build_step_input_sets(step, params, results)
    if input_error:
        return [], input_error

    rendered: list[RenderedRestCall] = []
    call = get_step_call(step)
    for input_set in input_sets:
        try:
            pp = render_param_refs(call.get("path_params") or {}, input_set.params)
            qp = render_param_refs(call.get("query_params") or {}, input_set.params)
            bd = render_param_refs(call.get("body") or {}, input_set.params)
        except KeyError as e:
            return [], str(e)

        rendered.append(
            RenderedRestCall(
                path_params=pp if isinstance(pp, dict) else None,
                query_params=qp if isinstance(qp, dict) else None,
                body=bd if bd else None,
                binding=input_set.binding,
            )
        )
    return rendered, ""


def get_rest_step_call(step: dict[str, Any]) -> tuple[str, str, str]:
    call = get_step_call(step)
    method = str(call.get("method", "GET")).upper()
    path = str(call.get("path", ""))
    return method, path, get_step_result_name(step)


def build_rest_call_record(
    *,
    method: str,
    path: str,
    name: str,
    rendered: RenderedRestCall,
) -> dict[str, Any]:
    return {
        "method": method,
        "path": path,
        "path_params": json.dumps(rendered.path_params) if rendered.path_params else "",
        "query_params": json.dumps(rendered.query_params) if rendered.query_params else "",
        "body": json.dumps(rendered.body) if rendered.body else "",
        "name": name,
        "success": True,
    }


def attach_binding_values(
    rows: list[Any],
    binding: dict[str, Any],
    output: dict[str, Any],
) -> list[Any]:
    """Attach map keys needed by downstream SQL joins."""
    attach = output.get("attach_binding") or []
    if not attach or not binding:
        return rows

    attached: list[Any] = []
    for row in rows:
        if not isinstance(row, dict):
            attached.append(row)
            continue
        next_row = dict(row)
        for name in attach:
            if isinstance(name, str) and name in binding:
                next_row[name] = binding[name]
        attached.append(next_row)
    return attached


def collect_step_rows(
    data: Any,
    step: dict[str, Any],
    binding: dict[str, Any],
) -> list[Any]:
    name = get_step_result_name(step)
    tables, _ = extract_tables_from_response(data, name)
    rows = tables.get(name)
    if not isinstance(rows, list):
        return []
    return attach_binding_values(rows, binding, get_step_output(step))


def store_step_rows(
    results: dict[str, Any],
    step: dict[str, Any],
    rows: list[Any],
) -> str:
    name = get_step_result_name(step)
    results[name] = rows
    return name


def format_recipe_response(
    last_result_var: ContextVar[list[Any]],
    executed_items: list[Any],
    executed_sql: list[str],
    item_key: str,
) -> str:
    """Format recipe JSON response with truncation."""
    try:
        last_rows = last_result_var.get()[0]
    except LookupError:
        last_rows = None

    base = {"success": True, item_key: executed_items, "executed_sql": executed_sql}
    if isinstance(last_rows, list):
        base.update(truncate_for_context(last_rows, "sql_result"))
    return json.dumps(base, indent=2)


def build_partial_result(
    last_data: Any,
    api_calls: list[Any],
    turn_info: str,
    call_key: str,
) -> dict[str, Any]:
    """Build partial result dict for MaxTurnsExceeded."""
    if last_data:
        return {
            "ok": True,
            "data": f"[Partial - {turn_info}] Max turns exceeded but data retrieved.",
            "result": last_data,
            call_key: api_calls,
            "error": None,
        }
    return {
        "ok": False,
        "data": None,
        "result": None,
        call_key: api_calls,
        "error": f"Max turns exceeded ({turn_info}), no data retrieved",
    }


async def execute_recipe_steps(
    recipe: dict[str, Any],
    params: dict[str, Any],
    query_results_var: ContextVar[dict[str, Any]],
    last_result_var: ContextVar[list[Any]],
    api_step_executor,
    executed_items_list,
) -> tuple[bool, Any, list[str], str]:
    """Execute recipe steps (API + SQL). Returns (success, last_data, executed_sql, error_json)."""
    results = _get_results_context(query_results_var)
    executed_sql: list[str] = []

    for step_idx, step in enumerate(get_recipe_steps(recipe)):
        if isinstance(step, dict) and step.get("kind") == "sql":
            success, sql, error = _execute_sql_step(step, params, results, last_result_var)
            if sql:
                executed_sql.append(sql)
            if not success:
                return False, None, executed_sql, error
            continue

        success, data, error, call_rec = await api_step_executor(step_idx, step, params, results)
        if not success:
            return False, None, executed_sql, error

        if call_rec:
            if isinstance(call_rec, list):
                executed_items_list.extend(call_rec)
            else:
                executed_items_list.append(call_rec)

        if data is not None:
            _set_last_result(last_result_var, data)

    try:
        return True, last_result_var.get()[0], executed_sql, ""
    except LookupError:
        return True, None, executed_sql, ""
