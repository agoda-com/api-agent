"""REST polling tool factory."""

from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar
from typing import Any

from agents import function_tool

from ..config import settings
from ..context import RequestContext
from ..executor import extract_tables_from_response, truncate_for_context
from .client import execute_request

_default_rest_calls: ContextVar[list[dict[str, Any]]] = ContextVar("poll_rest_calls")
_default_query_results: ContextVar[dict[str, Any]] = ContextVar("poll_query_results")
_default_last_result: ContextVar[list] = ContextVar("poll_last_result")


def _get_nested_value(data: dict | None, path: str) -> Any:
    if not data or not path:
        return None
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, (dict, list)):
            return None
        if isinstance(current, list) and key.isdigit():
            idx = int(key)
            if not 0 <= idx < len(current):
                return None
            current = current[idx]
        elif isinstance(current, dict):
            current = current.get(key)
        else:
            return None
        if current is None:
            return None
    return current


def _append_contextvar_list(var: ContextVar[list[dict[str, Any]]], item: dict[str, Any]) -> None:
    try:
        var.get().append(item)
    except LookupError:
        pass


def _store_poll_result(
    data: Any,
    name: str,
    query_results_var: ContextVar[dict[str, Any]],
    last_result_var: ContextVar[list],
) -> None:
    try:
        results = query_results_var.get()
        tables, _ = extract_tables_from_response(data, name)
        results.update(tables)
        stored = tables.get(name)
        if stored is not None:
            last_result_var.get()[0] = stored
    except LookupError:
        pass


def _poll_wait_ms(delay_ms: int) -> int:
    requested = delay_ms if delay_ms > 0 else settings.DEFAULT_POLL_DELAY_MS
    return max(0, min(requested, settings.MAX_POLL_DELAY_MS))


def create_poll_tool(
    ctx: RequestContext,
    base_url: str,
    *,
    rest_calls_var: ContextVar[list[dict[str, Any]]] = _default_rest_calls,
    query_results_var: ContextVar[dict[str, Any]] = _default_query_results,
    last_result_var: ContextVar[list] = _default_last_result,
):
    """Create poll_until_done tool with bound request context."""

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
        """Poll endpoint until done_field equals done_value."""
        path_params_dict = json.loads(path_params) if path_params else None
        query_params_dict = json.loads(query_params) if query_params else None
        try:
            body_dict = json.loads(body) if body else {}
        except json.JSONDecodeError as e:
            return json.dumps({"success": False, "error": f"Invalid body JSON: {e.msg}"})

        max_polls = settings.MAX_POLLS
        wait_ms = _poll_wait_ms(delay_ms)
        current = None

        attempt = 0
        while attempt < max_polls:
            attempt += 1

            result = await execute_request(
                method,
                path,
                path_params_dict,
                query_params_dict,
                body=body_dict if body_dict else None,
                base_url=base_url,
                headers=ctx.target_headers,
                allow_unsafe_paths=list(ctx.allow_unsafe_paths),
            )

            _append_contextvar_list(
                rest_calls_var,
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
            current = _get_nested_value(data, done_field)
            if current is None and attempt == 1:
                keys = list(data.keys()) if isinstance(data, dict) else []
                return json.dumps(
                    {
                        "success": False,
                        "error": f"done_field '{done_field}' not found in response. Available keys: {keys}",
                    }
                )

            if str(current).lower() == done_value.lower():
                _store_poll_result(data, name, query_results_var, last_result_var)
                return json.dumps(
                    {
                        "success": True,
                        **truncate_for_context(data if isinstance(data, list) else [data], name),
                        "attempts": attempt,
                    },
                    indent=2,
                )

            if attempt >= max_polls:
                break

            if wait_ms > 0:
                await asyncio.sleep(wait_ms / 1000)

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
