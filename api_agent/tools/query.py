"""Unified MCP tool for natural language API queries."""

from typing import Annotated

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from ..agent.graphql_agent import process_query
from ..agent.rest_agent import process_rest_query
from ..context import MissingHeaderError, get_request_context
from ..query_response import QueryResponse
from ..recipe.state import consume_recipe_changes, reset_recipe_change_flag
from ..utils.csv import to_csv


def _should_include_result(response: QueryResponse, req_ctx) -> bool:
    """Include rows when explicitly requested or when debug wraps direct CSV output."""
    return bool(req_ctx.include_result or (req_ctx.debug and response.should_return_csv))


def _should_return_csv(response: QueryResponse, req_ctx, *, return_directly: bool) -> bool:
    """Return raw CSV when requested and rows are available, unless debug wraps output."""
    return bool(
        response.result is not None
        and not req_ctx.debug
        and (return_directly or response.should_return_csv)
    )


def register_query_tool(mcp: FastMCP) -> None:
    """Register the unified query tool."""

    @mcp.tool(
        name="_query",
        title="Ask API",
        description="""Ask a natural-language question about the configured API.

Use when the client needs fresh API data, joins, filtering, ranking, or SQL-style post-processing.
The agent reads the API schema, calls the target API, and returns the answer plus calls made.""",
        tags={"query", "nl"},
        annotations={"title": "Ask API", "openWorldHint": True},
    )
    async def query(
        question: Annotated[str, Field(description="Natural language question about the API")],
        return_directly: Annotated[
            bool,
            Field(description="Return raw CSV directly when tabular result rows are available."),
        ] = False,
        ctx: Context | None = None,
    ) -> dict | str:
        """Process natural language query against configured API."""
        try:
            req_ctx = get_request_context()
        except MissingHeaderError as e:
            return {"ok": False, "error": str(e)}

        # Track recipe creation in this request
        reset_recipe_change_flag()

        if req_ctx.api_type == "graphql":
            result = await process_query(question, req_ctx)
        else:
            result = await process_rest_query(question, req_ctx)

        # Notify clients if recipes changed
        if ctx and consume_recipe_changes():
            try:
                notify = getattr(ctx, "send_tool_list_changed", None)
                if notify:
                    await notify()
            except Exception:
                pass

        # Direct return: just CSV, no wrapper
        calls_key = "queries" if req_ctx.api_type == "graphql" else "api_calls"
        response = QueryResponse.from_agent_result(result, calls_key)
        if _should_return_csv(response, req_ctx, return_directly=return_directly):
            return to_csv(response.result)

        return response.to_mcp_payload(
            include_result=_should_include_result(response, req_ctx),
            include_debug=req_ctx.debug,
        )
