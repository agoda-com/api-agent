"""Unified MCP tool for direct API execution."""

import json
from dataclasses import replace
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from ..auth import resolve_headers
from ..config import settings
from ..context import MissingHeaderError, get_request_context
from ..graphql import execute_query
from ..rest.client import execute_request
from ..rest.schema_loader import fetch_schema_context


def register_execute_tool(mcp: FastMCP) -> None:
    """Register the unified execute tool with generic internal name."""

    @mcp.tool(
        name="_execute",
        description="""Execute a specific API call directly.

For GraphQL: provide query (and optional variables)
For REST: provide method and path (and optional params/body)

Use this to re-run queries from the query tool or execute known operations.""",
        tags={"execute"},
    )
    async def execute(
        # GraphQL params
        query: Annotated[str | None, Field(description="GraphQL query string")] = None,
        variables: Annotated[dict[str, Any] | None, Field(description="GraphQL variables")] = None,
        # REST params
        method: Annotated[str | None, Field(description="HTTP method (GET, POST, etc.)")] = None,
        path: Annotated[str | None, Field(description="API path (e.g., /users/{id})")] = None,
        path_params: Annotated[
            dict[str, Any] | None, Field(description="Path parameter values")
        ] = None,
        query_params: Annotated[
            dict[str, Any] | None, Field(description="Query string parameters")
        ] = None,
        body: Annotated[
            dict[str, Any] | None, Field(description="Request body (for POST/PUT/PATCH)")
        ] = None,
    ) -> dict:
        """Execute API call directly."""
        try:
            ctx = get_request_context()
        except MissingHeaderError as e:
            return {"ok": False, "error": str(e)}

        # Auth middleware: resolve token and inject into headers
        resolved = await resolve_headers(ctx)
        ctx = replace(ctx, target_headers=resolved)

        if ctx.api_type == "graphql":
            # GraphQL execution
            if not query:
                return {"ok": False, "error": "query param required for GraphQL"}

            result = await execute_query(query, variables, ctx.target_url, ctx.target_headers)

            if not result.get("success"):
                return {"ok": False, "error": result.get("error", "Query failed")}

            data = result.get("data", {})
            data_str = json.dumps(data, indent=2)

            if len(data_str) > settings.MAX_RESPONSE_CHARS:
                return {
                    "ok": True,
                    "data": f"{data_str[: settings.MAX_RESPONSE_CHARS]}\n\n[TRUNCATED - Use pagination to fetch smaller chunks.]",
                }

            return {"ok": True, "data": data}

        else:
            # REST execution
            if not method or not path:
                return {"ok": False, "error": "method and path params required for REST"}

            # Get base URL from header override or spec
            base_url = ctx.base_url
            if not base_url:
                _, base_url, _ = await fetch_schema_context(ctx.target_url, ctx.target_headers)
            if not base_url:
                return {"ok": False, "error": "Could not extract base URL from OpenAPI spec"}

            result = await execute_request(
                method,
                path,
                path_params,
                query_params,
                body,
                base_url=base_url,
                headers=ctx.target_headers,
                allow_unsafe_paths=list(ctx.allow_unsafe_paths),
            )

            if not result.get("success"):
                return {"ok": False, "error": result.get("error", "Request failed")}

            data = result.get("data", {})
            data_str = json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data)

            if len(data_str) > settings.MAX_RESPONSE_CHARS:
                return {
                    "ok": True,
                    "data": f"{data_str[: settings.MAX_RESPONSE_CHARS]}\n\n[TRUNCATED - Use query params to limit results.]",
                }

            return {"ok": True, "data": data}
