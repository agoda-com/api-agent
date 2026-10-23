"""GraphQL client."""

import logging
from typing import Any

import httpx
from graphql import GraphQLError, OperationDefinitionNode, OperationType, parse

from ..utils.http import bounded_request

logger = logging.getLogger(__name__)


def _read_only_error(query: str) -> str | None:
    """Return an error unless every operation in the document is a query."""
    try:
        document = parse(query)
    except GraphQLError as e:
        return f"Invalid GraphQL query: {e.message}"
    for definition in document.definitions:
        if (
            isinstance(definition, OperationDefinitionNode)
            and definition.operation != OperationType.QUERY
        ):
            return "Mutations are not allowed (read-only mode)"
    return None


async def execute_query(
    query: str,
    variables: dict[str, Any] | None = None,
    endpoint: str = "",
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute GraphQL query. Mutations are blocked (read-only mode).

    Args:
        query: GraphQL query string
        variables: Optional query variables
        endpoint: GraphQL endpoint URL (required)
        headers: Headers to send (e.g., Authorization)

    Returns:
        Dict with success/data or error
    """
    if not endpoint:
        return {"success": False, "error": "No endpoint provided"}

    # Block mutations
    if error := _read_only_error(query):
        return {"success": False, "error": error}

    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)

    payload: dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables

    try:
        resp = await bounded_request("POST", endpoint, json=payload, headers=request_headers)
        resp.raise_for_status()
        result = resp.json()
        if "errors" in result:
            return {"success": False, "error": result["errors"]}
        return {"success": True, "data": result.get("data", {})}
    except httpx.HTTPStatusError as e:
        return {"success": False, "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        logger.exception("GraphQL error")
        return {"success": False, "error": str(e)}
