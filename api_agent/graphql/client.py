"""GraphQL client."""

import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Block mutations (read-only mode)
_MUTATION_PATTERN = re.compile(r"^\s*mutation\b", re.IGNORECASE | re.MULTILINE)


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
    if _MUTATION_PATTERN.search(query):
        return {"success": False, "error": "Mutations are not allowed (read-only mode)"}

    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)

    payload: dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                endpoint,
                json=payload,
                headers=request_headers,
            )
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
