"""Auth0 token middleware — fetch and cache bearer tokens for API requests."""

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Token cache: auth_url -> (token, expires_at_monotonic)
_token_cache: dict[str, tuple[str, float]] = {}

_DEFAULT_TTL = 300  # 5 minutes


def _get_nested_value(data: Any, path: str) -> Any:
    """Extract value from nested dict/list using dot notation.

    Args:
        data: Dictionary or list to extract from
        path: Dot-separated path (e.g., "access_token", "data.token")

    Returns:
        Value at path or None if not found
    """
    if not data or not path:
        return None
    current: Any = data
    for key in path.split("."):
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


async def fetch_auth_token(
    auth_url: str,
    auth_body: dict | None = None,
    token_path: str = "access_token",
) -> str | None:
    """Fetch bearer token from Auth0 endpoint, with caching.

    Args:
        auth_url: Auth0 token endpoint URL
        auth_body: POST body (e.g., client_credentials grant payload)
        token_path: Dot-path to extract token from response

    Returns:
        Token string, or None on failure
    """
    # Check cache freshness
    cached = _token_cache.get(auth_url)
    if cached:
        token, expires_at = cached
        if time.monotonic() < expires_at:
            logger.debug("Auth token cache hit for %s", auth_url)
            return token

    # Fetch new token from Auth0
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                auth_url,
                json=auth_body or {},
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        token = _get_nested_value(data, token_path)
        if not token or not isinstance(token, str):
            logger.warning(
                "Auth token not found at path '%s' in response from %s", token_path, auth_url
            )
            return None

        # Cache with TTL
        _token_cache[auth_url] = (token, time.monotonic() + _DEFAULT_TTL)
        logger.info("Auth token fetched and cached for %s", auth_url)
        return token

    except Exception:
        logger.exception("Failed to fetch auth token from %s", auth_url)
        return None


async def resolve_headers(ctx: Any) -> dict:
    """Resolve auth token and merge into target headers.

    If ctx.auth_url is configured, fetches a bearer token and injects
    Authorization header. Otherwise returns ctx.target_headers unchanged.

    Args:
        ctx: RequestContext with optional auth_url, auth_body, auth_token_path

    Returns:
        Headers dict ready for API calls
    """
    if not ctx.auth_url:
        return ctx.target_headers

    token = await fetch_auth_token(
        auth_url=ctx.auth_url,
        auth_body=ctx.auth_body,
        token_path=ctx.auth_token_path,
    )

    if not token:
        logger.warning("Auth token fetch failed; proceeding without auto-generated token")
        return ctx.target_headers

    headers = dict(ctx.target_headers)
    headers["Authorization"] = f"Bearer {token}"
    return headers
