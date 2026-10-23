"""REST API client with unsafe method blocking."""

import fnmatch
import logging
import re
from typing import Any
from urllib.parse import quote, urlencode, urljoin, urlsplit

import httpx

from ..utils.http import bounded_request

logger = logging.getLogger(__name__)

# Unsafe HTTP methods (blocked by default)
_UNSAFE_METHODS = {"POST", "PUT", "DELETE", "PATCH"}


def _extract_http_error_details(response: httpx.Response | None) -> Any | None:
    """Extract bounded error detail from non-2xx responses."""
    if response is None:
        return None

    try:
        payload = response.json()
    except Exception:
        payload = None

    if payload is not None:
        if isinstance(payload, dict):
            for key in ("errors", "error", "message"):
                if key in payload:
                    return payload[key]
        return payload

    raw = response.content[:1500] if response.content else b""
    text = raw.decode("utf-8", errors="replace").strip() if raw else ""
    return text[:1000] if text else None


def _build_http_error_response(e: httpx.HTTPStatusError) -> dict[str, Any]:
    status_code = e.response.status_code if e.response is not None else 0
    response: dict[str, Any] = {
        "success": False,
        "error": f"HTTP {status_code}",
        "status_code": status_code,
    }
    details = _extract_http_error_details(e.response)
    if details is not None:
        response["details"] = details
    return response


def _is_path_allowed(path: str, patterns: list[str]) -> bool:
    """Check if path matches any allowed pattern (fnmatch glob; {param} matches a value)."""
    return any(fnmatch.fnmatch(path, re.sub(r"\{[^/{}]+\}", "*", p)) for p in patterns)


def _build_url(
    path: str,
    base_url: str,
    path_params: dict[str, Any] | None = None,
    query_params: dict[str, Any] | None = None,
) -> str:
    """Build full URL with path and query params.

    Args:
        path: API path (e.g., /users/{id})
        base_url: Base URL for API
        path_params: Values to substitute in path (e.g., {"id": "123"})
        query_params: Query string parameters (e.g., {"limit": 10})

    Returns:
        Full URL string
    """
    # Substitute path params
    if path_params:
        for key, value in path_params.items():
            path = path.replace(f"{{{key}}}", quote(str(value), safe=""))

    if not base_url:
        raise ValueError("No base URL provided")

    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    base, target = urlsplit(base_url), urlsplit(url)
    if target[:2] != base[:2] or not (target.path + "/").startswith(base.path.rstrip("/") + "/"):
        raise ValueError("Path must stay on the API base URL")

    # Add query params
    if query_params:
        # Filter out None values
        filtered = {k: v for k, v in query_params.items() if v is not None}
        if filtered:
            url = f"{url}?{urlencode(filtered, doseq=True)}"

    return url


async def execute_request(
    method: str,
    path: str,
    path_params: dict[str, Any] | None = None,
    query_params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    base_url: str = "",
    headers: dict[str, str] | None = None,
    allow_unsafe: bool = False,
    allow_unsafe_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Execute REST API request. Unsafe methods blocked unless explicitly allowed.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE, PATCH)
        path: API path (e.g., /users/{id})
        path_params: Values to substitute in path
        query_params: Query string parameters
        body: Request body (for POST/PUT/PATCH)
        base_url: Base URL for API (required)
        headers: Headers to send (e.g., Authorization)
        allow_unsafe: Allow all POST/PUT/DELETE/PATCH methods
        allow_unsafe_paths: Glob patterns for paths where unsafe methods are allowed

    Returns:
        Dict with success/data or error
    """
    method = method.upper()
    if method not in {"GET", *_UNSAFE_METHODS}:
        return {"success": False, "error": f"Unsupported method: {method}"}

    try:
        url = _build_url(path, base_url, path_params, query_params)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # Block unsafe methods by default; match the allowlist on the resolved path.
    api_path = "/" + urlsplit(url).path[len(urlsplit(base_url).path.rstrip("/")) :].lstrip("/")
    if method in _UNSAFE_METHODS and not allow_unsafe:
        if not allow_unsafe_paths or not _is_path_allowed(api_path, allow_unsafe_paths):
            return {
                "success": False,
                "error": f"{method} method not allowed (read-only mode). Use X-Allow-Unsafe-Paths header.",
            }

    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    # Log request details without leaking header values or query strings (may hold API keys).
    logger.info(
        "REST request resolved: method=%s base_url=%s path=%s url=%s header_keys=%s",
        method,
        base_url,
        path,
        url.split("?", 1)[0],
        sorted(request_headers.keys()),
    )

    kwargs: dict[str, Any] = {}
    if method in {"POST", "PUT", "PATCH"}:
        request_headers["Content-Type"] = "application/json"
        kwargs["json"] = body
    try:
        resp = await bounded_request(method, url, headers=request_headers, **kwargs)
        resp.raise_for_status()

        # Handle different content types
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            data = resp.json()
        else:
            data = resp.text

        return {"success": True, "data": data}

    except httpx.HTTPStatusError as e:
        return _build_http_error_response(e)
    except Exception as e:
        logger.exception("REST API error")
        return {"success": False, "error": str(e)}
