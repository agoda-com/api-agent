"""Request context extraction from HTTP headers."""

import json
import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from fastmcp.server.dependencies import get_http_headers


_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _expand_env_vars(value: str) -> str:
    """Expand ``${VAR}`` references in *value* using environment variables.

    Unknown variables are left as-is.
    """
    return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)


class MissingHeaderError(Exception):
    """Required header missing from request."""

    pass


@dataclass(frozen=True)
class RequestContext:
    """Per-request context extracted from headers."""

    target_url: str  # X-Target-URL: GraphQL endpoint or OpenAPI spec URL
    api_type: str  # X-API-Type: "graphql" or "rest"
    target_headers: dict  # X-Target-Headers: parsed JSON headers
    allow_unsafe_paths: tuple[str, ...]  # X-Allow-Unsafe-Paths: glob patterns for POST/etc
    base_url: str | None  # X-Base-URL: override base URL (REST only)
    include_result: bool  # X-Include-Result: whether to include full result in output
    poll_paths: tuple[str, ...]  # X-Poll-Paths: paths that require polling (enables poll tool)


def get_request_context() -> RequestContext:
    """Extract context from current request headers.

    Required headers:
        X-Target-URL: Target API endpoint (GraphQL) or OpenAPI spec URL (REST)
        X-API-Type: "graphql" or "rest"

    Optional headers:
        X-Target-Headers: JSON object with auth headers to forward
        X-Allow-Unsafe-Paths: JSON array of glob patterns for POST/PUT/DELETE/PATCH
        X-Base-URL: Override base URL for REST API calls
        X-Include-Result: Include full uncapped result in output (default: false)
        X-Poll-Paths: JSON array of paths requiring polling (enables poll tool)

    Raises:
        MissingHeaderError: If required headers are missing or invalid
    """
    headers = get_http_headers()

    target_url = _expand_env_vars(headers.get("x-target-url") or "") or None
    api_type = headers.get("x-api-type")
    target_headers_raw = _expand_env_vars(headers.get("x-target-headers") or "{}")
    allow_unsafe_paths_raw = headers.get("x-allow-unsafe-paths") or "[]"
    base_url_raw = headers.get("x-base-url")
    include_result_raw = headers.get("x-include-result", "false")
    poll_paths_raw = headers.get("x-poll-paths") or "[]"

    base_url = _expand_env_vars(base_url_raw) if base_url_raw else None
    include_result = (include_result_raw or "").lower() in ("true", "1", "yes")

    if not target_url:
        raise MissingHeaderError("X-Target-URL header required")

    if not api_type:
        raise MissingHeaderError("X-API-Type header required (graphql|rest)")

    if api_type not in ("graphql", "rest"):
        raise MissingHeaderError(f"X-API-Type must be 'graphql' or 'rest', got '{api_type}'")

    try:
        target_headers = json.loads(target_headers_raw)
    except json.JSONDecodeError:
        target_headers = {}

    try:
        allow_unsafe_paths = tuple(json.loads(allow_unsafe_paths_raw))
    except json.JSONDecodeError:
        allow_unsafe_paths = ()

    try:
        poll_paths = tuple(json.loads(poll_paths_raw))
    except json.JSONDecodeError:
        poll_paths = ()

    return RequestContext(
        target_url=target_url,
        api_type=api_type,
        target_headers=target_headers,
        allow_unsafe_paths=allow_unsafe_paths,
        base_url=base_url,
        include_result=include_result,
        poll_paths=poll_paths,
    )


def _to_snake_case(name: str) -> str:
    """Convert string to snake_case."""
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-zA-Z0-9_]", "", name)
    return name.lower().strip("_")


def get_full_hostname(url: str | None) -> str:
    """Get full hostname from URL for description."""
    if not url:
        return "api"
    parsed = urlparse(url)
    return parsed.hostname or "api"


def get_tool_name_prefix(url: str | None) -> str:
    """Get semantic prefix for tool name (≤32 chars).

    Extracts meaningful parts from hostname, skipping generic TLDs and infra names.
    Example: flights-api-qa.internal.example.com → flights_api_example
    """
    if not url:
        return "api"

    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    if not hostname:
        return "api"

    parts = hostname.split(".")
    # Skip generic TLDs and internal infra names
    skip = {
        "com",
        "io",
        "is",
        "net",
        "org",
        "privatecloud",
        "qa",
        "dev",
        "internal",
        "api",
    }
    meaningful = [_to_snake_case(p) for p in parts if p.lower() not in skip and p]

    # Join meaningful parts, cap at 32 chars
    return "_".join(meaningful)[:32] or "api"


def extract_api_name(headers: dict | None = None) -> str:
    """Extract API name prefix from headers. Priority: X-API-Name > parse X-Target-URL."""
    if headers is None:
        headers = get_http_headers()

    # Explicit header takes priority
    if api_name := headers.get("x-api-name"):
        return _to_snake_case(api_name)[:32]

    # Fall back to semantic prefix from URL
    target_url = headers.get("x-target-url", "")
    return get_tool_name_prefix(target_url)
