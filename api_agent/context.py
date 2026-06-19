"""Request context extraction from HTTP headers."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal, cast
from urllib.parse import urlparse

from fastmcp.server.dependencies import get_http_headers

logger = logging.getLogger(__name__)


ApiType = Literal["graphql", "rest"]
_TRUE_VALUES = {"true", "1", "yes"}


class MissingHeaderError(Exception):
    """Required header missing from request."""

    pass


@dataclass(frozen=True)
class RequestContext:
    """Per-request context extracted from headers."""

    target_url: str  # X-Target-URL: GraphQL endpoint or OpenAPI spec URL
    api_type: ApiType  # X-API-Type: "graphql" or "rest"
    target_headers: dict  # X-Target-Headers: parsed JSON headers
    allow_unsafe_paths: tuple[str, ...]  # X-Allow-Unsafe-Paths: glob patterns for POST/etc
    base_url: str | None  # X-Base-URL: override base URL (REST only)
    include_result: bool  # X-Include-Result: whether to include full result in output
    poll_paths: tuple[str, ...]  # X-Poll-Paths: paths that require polling (enables poll tool)
    learning_rate: float | None = None  # X-Recipe-Learn-Rate: per-request sample rate
    debug: bool = False  # X-Debug: include trace/call debugging metadata


def get_request_context() -> RequestContext:
    """Extract context from current HTTP request headers."""
    return parse_request_context(get_http_headers(include={"authorization"}))


def parse_request_context(headers: dict[str, str]) -> RequestContext:
    """Parse request context from normalized HTTP headers.

    Required headers:
        X-Target-URL: Target API endpoint (GraphQL) or OpenAPI spec URL (REST)
        X-API-Type: "graphql" or "rest"

    Optional headers:
        X-Target-Headers: JSON object with auth headers to forward
        X-Passthrough-Headers: JSON array of header names to copy from this request
            into target headers (merged after X-Target-Headers)
        X-Allow-Unsafe-Paths: JSON array of glob patterns for POST/PUT/DELETE/PATCH
        X-Base-URL: Override base URL for REST API calls
        X-Include-Result: Include full uncapped result in output (default: false)
        X-Poll-Paths: JSON array of paths requiring polling (enables poll tool)
        X-Recipe-Learn-Rate: Override recipe sample rate for this request (0..1)
        X-Debug: Include trace/call debugging metadata

    Raises:
        MissingHeaderError: If required headers are missing or invalid
    """
    target_url = headers.get("x-target-url")
    api_type = headers.get("x-api-type")
    target_headers_raw = headers.get("x-target-headers") or "{}"
    passthrough_headers_raw = headers.get("x-passthrough-headers") or "[]"
    allow_unsafe_paths_raw = headers.get("x-allow-unsafe-paths") or "[]"
    base_url_raw = headers.get("x-base-url")
    include_result_raw = headers.get("x-include-result", "false")
    poll_paths_raw = headers.get("x-poll-paths") or "[]"
    learning_rate_raw = headers.get("x-recipe-learn-rate")
    debug_raw = headers.get("x-debug", "false")

    if not target_url:
        raise MissingHeaderError("X-Target-URL header required")

    if not api_type:
        raise MissingHeaderError("X-API-Type header required (graphql|rest)")

    if api_type not in ("graphql", "rest"):
        raise MissingHeaderError(f"X-API-Type must be 'graphql' or 'rest', got '{api_type}'")

    typed_api_type = cast(ApiType, api_type)
    extra = {"target_url": target_url, "api_type": api_type}
    target_headers = _parse_json_object(target_headers_raw, "target headers", extra)
    passthrough_names = _parse_json_string_tuple(
        passthrough_headers_raw,
        "passthrough headers",
        extra,
    )
    _merge_passthrough_headers(target_headers, headers, passthrough_names)

    return RequestContext(
        target_url=target_url,
        api_type=typed_api_type,
        target_headers=target_headers,
        allow_unsafe_paths=_parse_json_string_tuple(
            allow_unsafe_paths_raw,
            "allow unsafe paths",
            extra,
        ),
        base_url=base_url_raw if base_url_raw else None,
        include_result=_is_truthy(include_result_raw),
        poll_paths=_parse_json_string_tuple(poll_paths_raw, "poll paths", extra),
        learning_rate=_parse_learning_rate(learning_rate_raw),
        debug=_is_truthy(debug_raw),
    )


def _is_truthy(value: str | None) -> bool:
    return (value or "").lower() in _TRUE_VALUES


def _parse_learning_rate(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        rate = float(raw)
    except ValueError as exc:
        raise MissingHeaderError("X-Recipe-Learn-Rate must be a number from 0 to 1") from exc
    if rate < 0 or rate > 1:
        raise MissingHeaderError("X-Recipe-Learn-Rate must be between 0 and 1")
    return rate


def _parse_json_object(raw: str, label: str, extra: dict[str, str]) -> dict:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON for %s", label, extra=extra)
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _parse_json_string_tuple(raw: str, label: str, extra: dict[str, str]) -> tuple[str, ...]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON for %s", label, extra=extra)
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(value for value in parsed if isinstance(value, str))


def _to_snake_case(name: str) -> str:
    """Convert string to snake_case."""
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-zA-Z0-9_]", "", name)
    return name.lower().strip("_")


def _normalize_explicit_api_name(name: str) -> str:
    """Normalize explicit X-API-Name while preserving hyphens."""
    cleaned = (name or "").strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_-]", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned)
    return cleaned.strip("_-") or "api"


def get_full_hostname(url: str | None) -> str:
    """Get full hostname from URL for description."""
    if not url:
        return "api"
    parsed = urlparse(url)
    return parsed.hostname or "api"


def get_tool_name_prefix(url: str | None) -> str:
    """Get semantic prefix for tool name (≤32 chars).

    Extracts meaningful parts from hostname, skipping generic TLDs and infra names.
    Example: flights-service.internal.example.com → flights_service_example
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
        return _normalize_explicit_api_name(api_name)[:32]

    # Fall back to semantic prefix from URL
    target_url = headers.get("x-target-url", "")
    return get_tool_name_prefix(target_url)


def _merge_passthrough_headers(
    target_headers: dict[str, str],
    client_headers: dict[str, str],
    passthrough_names: tuple[str, ...],
) -> None:
    """Copy listed headers from the MCP request into target_headers (mutates target_headers)."""
    for raw_name in passthrough_names:
        if not isinstance(raw_name, str):
            continue
        key_lower = raw_name.lower().strip()
        if not key_lower or key_lower not in client_headers:
            continue
        canonical = _canonical_http_header_name(key_lower)
        target_headers[canonical] = client_headers[key_lower]


def _canonical_http_header_name(name: str) -> str:
    """Turn a lowercased header name into conventional Title-Case (e.g. x-request-id → X-Request-Id)."""
    return "-".join(part.capitalize() for part in name.strip().split("-"))
