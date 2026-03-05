"""HTTP error detail extraction utilities."""

from typing import Any

import httpx


def build_http_error_response(e: httpx.HTTPStatusError) -> dict[str, Any]:
    """Build a consistent error payload from HTTPStatusError."""
    status_code = e.response.status_code if e.response is not None else 0
    out: dict[str, Any] = {
        "success": False,
        "error": f"HTTP {status_code}",
        "status_code": status_code,
    }
    details = extract_http_error_details(e.response)
    if details is not None:
        out["details"] = details
    return out


def extract_http_error_details(response: httpx.Response | None) -> Any | None:
    """Extract useful error payload from non-2xx HTTP responses."""
    if response is None:
        return None

    try:
        payload = response.json()
    except Exception:
        payload = None

    if payload is not None:
        if isinstance(payload, dict):
            if "errors" in payload:
                return payload["errors"]
            if "error" in payload:
                return payload["error"]
            if "message" in payload:
                return payload["message"]
        return payload

    # Bound fallback text extraction to avoid loading huge payloads.
    raw = response.content[:1500] if response.content else b""
    text = raw.decode("utf-8", errors="replace").strip() if raw else ""
    if text:
        return text[:1000]
    return None
