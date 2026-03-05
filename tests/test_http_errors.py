"""Tests for HTTP error detail helpers."""

import httpx

from api_agent.utils.http_errors import build_http_error_response, extract_http_error_details


def _response(status: int, *, json_body=None, text_body: str = "") -> httpx.Response:
    request = httpx.Request("GET", "https://api.example.com/test")
    if json_body is not None:
        return httpx.Response(status, request=request, json=json_body)
    return httpx.Response(status, request=request, text=text_body)


def test_extract_http_error_details_prefers_errors_field():
    response = _response(400, json_body={"errors": [{"message": "bad field"}], "message": "ignored"})
    assert extract_http_error_details(response) == [{"message": "bad field"}]


def test_extract_http_error_details_limits_text_fallback():
    response = _response(500, text_body="x" * 5000)
    details = extract_http_error_details(response)
    assert isinstance(details, str)
    assert len(details) == 1000


def test_build_http_error_response_includes_status_and_details():
    response = _response(404, json_body={"error": "missing"})
    request = response.request
    exc = httpx.HTTPStatusError("not found", request=request, response=response)

    out = build_http_error_response(exc)
    assert out["success"] is False
    assert out["error"] == "HTTP 404"
    assert out["status_code"] == 404
    assert out["details"] == "missing"
