"""Tests for REST client."""

import gzip
from unittest.mock import patch

import httpx
import pytest

from api_agent.rest.client import _build_url, execute_request
from api_agent.utils.http import MAX_RESPONSE_BYTES, bounded_request


def _mock_client(handler):
    real = httpx.AsyncClient
    return patch(
        "api_agent.utils.http.httpx.AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )


@pytest.mark.asyncio
async def test_bounded_request_rejects_oversized_body():
    with _mock_client(lambda _: httpx.Response(200, content=b"x" * (MAX_RESPONSE_BYTES + 1))):
        with pytest.raises(ValueError, match="exceeds"):
            await bounded_request("GET", "https://api.example.com/big")


@pytest.mark.asyncio
async def test_bounded_request_returns_decoded_body():
    body = gzip.compress(b'{"ok": true}')
    headers = {"content-encoding": "gzip", "content-type": "application/json"}
    with _mock_client(lambda _: httpx.Response(200, content=body, headers=headers)):
        resp = await bounded_request("GET", "https://api.example.com/x")

    assert resp.json() == {"ok": True}


class TestExecuteRequest:
    """Test request execution and method blocking."""

    @pytest.mark.asyncio
    async def test_blocks_post_by_default(self):
        result = await execute_request(
            "POST",
            "/users",
            base_url="https://api.example.com",
            body={"name": "test"},
            allow_unsafe=False,
        )
        assert result["success"] is False
        assert "not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_blocks_put_by_default(self):
        result = await execute_request(
            "PUT",
            "/users/123",
            base_url="https://api.example.com",
            body={"name": "test"},
            allow_unsafe=False,
        )
        assert result["success"] is False
        assert "not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_blocks_delete_by_default(self):
        result = await execute_request(
            "DELETE",
            "/users/123",
            base_url="https://api.example.com",
            allow_unsafe=False,
        )
        assert result["success"] is False
        assert "not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_blocks_patch_by_default(self):
        result = await execute_request(
            "PATCH",
            "/users/123",
            base_url="https://api.example.com",
            body={"name": "test"},
            allow_unsafe=False,
        )
        assert result["success"] is False
        assert "not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_no_base_url_returns_error(self):
        result = await execute_request("GET", "/users", base_url="")
        assert result["success"] is False
        assert "No base URL" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("path", "path_params"),
        [
            ("https://evil.example/x", None),
            ("http://169.254.169.254/latest/meta-data/", None),
            ("../admin", None),
            ("{u}/admin", {"u": ".."}),
        ],
    )
    async def test_rejects_paths_leaving_base_url(self, path, path_params):
        with patch("api_agent.rest.client.bounded_request") as client:
            result = await execute_request(
                "GET",
                path,
                path_params=path_params,
                base_url="https://api.example.com/v1",
                headers={"Authorization": "Bearer secret"},
            )

        assert result == {"success": False, "error": "Path must stay on the API base URL"}
        client.assert_not_called()

    def test_list_query_params_use_repeated_keys(self):
        url = _build_url(
            "/key-results",
            "https://api.example.com",
            query_params={"ids": [10, 11], "cycle": "Y2026Q2"},
        )

        assert url == "https://api.example.com/key-results?ids=10&ids=11&cycle=Y2026Q2"

    def test_path_params_are_one_encoded_segment(self):
        url = _build_url(
            "/users/{id}",
            "https://api.example.com/v1",
            path_params={"id": "1/../../admin?role=admin#"},
        )

        assert url == "https://api.example.com/v1/users/1%2F..%2F..%2Fadmin%3Frole%3Dadmin%23"

    @pytest.mark.asyncio
    async def test_unsafe_allowlist_matches_resolved_path(self):
        with patch("api_agent.rest.client.bounded_request") as client:
            result = await execute_request(
                "DELETE",
                "/users/../admin/wipe",
                base_url="https://api.example.com/v1",
                allow_unsafe_paths=["/users/*"],
            )

        assert "not allowed" in result["error"]
        client.assert_not_called()

    @pytest.mark.asyncio
    async def test_unsafe_allowlist_accepts_route_template_patterns(self):
        response = httpx.Response(204, request=httpx.Request("DELETE", "https://x"))
        with patch("api_agent.rest.client.bounded_request", return_value=response) as client:
            result = await execute_request(
                "DELETE",
                "/users/{id}",
                path_params={"id": 123},
                base_url="https://api.example.com/v1",
                allow_unsafe_paths=["/users/{id}"],
            )

        assert result["success"] is True
        client.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_allowed_with_matching_path(self):
        # POST is allowed when path matches allow_unsafe_paths
        result = await execute_request(
            "POST",
            "/search",
            base_url="https://api.example.com",
            body={"query": "test"},
            allow_unsafe_paths=["/search", "/_search"],
        )
        # Will fail with connection error (no real server) but NOT blocked
        assert "not allowed" not in result.get("error", "")

    @pytest.mark.asyncio
    async def test_post_blocked_with_non_matching_path(self):
        result = await execute_request(
            "POST",
            "/users",
            base_url="https://api.example.com",
            body={"name": "test"},
            allow_unsafe_paths=["/search"],
        )
        assert result["success"] is False
        assert "not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_post_allowed_with_glob_pattern(self):
        result = await execute_request(
            "POST",
            "/api/v1/search",
            base_url="https://api.example.com",
            body={"query": "test"},
            allow_unsafe_paths=["/api/*/search"],
        )
        # Will fail with connection error but NOT blocked
        assert "not allowed" not in result.get("error", "")

    @pytest.mark.asyncio
    async def test_nested_search_pattern(self):
        """Test nested wildcard pattern for search APIs."""
        # Should match /api/booking/search/v1/hotels
        result = await execute_request(
            "POST",
            "/api/booking/search/v1/hotels",
            base_url="https://api.example.com",
            body={"query": "test"},
            allow_unsafe_paths=["/api/booking/search/*"],
        )
        # Will fail with connection error but NOT blocked
        assert "not allowed" not in result.get("error", "")

    @pytest.mark.asyncio
    async def test_http_status_error_includes_status_code_and_details(self):
        request = httpx.Request("GET", "https://api.example.com/users")
        response = httpx.Response(404, request=request, json={"error": "missing"})

        with patch("api_agent.rest.client.bounded_request", return_value=response):
            result = await execute_request(
                "GET",
                "/users",
                base_url="https://api.example.com",
            )

        assert result["success"] is False
        assert result["error"] == "HTTP 404"
        assert result["status_code"] == 404
        assert result["details"] == "missing"
