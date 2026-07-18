"""Tests for REST client."""

from unittest.mock import patch

import httpx
import pytest

from api_agent.rest.client import _build_url, execute_request


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

    def test_list_query_params_use_repeated_keys(self):
        url = _build_url(
            "/key-results",
            "https://api.example.com",
            query_params={"ids": [10, 11], "cycle": "Y2026Q2"},
        )

        assert url == "https://api.example.com/key-results?ids=10&ids=11&cycle=Y2026Q2"

    def test_path_params_are_percent_encoded(self):
        url = _build_url(
            "/users/{user_id}/posts",
            "https://api.example.com/v1",
            path_params={"user_id": "alice/bob + team"},
        )

        assert url == "https://api.example.com/v1/users/alice%2Fbob%20%2B%20team/posts"

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

        class _Resp:
            def raise_for_status(self):
                raise httpx.HTTPStatusError("Not found", request=request, response=response)

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, *_args, **_kwargs):
                return _Resp()

        with patch("api_agent.rest.client.httpx.AsyncClient", return_value=_Client()):
            result = await execute_request(
                "GET",
                "/users",
                base_url="https://api.example.com",
            )

        assert result["success"] is False
        assert result["error"] == "HTTP 404"
        assert result["status_code"] == 404
        assert result["details"] == "missing"
