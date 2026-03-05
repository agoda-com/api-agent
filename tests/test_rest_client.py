"""Tests for REST client."""

from unittest.mock import patch

import httpx
import pytest

from api_agent.rest.client import _build_url, _is_path_allowed, execute_request


class TestIsPathAllowed:
    """Test path allowlist matching."""

    def test_exact_match(self):
        assert _is_path_allowed("/search", ["/search"]) is True

    def test_no_match(self):
        assert _is_path_allowed("/users", ["/search"]) is False

    def test_glob_star(self):
        assert _is_path_allowed("/api/v1/search", ["/api/*/search"]) is True
        assert _is_path_allowed("/api/v2/search", ["/api/*/search"]) is True
        assert _is_path_allowed("/api/search", ["/api/*/search"]) is False

    def test_multiple_patterns(self):
        patterns = ["/search", "/_search", "/api/*/query"]
        assert _is_path_allowed("/search", patterns) is True
        assert _is_path_allowed("/_search", patterns) is True
        assert _is_path_allowed("/api/v1/query", patterns) is True
        assert _is_path_allowed("/users", patterns) is False

    def test_empty_patterns(self):
        assert _is_path_allowed("/search", []) is False

    def test_nested_wildcard_pattern_matching(self):
        """Verify nested wildcard patterns match expected paths."""
        pattern = "/api/booking/search/*"

        # Should match
        assert _is_path_allowed("/api/booking/search/v1/hotels", [pattern]) is True
        assert _is_path_allowed("/api/booking/search/anything", [pattern]) is True

        # Should NOT match
        assert _is_path_allowed("/api/booking/search", [pattern]) is False
        assert _is_path_allowed("/api/booking/other", [pattern]) is False
        assert _is_path_allowed("/api/other/search/v1", [pattern]) is False


class TestBuildUrl:
    """Test URL building."""

    def test_simple_path(self):
        url = _build_url("/users", base_url="https://api.example.com")
        assert url == "https://api.example.com/users"

    def test_path_params(self):
        url = _build_url(
            "/users/{id}", base_url="https://api.example.com", path_params={"id": "123"}
        )
        assert url == "https://api.example.com/users/123"

    def test_query_params(self):
        url = _build_url(
            "/users", base_url="https://api.example.com", query_params={"limit": 10, "offset": 0}
        )
        assert "limit=10" in url
        assert "offset=0" in url

    def test_query_params_filters_none(self):
        url = _build_url(
            "/users", base_url="https://api.example.com", query_params={"limit": 10, "offset": None}
        )
        assert "limit=10" in url
        assert "offset" not in url

    def test_no_base_url_raises(self):
        with pytest.raises(ValueError, match="No base URL provided"):
            _build_url("/users", base_url="")


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
