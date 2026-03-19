"""Tests for Auth0 token middleware."""

import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from api_agent.auth import (
    _get_nested_value,
    _token_cache,
    fetch_auth_token,
    resolve_headers,
)
from api_agent.context import RequestContext


def _make_ctx(
    target_headers: dict | None = None,
    auth_url: str | None = None,
    auth_body: dict | None = None,
    auth_token_path: str = "access_token",
) -> RequestContext:
    """Helper to create RequestContext with auth fields."""
    return RequestContext(
        target_url="https://api.example.com/openapi.json",
        api_type="rest",
        target_headers=target_headers or {"X-Custom": "existing"},
        allow_unsafe_paths=(),
        base_url=None,
        include_result=False,
        poll_paths=(),
        auth_url=auth_url,
        auth_body=auth_body,
        auth_token_path=auth_token_path,
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear token cache before each test."""
    _token_cache.clear()
    yield
    _token_cache.clear()


class TestGetNestedValue:
    """Test dot-path extraction."""

    def test_simple_key(self):
        assert _get_nested_value({"access_token": "abc"}, "access_token") == "abc"

    def test_nested_key(self):
        assert _get_nested_value({"data": {"token": "xyz"}}, "data.token") == "xyz"

    def test_missing_key(self):
        assert _get_nested_value({"foo": "bar"}, "access_token") is None

    def test_none_data(self):
        assert _get_nested_value(None, "access_token") is None

    def test_empty_path(self):
        assert _get_nested_value({"a": 1}, "") is None

    def test_list_index(self):
        assert _get_nested_value({"items": ["a", "b"]}, "items.1") == "b"


class TestFetchAuthToken:
    """Test token fetching from Auth0."""

    @pytest.mark.asyncio
    async def test_success(self):
        mock_response = httpx.Response(
            200,
            json={"access_token": "test-token-123"},
            request=httpx.Request("POST", "https://auth0.example.com/token"),
        )
        with patch("api_agent.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            token = await fetch_auth_token(
                "https://auth0.example.com/token",
                auth_body={"grant_type": "client_credentials"},
            )

            assert token == "test-token-123"
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_nested_token_path(self):
        mock_response = httpx.Response(
            200,
            json={"data": {"token": "nested-token"}},
            request=httpx.Request("POST", "https://auth0.example.com/token"),
        )
        with patch("api_agent.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            token = await fetch_auth_token(
                "https://auth0.example.com/token",
                token_path="data.token",
            )

            assert token == "nested-token"

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Cached token returned without HTTP call."""
        _token_cache["https://auth0.example.com/token"] = (
            "cached-token",
            time.monotonic() + 300,
        )

        with patch("api_agent.auth.httpx.AsyncClient") as mock_client_cls:
            token = await fetch_auth_token("https://auth0.example.com/token")

            assert token == "cached-token"
            mock_client_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_expiry(self):
        """Expired cache triggers new fetch."""
        _token_cache["https://auth0.example.com/token"] = (
            "old-token",
            time.monotonic() - 1,  # expired
        )

        mock_response = httpx.Response(
            200,
            json={"access_token": "fresh-token"},
            request=httpx.Request("POST", "https://auth0.example.com/token"),
        )
        with patch("api_agent.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            token = await fetch_auth_token("https://auth0.example.com/token")

            assert token == "fresh-token"
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_auth_failure_returns_none(self):
        """Auth endpoint error returns None."""
        with patch("api_agent.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            token = await fetch_auth_token("https://auth0.example.com/token")

            assert token is None

    @pytest.mark.asyncio
    async def test_token_not_found_returns_none(self):
        """Missing token path in response returns None."""
        mock_response = httpx.Response(
            200,
            json={"error": "no token here"},
            request=httpx.Request("POST", "https://auth0.example.com/token"),
        )
        with patch("api_agent.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            token = await fetch_auth_token("https://auth0.example.com/token")

            assert token is None


class TestResolveHeaders:
    """Test header resolution with auth token injection."""

    @pytest.mark.asyncio
    async def test_passthrough_no_auth_url(self):
        """No auth_url returns target_headers unchanged."""
        ctx = _make_ctx(target_headers={"X-Custom": "value"})
        headers = await resolve_headers(ctx)
        assert headers == {"X-Custom": "value"}

    @pytest.mark.asyncio
    async def test_injects_bearer_token(self):
        """Auth token injected as Authorization: Bearer."""
        _token_cache["https://auth0.example.com/token"] = (
            "my-token",
            time.monotonic() + 300,
        )

        ctx = _make_ctx(
            auth_url="https://auth0.example.com/token",
            target_headers={"X-Custom": "value"},
        )
        headers = await resolve_headers(ctx)

        assert headers["Authorization"] == "Bearer my-token"
        assert headers["X-Custom"] == "value"

    @pytest.mark.asyncio
    async def test_preserves_existing_headers(self):
        """Existing target_headers are preserved alongside auth token."""
        _token_cache["https://auth0.example.com/token"] = (
            "tok",
            time.monotonic() + 300,
        )

        ctx = _make_ctx(
            auth_url="https://auth0.example.com/token",
            target_headers={"X-Api-Key": "secret", "Accept": "application/json"},
        )
        headers = await resolve_headers(ctx)

        assert headers["X-Api-Key"] == "secret"
        assert headers["Accept"] == "application/json"
        assert headers["Authorization"] == "Bearer tok"

    @pytest.mark.asyncio
    async def test_failure_returns_original_headers(self):
        """Failed token fetch returns original headers without Authorization."""
        with patch("api_agent.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("fail")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            ctx = _make_ctx(
                auth_url="https://auth0.example.com/token",
                target_headers={"X-Custom": "value"},
            )
            headers = await resolve_headers(ctx)

            assert headers == {"X-Custom": "value"}
            assert "Authorization" not in headers
