"""Tests for GraphQL client behavior."""

import httpx
import pytest

from api_agent.graphql.client import execute_query


def _mock_response(json_data=None, raise_status=None):
    """Create mock response with optional error."""

    class _Response:
        def raise_for_status(self):
            if raise_status:
                raise raise_status
            return None

        def json(self):
            return json_data

    return _Response()


def _mock_client(response_or_fn):
    """Create mock async client that returns response or calls fn."""

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, endpoint, json, headers):
            if callable(response_or_fn):
                return response_or_fn(endpoint, json, headers)
            return response_or_fn

    return _Client()


@pytest.mark.asyncio
async def test_execute_query_requires_endpoint():
    result = await execute_query("query { users { id } }", endpoint="")
    assert result["success"] is False
    assert result["error"] == "No endpoint provided"


@pytest.mark.asyncio
async def test_execute_query_blocks_mutations():
    result = await execute_query("mutation { createUser(name: \"x\") { id } }", endpoint="https://api")
    assert result["success"] is False
    assert result["error"] == "Mutations are not allowed (read-only mode)"


@pytest.mark.asyncio
async def test_execute_query_success_includes_variables(monkeypatch):
    captured: dict = {}

    def _post(endpoint, json, headers):
        captured["endpoint"] = endpoint
        captured["json"] = json
        captured["headers"] = headers
        return _mock_response(json_data={"data": {"users": [{"id": 1}]}})

    client = _mock_client(_post)
    monkeypatch.setattr("api_agent.graphql.client.httpx.AsyncClient", lambda **_kwargs: client)
    result = await execute_query(
        "query GetUser($id: ID!) { user(id: $id) { id } }",
        variables={"id": "u1"},
        endpoint="https://api.example.com/graphql",
        headers={"Authorization": "Bearer t"},
    )

    assert result == {"success": True, "data": {"users": [{"id": 1}]}}
    assert captured["endpoint"] == "https://api.example.com/graphql"
    assert captured["json"]["variables"] == {"id": "u1"}
    assert captured["headers"]["Authorization"] == "Bearer t"


@pytest.mark.asyncio
async def test_execute_query_success_without_variables_omits_key(monkeypatch):
    captured: dict = {}

    def _post(endpoint, json, headers):
        captured["json"] = json
        return _mock_response(json_data={"data": {"ok": True}})

    client = _mock_client(_post)
    monkeypatch.setattr("api_agent.graphql.client.httpx.AsyncClient", lambda **_kwargs: client)
    result = await execute_query("query { ping }", endpoint="https://api.example.com/graphql")
    assert result == {"success": True, "data": {"ok": True}}
    assert "variables" not in captured["json"]


@pytest.mark.asyncio
async def test_execute_query_returns_graphql_errors(monkeypatch):
    response = _mock_response(json_data={"errors": [{"message": "bad field"}]})
    client = _mock_client(response)
    monkeypatch.setattr("api_agent.graphql.client.httpx.AsyncClient", lambda **_kwargs: client)
    result = await execute_query("query { badField }", endpoint="https://api.example.com/graphql")
    assert result["success"] is False
    assert result["error"] == [{"message": "bad field"}]


@pytest.mark.asyncio
async def test_execute_query_returns_http_status_error(monkeypatch):
    request = httpx.Request("POST", "https://api.example.com/graphql")
    response = httpx.Response(404, request=request)
    http_error = httpx.HTTPStatusError("Not found", request=request, response=response)
    mock_resp = _mock_response(raise_status=http_error)
    client = _mock_client(mock_resp)

    monkeypatch.setattr("api_agent.graphql.client.httpx.AsyncClient", lambda **_kwargs: client)
    result = await execute_query("query { users { id } }", endpoint="https://api.example.com/graphql")
    assert result["success"] is False
    assert result["error"] == "HTTP 404"
    assert result["status_code"] == 404


@pytest.mark.asyncio
async def test_execute_query_returns_generic_exception(monkeypatch):
    def _raise_error(endpoint, json, headers):
        raise RuntimeError("boom")

    client = _mock_client(_raise_error)
    monkeypatch.setattr("api_agent.graphql.client.httpx.AsyncClient", lambda **_kwargs: client)
    result = await execute_query("query { users { id } }", endpoint="https://api.example.com/graphql")
    assert result["success"] is False
    assert result["error"] == "boom"
