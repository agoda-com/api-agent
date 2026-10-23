"""Tests for endpoint-scoped schema caching and outage fallback."""

import json

import httpx
import pytest

from api_agent import config
from api_agent.agent import graphql_agent
from api_agent.rest import schema_loader
from api_agent.schema_cache import SchemaCache, schema_cache, schema_cache_key


class FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        return None


class FakeAsyncClient:
    def __init__(self, responses, calls):
        self._responses = iter(responses)
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url, *, headers):
        self._calls.append((url, dict(headers)))
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_cache_freshness_staleness_backoff_and_copy_isolation():
    now = [100.0]
    cache = SchemaCache(clock=lambda: now[0])
    cache.record_success("key", {"paths": {"/one": {}}})

    fresh = cache.get("key", ttl_seconds=10, stale_max_age_seconds=20)
    assert fresh is not None and fresh.stale is False
    fresh.value["paths"].clear()

    now[0] = 110.0
    stale = cache.get("key", ttl_seconds=10, stale_max_age_seconds=20)
    assert stale is not None and stale.stale is True
    assert "/one" in stale.value["paths"]

    cache.record_failure("key", backoff_seconds=5)
    assert cache.is_circuit_open("key") is True
    now[0] = 115.0
    assert cache.is_circuit_open("key") is False

    now[0] = 130.0
    assert cache.get("key", ttl_seconds=10, stale_max_age_seconds=20) is None


def test_schema_cache_key_scopes_headers_without_retaining_them():
    first = schema_cache_key("rest", "http://127.0.0.1/spec", {"Authorization": "Bearer one"})
    same = schema_cache_key("rest", "http://127.0.0.1/spec", {"authorization": "Bearer one"})
    other = schema_cache_key("rest", "http://127.0.0.1/spec", {"Authorization": "Bearer two"})

    assert first == same
    assert first != other
    assert "Bearer one" not in first


@pytest.mark.asyncio
async def test_openapi_loader_uses_stale_schema_and_suppresses_retries(monkeypatch):
    schema_cache.clear()
    monkeypatch.setattr(config.settings, "SCHEMA_CACHE_TTL_SECONDS", 0)
    monkeypatch.setattr(config.settings, "SCHEMA_STALE_MAX_AGE_SECONDS", 300)
    monkeypatch.setattr(config.settings, "SCHEMA_FAILURE_BACKOFF_SECONDS", 30)

    spec = {"openapi": "3.0.0", "info": {"title": "Synthetic", "version": "1"}}
    calls = []
    responses = [
        FakeResponse(json.dumps(spec)),
        httpx.ConnectError("synthetic offline response"),
        httpx.ConnectError("must not be reached during backoff"),
        httpx.ConnectError("synthetic second auth context is offline"),
    ]
    response_iter = iter(responses)
    monkeypatch.setattr(
        schema_loader.httpx,
        "AsyncClient",
        lambda **_kwargs: FakeAsyncClient(response_iter, calls),
    )

    url = "http://127.0.0.1:8000/openapi.json"
    headers = {"Authorization": "Bearer synthetic-one"}
    assert await schema_loader.load_openapi_spec(url, headers) == spec
    assert await schema_loader.load_openapi_spec(url, headers) == spec
    assert schema_cache.is_circuit_open(schema_cache_key("rest", url, headers))
    assert await schema_loader.load_openapi_spec(url, headers) == spec
    assert len(calls) == 2

    # A different auth context cannot reuse the first context's cached schema.
    assert (
        await schema_loader.load_openapi_spec(url, {"Authorization": "Bearer synthetic-two"}) == {}
    )
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_graphql_loader_falls_back_to_cached_introspection(monkeypatch):
    schema_cache.clear()
    monkeypatch.setattr(config.settings, "SCHEMA_CACHE_TTL_SECONDS", 0)
    monkeypatch.setattr(config.settings, "SCHEMA_STALE_MAX_AGE_SECONDS", 300)
    monkeypatch.setattr(config.settings, "SCHEMA_FAILURE_BACKOFF_SECONDS", 30)
    calls = []
    schema = {
        "queryType": {"name": "Query"},
        "types": [{"kind": "OBJECT", "name": "Query", "fields": []}],
    }
    results = [
        {"success": True, "data": {"__schema": schema}},
        {"success": False, "error": "synthetic endpoint unavailable"},
        {"success": False, "error": "must not be reached during backoff"},
    ]

    async def fetch(*_args):
        calls.append(1)
        return results.pop(0)

    monkeypatch.setattr(graphql_agent, "graphql_fetch", fetch)
    endpoint = "http://127.0.0.1:8001/graphql"
    headers = {"Authorization": "Bearer synthetic"}

    expected = json.dumps(schema, indent=2)
    assert await graphql_agent.fetch_graphql_schema_raw(endpoint, headers) == expected
    assert await graphql_agent.fetch_graphql_schema_raw(endpoint, headers) == expected
    assert await graphql_agent.fetch_graphql_schema_raw(endpoint, headers) == expected
    assert len(calls) == 2
