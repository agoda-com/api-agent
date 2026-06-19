"""Behavior tests for public agent query entrypoints."""

from types import SimpleNamespace
from typing import Literal

import pytest
from agents import ModelRefusalError

from api_agent.agent.graphql_agent import process_query
from api_agent.agent.rest_agent import process_rest_query
from api_agent.context import RequestContext


def _request_context(
    api_type: Literal["graphql", "rest"], *, base_url: str | None = None
) -> RequestContext:
    return RequestContext(
        target_url="https://api.example.com/schema",
        api_type=api_type,
        target_headers={},
        allow_unsafe_paths=(),
        base_url=base_url,
        include_result=False,
        poll_paths=(),
    )


@pytest.mark.asyncio
async def test_graphql_query_returns_agent_answer(monkeypatch):
    seen_query = ""

    async def fetch_schema(_query, _variables, _endpoint, _headers):
        return {"success": False, "error": "schema unavailable"}

    async def run_agent(_agent, query, max_turns, run_config):
        nonlocal seen_query
        seen_query = query
        return SimpleNamespace(final_output="answer")

    monkeypatch.setattr("api_agent.agent.graphql_agent.graphql_fetch", fetch_schema)
    monkeypatch.setattr("api_agent.agent.runtime.Runner.run", run_agent)
    monkeypatch.setattr("api_agent.agent.runtime.settings.ENABLE_RECIPES", False)

    result = await process_query("list users", _request_context("graphql"))

    assert seen_query == "list users"
    assert result == {
        "ok": True,
        "data": "answer",
        "result": None,
        "queries": [],
        "error": None,
    }


@pytest.mark.asyncio
async def test_rest_query_uses_loaded_schema_and_returns_agent_answer(monkeypatch):
    seen_query = ""

    async def fetch_schema_context(_target_url, _headers):
        return "<endpoints>\nGET /users", "https://api.example.com", '{"openapi":"3.0.0"}'

    async def run_agent(_agent, query, max_turns, run_config):
        nonlocal seen_query
        seen_query = query
        return SimpleNamespace(final_output="answer")

    monkeypatch.setattr("api_agent.agent.rest_agent.fetch_schema_context", fetch_schema_context)
    monkeypatch.setattr("api_agent.agent.runtime.Runner.run", run_agent)
    monkeypatch.setattr("api_agent.agent.runtime.settings.ENABLE_RECIPES", False)

    result = await process_rest_query("list users", _request_context("rest"))

    assert seen_query == "<endpoints>\nGET /users\n\nQuestion: list users"
    assert result == {
        "ok": True,
        "data": "answer",
        "result": None,
        "api_calls": [],
        "error": None,
    }


@pytest.mark.asyncio
async def test_rest_query_returns_clean_model_refusal(monkeypatch):
    async def fetch_schema_context(_target_url, _headers):
        return "<endpoints>\nGET /users", "https://api.example.com", '{"openapi":"3.0.0"}'

    async def run_agent(_agent, _query, max_turns, run_config):
        raise ModelRefusalError("cannot comply")

    monkeypatch.setattr("api_agent.agent.rest_agent.fetch_schema_context", fetch_schema_context)
    monkeypatch.setattr("api_agent.agent.runtime.Runner.run", run_agent)
    monkeypatch.setattr("api_agent.agent.runtime.settings.ENABLE_RECIPES", False)

    result = await process_rest_query("list users", _request_context("rest"))

    assert result == {
        "ok": False,
        "data": None,
        "api_calls": [],
        "error": "Model refused: cannot comply",
    }
