from contextvars import ContextVar
from types import SimpleNamespace

import pytest

from api_agent.agent.runtime import (
    AgentRunResult,
    AgentRuntimeConfig,
    AgentRuntimeState,
    LoadedSchema,
    _load_recipe_context,
    run_agent_query,
)
from api_agent.context import RequestContext


def _request_context() -> RequestContext:
    return RequestContext(
        target_url="https://spec",
        api_type="rest",
        target_headers={},
        allow_unsafe_paths=(),
        base_url="https://api",
        include_result=False,
        poll_paths=(),
    )


def _runtime_config(load_schema, log=lambda _msg: None) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        agent_name="test",
        agent_type="rest",
        call_key="calls",
        calls_var=ContextVar("calls"),
        recipe_steps_var=ContextVar("recipe_steps"),
        query_results_var=ContextVar("query_results"),
        last_result_var=ContextVar("last_result"),
        raw_schema_var=ContextVar("raw_schema"),
        load_schema=load_schema,
        build_tools=lambda _ctx, _state: [],
        build_prompt=lambda _ctx, _state: "",
        build_api_id=lambda _ctx, _state: "rest:https://spec|https://api",
        log=log,
        done_log_label="calls",
        exception_message="failed",
    )


@pytest.mark.asyncio
async def test_load_recipe_context_continues_when_lookup_fails(monkeypatch):
    monkeypatch.setattr("api_agent.agent.runtime.settings.ENABLE_RECIPES", True)

    async def load_schema(_ctx):
        return LoadedSchema(schema_context="")

    async def fail_search(*_args, **_kwargs):
        raise RuntimeError("redis down")

    monkeypatch.setattr("api_agent.agent.runtime.search_recipes", fail_search)

    logs = []
    state = AgentRuntimeState(schema_context="schema", raw_schema='{"openapi":"3.0.0"}')

    await _load_recipe_context(
        "list users", _request_context(), _runtime_config(load_schema, logs.append), state
    )

    assert state.suggestions == []
    assert state.recipe_context == ""
    assert logs == ["PRE-FLIGHT no matches for api_id=rest:https://spec|https://api"]


@pytest.mark.asyncio
async def test_run_agent_query_skips_learning_after_recipe_tool_use(monkeypatch):
    async def load_schema(_ctx):
        return LoadedSchema(schema_context="", raw_schema='{"openapi":"3.0.0"}')

    async def run_agent(_agent, _question, _config, _state):
        return AgentRunResult(
            output=SimpleNamespace(final_output="answer"),
            calls=[],
            last_data=[{"id": 1}],
            turn_info="turn 1/10",
            trace_id=None,
        )

    captured = {}

    async def maybe_extract(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("api_agent.agent.runtime._run_agent", run_agent)
    monkeypatch.setattr("api_agent.agent.runtime.maybe_extract_and_save_recipe", maybe_extract)
    monkeypatch.setattr("api_agent.agent.runtime.recipe_tool_was_used", lambda: True)
    monkeypatch.setattr("api_agent.agent.runtime.settings.ENABLE_RECIPES", True)

    result = await run_agent_query("list users", _request_context(), _runtime_config(load_schema))

    assert result["ok"] is True
    assert captured["skip_condition"] is True
    assert captured["original_result"] == [{"id": 1}]
    assert captured["validate_candidate"] is not None
