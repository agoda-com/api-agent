"""Shared agent runtime orchestration."""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from agents import Agent, MaxTurnsExceeded, ModelRefusalError, Runner

from ..config import settings
from ..context import RequestContext
from ..recipe.execution import build_partial_result
from ..recipe.learning import maybe_extract_and_save_recipe
from ..recipe.search import search_recipes
from ..recipe.state import (
    _return_directly_flag,
    _tools_to_final_output,
    recipe_tool_was_used,
    reset_recipe_tool_usage,
)
from ..tracing import agent_span_attributes, span_trace_id, trace_metadata, trace_span
from .contextvar_utils import safe_get_contextvar
from .model import get_run_config, model
from .progress import get_turn_context, reset_progress

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedSchema:
    """Protocol-specific schema load result."""

    schema_context: str
    raw_schema: str = ""
    base_url: str = ""
    early_response: dict[str, Any] | None = None


@dataclass
class AgentRuntimeState:
    """Per-run state shared with protocol adapters."""

    schema_context: str
    raw_schema: str
    base_url: str = ""
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    recipe_context: str = ""


@dataclass(frozen=True)
class AgentRunResult:
    """Normalized result from one Agents SDK run."""

    output: Any | None
    calls: list[Any]
    last_data: Any
    turn_info: str
    trace_id: str | None
    error: str | None = None


@dataclass(frozen=True)
class AgentRuntimeConfig:
    """Protocol adapter hooks for the shared runner."""

    agent_name: str
    agent_type: str
    call_key: str
    calls_var: ContextVar[list[Any]]
    recipe_steps_var: ContextVar[list[dict[str, Any]]]
    query_results_var: ContextVar[dict[str, Any]]
    last_result_var: ContextVar[list[Any]]
    raw_schema_var: ContextVar[str]
    load_schema: Callable[[RequestContext], Awaitable[LoadedSchema]]
    build_tools: Callable[[RequestContext, AgentRuntimeState], list[Any]]
    build_prompt: Callable[[RequestContext, AgentRuntimeState], str]
    build_api_id: Callable[[RequestContext, AgentRuntimeState], str]
    log: Callable[[str], None]
    done_log_label: str
    exception_message: str
    skip_recipe: Callable[[RequestContext, AgentRuntimeState], bool] = lambda _ctx, _state: False
    validate_recipe_candidate: (
        Callable[
            [RequestContext, AgentRuntimeState, dict[str, Any], dict[str, Any]], Awaitable[Any]
        ]
        | None
    ) = None


async def run_agent_query(
    question: str,
    ctx: RequestContext,
    config: AgentRuntimeConfig,
) -> dict[str, Any]:
    """Run one protocol-specific agent through the shared lifecycle."""
    try:
        config.log(f"QUERY {question[:80]}")
        _reset_runtime(config)

        loaded = await config.load_schema(ctx)
        config.raw_schema_var.set(loaded.raw_schema)
        if loaded.early_response is not None:
            return loaded.early_response

        state = AgentRuntimeState(
            schema_context=loaded.schema_context,
            raw_schema=loaded.raw_schema,
            base_url=loaded.base_url,
        )
        await _load_recipe_context(question, ctx, config, state)

        agent = Agent(
            name=config.agent_name,
            model=model,
            instructions=config.build_prompt(ctx, state),
            tools=config.build_tools(ctx, state),
            tool_use_behavior=_tools_to_final_output,
        )

        run = await _run_agent(agent, question, config, state)
        if run.error:
            return _with_trace_id(
                {
                    "ok": False,
                    "data": None,
                    config.call_key: run.calls,
                    "error": run.error,
                },
                run.trace_id,
            )
        if run.output is None:
            return _with_trace_id(
                build_partial_result(run.last_data, run.calls, run.turn_info, config.call_key),
                run.trace_id,
            )

        is_direct_return = _is_direct_return(run.output.final_output)
        if not run.output.final_output and not is_direct_return:
            return _with_trace_id(
                _empty_output_result(run.last_data, run.calls, run.turn_info, config.call_key),
                run.trace_id,
            )

        agent_output = None if is_direct_return else str(run.output.final_output)
        if agent_output is not None:
            config.log(f"DONE {config.done_log_label}={len(run.calls)} output={agent_output[:100]}")

        async def validate_candidate(recipe: dict[str, Any], tool_args: dict[str, Any]) -> Any:
            if config.validate_recipe_candidate is None:
                return None
            return await config.validate_recipe_candidate(ctx, state, recipe, tool_args)

        await maybe_extract_and_save_recipe(
            api_type=config.agent_type,
            api_id=config.build_api_id(ctx, state),
            question=question,
            steps=safe_get_contextvar(config.recipe_steps_var, []),
            raw_schema=safe_get_contextvar(config.raw_schema_var, ""),
            skip_condition=config.skip_recipe(ctx, state) or recipe_tool_was_used(),
            learn_rate=ctx.learning_rate,
            strong_recipe_match=any(s.get("score", 0) >= 0.8 for s in state.suggestions),
            original_result=run.last_data,
            validate_candidate=validate_candidate,
        )

        return _with_trace_id(
            {
                "ok": True,
                "data": agent_output,
                "result": run.last_data,
                config.call_key: run.calls,
                "error": None,
            },
            run.trace_id,
        )

    except Exception as e:
        logger.exception(config.exception_message)
        return {
            "ok": False,
            "data": None,
            config.call_key: [],
            "error": str(e),
        }


def _reset_runtime(config: AgentRuntimeConfig) -> None:
    config.calls_var.set([])
    config.recipe_steps_var.set([])
    config.query_results_var.set({})
    config.last_result_var.set([None])
    config.raw_schema_var.set("")
    _return_directly_flag.set([])
    reset_recipe_tool_usage()
    reset_progress()


async def _load_recipe_context(
    question: str,
    ctx: RequestContext,
    config: AgentRuntimeConfig,
    state: AgentRuntimeState,
) -> None:
    if not settings.ENABLE_RECIPES:
        return

    api_id = config.build_api_id(ctx, state)
    try:
        suggestions, recipe_context = await search_recipes(api_id, state.raw_schema, question)
    except Exception:
        logger.exception(
            "Recipe lookup failed; continuing without recipes api_id=%s agent_type=%s",
            api_id[:100],
            config.agent_type,
        )
        suggestions, recipe_context = [], ""
    state.suggestions = suggestions
    state.recipe_context = recipe_context

    if suggestions:
        config.log(
            f"PRE-FLIGHT found={len(suggestions)} ids={[s['recipe_id'] for s in suggestions]}"
        )
    elif state.raw_schema:
        config.log(f"PRE-FLIGHT no matches for api_id={api_id[:50]}")


async def _run_agent(
    agent: Agent,
    question: str,
    config: AgentRuntimeConfig,
    state: AgentRuntimeState,
) -> AgentRunResult:
    augmented_query = (
        f"{state.schema_context}\n\nQuestion: {question}" if state.schema_context else question
    )

    trace_id = None
    with trace_span(
        f"{config.agent_type}.query",
        agent_span_attributes(settings.MCP_SLUG, config.agent_type),
    ) as span:
        trace_id = span_trace_id(span)
        with trace_metadata({"mcp_name": settings.MCP_SLUG, "agent_type": config.agent_type}):
            try:
                result = await Runner.run(
                    agent,
                    augmented_query,
                    max_turns=settings.MAX_AGENT_TURNS,
                    run_config=get_run_config(),
                )
            except ModelRefusalError as e:
                return AgentRunResult(
                    output=None,
                    calls=[],
                    last_data=config.last_result_var.get()[0],
                    turn_info=get_turn_context(settings.MAX_AGENT_TURNS),
                    trace_id=trace_id,
                    error=f"Model refused: {e.refusal}",
                )
            except MaxTurnsExceeded:
                return AgentRunResult(
                    output=None,
                    calls=config.calls_var.get(),
                    last_data=config.last_result_var.get()[0],
                    turn_info=get_turn_context(settings.MAX_AGENT_TURNS),
                    trace_id=trace_id,
                )

    return AgentRunResult(
        output=result,
        calls=config.calls_var.get(),
        last_data=config.last_result_var.get()[0],
        turn_info=get_turn_context(settings.MAX_AGENT_TURNS),
        trace_id=trace_id,
    )


def _is_direct_return(final_output: Any) -> bool:
    try:
        return final_output == "__DIRECT_RETURN__" or bool(_return_directly_flag.get())
    except LookupError:
        return False


def _empty_output_result(
    last_data: Any,
    calls: list[Any],
    turn_info: str,
    call_key: str,
) -> dict[str, Any]:
    if last_data:
        return {
            "ok": True,
            "data": f"[Partial - {turn_info}] Data retrieved but agent didn't complete.",
            "result": last_data,
            call_key: calls,
            "error": None,
        }
    return {
        "ok": False,
        "data": None,
        "result": None,
        call_key: calls,
        "error": f"No output ({turn_info})",
    }


def _with_trace_id(payload: dict[str, Any], trace_id: str | None) -> dict[str, Any]:
    if trace_id:
        payload["trace_id"] = trace_id
    return payload
