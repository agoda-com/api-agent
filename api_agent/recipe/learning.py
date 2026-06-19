"""Recipe extraction and persistence."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ..config import settings
from ..store import ASYNC_API_AGENT_STORE, sha256_hex
from ..tracing import trace_span
from .contracts import (
    get_recipe_tool_args,
    get_recipe_tool_name,
    get_validation_tool_args,
    has_recipe_contract,
    results_equivalent,
)
from .execution import error_json, validate_recipe_params
from .extractor import extract_recipe
from .identity import recipe_behavior_payload
from .state import mark_recipe_changed
from .tooling import deduplicate_tool_name

logger = logging.getLogger(__name__)


_MAX_RECIPE_REPAIR_ATTEMPTS = 2
_VALIDATION_SAMPLE_ROWS = 5
_SKIP_REASON_MESSAGES = {
    "disabled": "disabled",
    "skip_condition": "skip condition",
    "strong_recipe_match": "strong recipe match",
    "missing_steps": "missing steps",
    "missing_schema": "missing schema",
    "sampled_out": "sampled out",
    "extractor_rejected": "extractor rejected",
    "missing_validation_runner": "missing validation runner",
    "candidate_result_mismatch": "candidate result mismatch",
}


def _recipes_equivalent(existing: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return recipe_behavior_payload(existing) == recipe_behavior_payload(candidate)


async def maybe_extract_and_save_recipe(
    api_type: str,
    api_id: str,
    question: str,
    steps: list,
    raw_schema: str,
    skip_condition: bool = False,
    learn_rate: float | None = None,
    strong_recipe_match: bool = False,
    original_result: Any | None = None,
    validate_candidate: Callable[[dict[str, Any], dict[str, Any]], Awaitable[Any]] | None = None,
) -> None:
    """Extract and save recipe if conditions met."""
    span_attrs = {
        "recipe.api_type": api_type,
        "recipe.api_id": api_id[:100],
        "recipe.step_count": len(steps),
        "recipe.has_schema": bool(raw_schema),
    }
    with trace_span("recipe.learning", span_attrs) as span:
        await _maybe_extract_and_save_recipe(
            api_type=api_type,
            api_id=api_id,
            question=question,
            steps=steps,
            raw_schema=raw_schema,
            skip_condition=skip_condition,
            learn_rate=learn_rate,
            strong_recipe_match=strong_recipe_match,
            original_result=original_result,
            validate_candidate=validate_candidate,
            span=span,
        )


async def _maybe_extract_and_save_recipe(
    *,
    api_type: str,
    api_id: str,
    question: str,
    steps: list,
    raw_schema: str,
    skip_condition: bool,
    learn_rate: float | None,
    strong_recipe_match: bool,
    original_result: Any | None,
    validate_candidate: Callable[[dict[str, Any], dict[str, Any]], Awaitable[Any]] | None,
    span: Any,
) -> None:
    if not settings.ENABLE_RECIPES:
        _skip_recipe_learning(span, "disabled")
        return
    if skip_condition:
        _skip_recipe_learning(span, "skip_condition")
        return
    effective_learn_rate = settings.RECIPE_LEARN_RATE if learn_rate is None else learn_rate
    _set_recipe_span_attributes(span, {"recipe.learn_rate": effective_learn_rate})
    if strong_recipe_match and effective_learn_rate < 1:
        _skip_recipe_learning(span, "strong_recipe_match")
        return
    if not steps:
        _skip_recipe_learning(span, "missing_steps")
        return
    if not raw_schema:
        _skip_recipe_learning(span, "missing_schema")
        return

    schema_hash = ""
    try:
        schema_hash = sha256_hex(raw_schema)
        _set_recipe_span_attributes(span, {"recipe.schema_hash": schema_hash[:12]})
        if not should_learn_recipe(
            api_id=api_id,
            schema_hash=schema_hash,
            question=question,
            learn_rate=effective_learn_rate,
        ):
            _skip_recipe_learning(span, "sampled_out")
            return

        existing_recipes = [
            recipe
            for recipe in await ASYNC_API_AGENT_STORE.list_recipes(
                api_id=api_id, schema_hash=schema_hash
            )
            if has_recipe_contract(recipe)
        ]
        extractor_existing_recipes = _extractor_existing_recipes_context(existing_recipes)
        _set_recipe_span_attributes(span, {"recipe.existing_count": len(existing_recipes)})
        learning_steps = _recipe_steps_for_learning(steps, span)
        validation_result = _validation_result_from_steps(learning_steps, original_result)
        extractor_steps = _extractor_steps_context(learning_steps)
        extractor_result = _extractor_result_context(validation_result)
        recipe: dict[str, Any] | None = None
        candidate_result: Any | None = None
        validation_feedback: dict[str, Any] | None = None
        for validation_attempt in range(_MAX_RECIPE_REPAIR_ATTEMPTS + 1):
            _set_recipe_span_attributes(
                span,
                {"recipe.validation_attempt": validation_attempt + 1},
            )
            recipe = await extract_recipe(
                api_type=api_type,
                question=question,
                steps=extractor_steps,
                result=extractor_result,
                existing_recipes=extractor_existing_recipes,
                validation_feedback=validation_feedback,
            )
            if not recipe:
                break
            if not validate_candidate:
                _skip_recipe_learning(span, "missing_validation_runner")
                return

            fixture_args = get_validation_tool_args(recipe)
            candidate_result = await validate_candidate(recipe, fixture_args)
            if results_equivalent(candidate_result, validation_result):
                break

            validation_feedback = _record_validation_mismatch(
                span=span,
                validation_attempt=validation_attempt + 1,
                recipe=recipe,
                fixture_args=fixture_args,
                candidate_result=candidate_result,
                expected_result=validation_result,
            )

        if not recipe:
            _skip_recipe_learning(span, "extractor_rejected")
            return

        if not results_equivalent(candidate_result, validation_result):
            _skip_recipe_learning(
                span,
                "candidate_result_mismatch",
                candidate_rows=_row_count(candidate_result),
                expected_rows=_row_count(validation_result),
            )
            return
        recipe.pop("validation_fixture", None)

        for existing in existing_recipes:
            if _recipes_equivalent(existing, recipe):
                if existing.get("description"):
                    _record_recipe_learning(span, "duplicate", recipe_id=existing.get("recipe_id"))
                    return
                tool_name = existing.get("tool_name") or get_recipe_tool_name(existing)
                if tool_name:
                    recipe["public_contract"]["tool_name"] = tool_name
                recipe_id = await ASYNC_API_AGENT_STORE.save_or_touch(
                    api_id=api_id,
                    schema_hash=schema_hash,
                    question=question,
                    recipe=recipe,
                    tool_name=recipe["public_contract"]["tool_name"],
                )
                mark_recipe_changed(recipe_id)
                _record_recipe_learning(span, "updated", recipe_id=recipe_id)
                return

        seen: set[str] = {r["tool_name"] for r in existing_recipes if r.get("tool_name")}
        tool_name = deduplicate_tool_name(get_recipe_tool_name(recipe), seen_names=seen, max_len=40)
        recipe["public_contract"]["tool_name"] = tool_name
        recipe_id = await ASYNC_API_AGENT_STORE.save_or_touch(
            api_id=api_id,
            schema_hash=schema_hash,
            question=question,
            recipe=recipe,
            tool_name=tool_name,
        )
        mark_recipe_changed(recipe_id)
        _record_recipe_learning(span, "saved", recipe_id=recipe_id, tool_name=tool_name)
    except Exception:
        _record_recipe_learning(span, "error", reason="exception", schema_hash=schema_hash[:12])
        logger.exception(
            "Recipe extraction failed api_type=%s api_id=%s schema_hash=%s",
            api_type,
            api_id[:100],
            schema_hash[:12],
        )


def _recipe_steps_for_learning(steps: list[Any], span: Any) -> list[Any]:
    learning_steps = _drop_empty_intermediate_steps(steps)
    pruned_count = len(steps) - len(learning_steps)
    if pruned_count:
        attributes = {"recipe.pruned_step_count": pruned_count}
        _set_recipe_span_attributes(span, attributes)
        _add_recipe_span_event(span, "recipe.learning.pruned_steps", attributes)
        logger.info("Pruned recipe learning dead-end steps count=%s", pruned_count)
    return learning_steps


def _extractor_steps_context(steps: list[Any]) -> list[Any]:
    return [_extractor_step_context(step) for step in steps]


def _extractor_step_context(step: Any) -> Any:
    if not isinstance(step, dict):
        return step
    context = dict(step)
    if "result" in context:
        context["result"] = _extractor_result_context(context["result"])
    return context


def _extractor_result_context(value: Any) -> Any:
    if isinstance(value, list) and len(value) > _VALIDATION_SAMPLE_ROWS:
        return {
            "row_count": len(value),
            "sample": value[:_VALIDATION_SAMPLE_ROWS],
            "truncated": True,
        }
    return value


def _extractor_existing_recipes_context(
    existing_recipes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    for recipe in existing_recipes:
        public_contract = recipe.get("public_contract")
        entry: dict[str, Any] = {
            "tool_name": recipe.get("tool_name") or get_recipe_tool_name(recipe),
            "question": recipe.get("question", ""),
            "description": recipe.get("description", ""),
            "tool_args": get_recipe_tool_args(recipe),
        }
        if isinstance(public_contract, dict):
            entry["public_contract"] = {
                "tool_name": public_contract.get("tool_name"),
                "description": public_contract.get("description"),
                "tool_args": dict(get_recipe_tool_args(recipe)),
            }
        if recipe.get("fingerprint"):
            entry["fingerprint"] = recipe.get("fingerprint")
        context.append(entry)
    return context


def _drop_empty_intermediate_steps(steps: list[Any]) -> list[Any]:
    if len(steps) < 2:
        return steps

    kept: list[Any] = []
    later_has_rows = False
    for step in reversed(steps):
        is_dead_end_sql = (
            isinstance(step, dict)
            and step.get("kind") == "sql"
            and step.get("result") == []
            and later_has_rows
        )
        if not is_dead_end_sql:
            kept.append(step)
        later_has_rows = later_has_rows or _step_has_rows(step)
    kept.reverse()
    return kept


def _step_has_rows(step: Any) -> bool:
    return isinstance(step, dict) and bool(step.get("result"))


def _skip_recipe_learning(span: Any, reason: str, **attributes: Any) -> None:
    _record_recipe_learning(span, "skipped", reason=reason, **attributes)
    details = " ".join(
        f"{key}={_span_attribute_value(value)}"
        for key, value in attributes.items()
        if value is not None
    )
    suffix = f" {details}" if details else ""
    logger.info(
        "Skipping recipe extraction (%s)%s",
        _SKIP_REASON_MESSAGES.get(reason, reason.replace("_", " ")),
        suffix,
    )


def _record_recipe_learning(span: Any, outcome: str, **attributes: Any) -> None:
    event_attributes = {"recipe.learning.outcome": outcome}
    reason = attributes.pop("reason", None)
    if reason is not None:
        reason_key = (
            "recipe.learning.skip_reason" if outcome == "skipped" else "recipe.learning.reason"
        )
        event_attributes[reason_key] = reason
    for key, value in attributes.items():
        attr_key = key if key.startswith("recipe.") else f"recipe.{key}"
        event_attributes[attr_key] = value

    _set_recipe_span_attributes(span, event_attributes)
    _add_recipe_span_event(span, f"recipe.learning.{outcome}", event_attributes)


def _set_recipe_span_attributes(span: Any, attributes: dict[str, Any]) -> None:
    if span is None:
        return
    for key, value in attributes.items():
        if value is None:
            continue
        try:
            span.set_attribute(key, _span_attribute_value(value))
        except Exception:
            logger.debug("Failed to set recipe span attribute %s", key, exc_info=True)


def _add_recipe_span_event(span: Any, name: str, attributes: dict[str, Any]) -> None:
    if span is None:
        return
    try:
        span.add_event(
            name,
            {key: _span_attribute_value(value) for key, value in attributes.items()},
        )
    except Exception:
        logger.debug("Failed to add recipe span event %s", name, exc_info=True)


def _span_attribute_value(value: Any) -> bool | int | float | str:
    if isinstance(value, (bool, int, float, str)):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _row_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if value is None:
        return 0
    return 1


def _record_validation_mismatch(
    *,
    span: Any,
    validation_attempt: int,
    recipe: dict[str, Any],
    fixture_args: dict[str, Any],
    candidate_result: Any,
    expected_result: Any,
) -> dict[str, Any]:
    feedback = _build_validation_feedback(
        recipe=recipe,
        fixture_args=fixture_args,
        candidate_result=candidate_result,
        expected_result=expected_result,
    )
    _add_recipe_span_event(
        span,
        "recipe.learning.validation_mismatch",
        {
            "recipe.validation_attempt": validation_attempt,
            "recipe.candidate_rows": _row_count(candidate_result),
            "recipe.expected_rows": _row_count(expected_result),
        },
    )
    return feedback


def _build_validation_feedback(
    *,
    recipe: dict[str, Any],
    fixture_args: dict[str, Any],
    candidate_result: Any,
    expected_result: Any,
) -> dict[str, Any]:
    return {
        "reason": "candidate_result_mismatch",
        "previous_candidate": recipe,
        "fixture_args": fixture_args,
        "candidate_result": {
            "row_count": _row_count(candidate_result),
            "sample": _sample_result(candidate_result),
        },
        "expected_result": {
            "row_count": _row_count(expected_result),
            "sample": _sample_result(expected_result),
        },
    }


def _sample_result(value: Any) -> Any:
    if isinstance(value, list):
        return value[:_VALIDATION_SAMPLE_ROWS]
    return value


def _validation_result_from_steps(steps: list[Any], fallback: Any) -> Any:
    combined = _combined_terminal_step_results(steps)
    return combined if combined is not None else fallback


def _combined_terminal_step_results(steps: list[Any]) -> list[Any] | None:
    if not steps:
        return None

    last_key = _repeatable_step_key(steps[-1])
    if last_key is None:
        return None

    grouped: list[dict[str, Any]] = []
    for step in reversed(steps):
        if not isinstance(step, dict) or _repeatable_step_key(step) != last_key:
            break
        if not isinstance(step.get("result"), list):
            return None
        grouped.append(step)

    if len(grouped) < 2:
        return None

    rows: list[Any] = []
    for step in reversed(grouped):
        rows.extend(step["result"])
    return rows


def _repeatable_step_key(step: Any) -> tuple[str, str, str] | None:
    if not isinstance(step, dict):
        return None
    if step.get("kind") != "rest":
        return None
    method = step.get("method")
    path = step.get("path")
    if not isinstance(method, str) or not isinstance(path, str):
        return None
    return ("rest", method.upper(), path)


def should_learn_recipe(
    *,
    api_id: str,
    schema_hash: str,
    question: str,
    learn_rate: float,
) -> bool:
    """Deterministically sample recipe extraction."""
    try:
        rate = float(learn_rate)
    except (TypeError, ValueError):
        rate = 0.0
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    seed = sha256_hex(f"{api_id}|{schema_hash}|{question}")
    bucket = int(seed[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return bucket < rate


async def async_validate_and_prepare_recipe(
    recipe_id: str,
    params_json: str,
    raw_schema_var,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    """Async-safe recipe validation for request/tool execution paths."""
    if not _raw_schema_from_var(raw_schema_var):
        return None, None, error_json("schema not loaded")

    recipe = await ASYNC_API_AGENT_STORE.get_recipe(recipe_id)
    return _validate_loaded_recipe(recipe, recipe_id, params_json)


def _raw_schema_from_var(raw_schema_var) -> str:
    try:
        return raw_schema_var.get()
    except LookupError:
        return ""


def _validate_loaded_recipe(
    recipe: dict[str, Any] | None,
    recipe_id: str,
    params_json: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    if not recipe or not has_recipe_contract(recipe):
        return None, None, error_json(f"recipe not found: {recipe_id}")

    provided, err = _parse_recipe_params(params_json)
    if err:
        return None, None, err

    validated, err = validate_recipe_params(get_recipe_tool_args(recipe), provided)
    if err:
        return None, None, err
    return recipe, validated, ""


def _parse_recipe_params(params_json: str) -> tuple[dict[str, Any], str]:
    provided: dict[str, Any] = {}
    if params_json:
        try:
            provided = json.loads(params_json)
        except json.JSONDecodeError as e:
            return {}, error_json(f"invalid params_json: {e.msg}")
    return provided, ""
