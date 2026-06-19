"""LLM-assisted extraction of parameterized recipes from successful executions."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agents import Agent, AgentOutputSchema, Runner
from agents.exceptions import ModelBehaviorError

from .contracts import (
    get_recipe_description,
    get_recipe_steps,
    get_recipe_tool_args,
    get_recipe_tool_name,
    get_validation_tool_args,
    validate_recipe_contract,
)
from .extractor_models import ExtractedRecipeOutput
from .extractor_prompts import build_extractor_instructions

logger = logging.getLogger(__name__)


def _structured_recipe(output: Any) -> dict[str, Any] | None:
    if isinstance(output, dict):
        try:
            output = ExtractedRecipeOutput.model_validate(output)
        except ValueError:
            _reject_recipe("invalid structured extractor output")
            return None

    if not isinstance(output, ExtractedRecipeOutput):
        _reject_recipe("invalid extractor output type")
        return None

    return output.model_dump(exclude_none=True, by_alias=True)


def _reject_recipe(reason: str) -> None:
    logger.info("Skipping recipe extraction: %s", reason)


_RESERVED_TOOL_PREFIXES = ("r_", "api_", "rest_", "graphql_")
_MAX_EXTRACTOR_TURNS = 2
_LOWER_EQUALS_VAR_RE = re.compile(
    r"lower\((?P<field>[^)]+)\)\s*=\s*(?:lower\()?['\"]\{\{(?P<var>[A-Za-z_][A-Za-z0-9_]*)\}\}['\"]\)?",
    re.IGNORECASE,
)


async def extract_recipe(
    *,
    api_type: str,
    question: str,
    steps: list[dict[str, Any]],
    result: Any | None = None,
    existing_recipes: list[dict[str, Any]] | None = None,
    validation_feedback: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Extract parameterized recipe from execution trace. Returns recipe or None."""
    from ..agent.model import get_run_config, model

    agent = Agent(
        name="recipe-extractor",
        model=model,
        instructions=build_extractor_instructions(api_type),
        tools=[],
        output_type=AgentOutputSchema(ExtractedRecipeOutput, strict_json_schema=False),
    )

    payload = {
        "api_type": api_type,
        "question": question,
        "steps": steps,
        "result": result,
        "existing_recipes": existing_recipes or [],
    }
    if validation_feedback:
        payload["validation_feedback"] = validation_feedback

    try:
        run_result = await Runner.run(
            agent,
            json.dumps(payload, indent=2),
            max_turns=_MAX_EXTRACTOR_TURNS,
            run_config=get_run_config(),
        )
    except ModelBehaviorError:
        _reject_recipe("invalid extractor output")
        return None
    if not run_result.final_output:
        _reject_recipe("empty extractor output")
        return None

    recipe = _structured_recipe(run_result.final_output)
    if not recipe:
        return None

    tool_name = get_recipe_tool_name(recipe)
    if not tool_name:
        _reject_recipe("missing tool_name")
        return None
    if not re.match(r"^[a-z][a-z0-9_]{0,39}$", tool_name):
        _reject_recipe("invalid tool_name")
        return None
    if tool_name.startswith(_RESERVED_TOOL_PREFIXES):
        _reject_recipe("invalid tool_name prefix")
        return None

    description = " ".join(get_recipe_description(recipe).split())
    if len(description) < 40 or len(description) > 1000:
        _reject_recipe("invalid description length")
        return None
    if "recipe name" in description.lower():
        _reject_recipe("invalid description text")
        return None
    recipe["public_contract"]["description"] = description
    _repair_alias_sql_filters(recipe)

    if err := validate_recipe_contract(recipe, api_type):
        _reject_recipe(err)
        return None

    return recipe


def _repair_alias_sql_filters(recipe: dict[str, Any]) -> None:
    tool_args = get_recipe_tool_args(recipe)
    fixture_args = get_validation_tool_args(recipe)

    for step in get_recipe_steps(recipe):
        if not isinstance(step, dict) or step.get("kind") != "sql":
            continue
        query_template = step.get("query_template")
        if not isinstance(query_template, str):
            continue
        step_input = step.get("input")
        with_vars = step_input.get("with") if isinstance(step_input, dict) else None
        if not isinstance(with_vars, dict):
            continue

        step["query_template"] = _LOWER_EQUALS_VAR_RE.sub(
            lambda match: _repaired_sql_match(match, with_vars, tool_args, fixture_args),
            query_template,
        )


def _repaired_sql_match(
    match: re.Match[str],
    with_vars: dict[str, Any],
    tool_args: dict[str, Any],
    fixture_args: dict[str, Any],
) -> str:
    var_name = match.group("var")
    source = with_vars.get(var_name)
    if not isinstance(source, dict):
        return match.group(0)

    arg_name = source.get("value")
    arg_spec = tool_args.get(arg_name)
    if not isinstance(arg_name, str) or not isinstance(arg_spec, dict):
        return match.group(0)
    if arg_spec.get("type") != "str":
        return match.group(0)

    transform = source.get("transform")
    if transform is None and not _looks_like_alias(fixture_args.get(arg_name)):
        return match.group(0)
    if transform not in (None, "contains_pattern"):
        return match.group(0)

    source["transform"] = "contains_pattern"
    return f"lower({match.group('field')}) LIKE lower('{{{{{var_name}}}}}')"


def _looks_like_alias(value: Any) -> bool:
    return isinstance(value, str) and any(
        not char.isalnum() and not char.isspace() for char in value
    )
