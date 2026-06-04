"""Recipe lookup and prompt context formatting."""

from __future__ import annotations

from typing import Any

from ..store import ASYNC_API_AGENT_STORE, sha256_hex
from .contracts import (
    get_recipe_steps,
    get_recipe_tool_args,
    get_recipe_tool_name,
    has_recipe_contract,
)
from .tooling import _sanitize_for_tool_name


def build_api_id(ctx, api_type: str, base_url: str = "") -> str:
    """Build api_id string for recipe matching."""
    if api_type == "graphql":
        return f"graphql:{ctx.target_url}"
    return f"rest:{ctx.target_url}|{base_url}"


def _score_hint(score: float) -> str:
    """Get human-readable hint for recipe match score."""
    if score >= 0.8:
        return "STRONG MATCH - highly recommended"
    if score >= 0.6:
        return "Good match - verify params"
    return "Possible match - check alignment"


def _steps_summary(steps: list) -> str:
    """Build step summary string."""
    parts = []
    api_count = 0
    sql_count = 0
    for step in steps:
        if isinstance(step, dict) and step.get("kind") == "sql":
            sql_count += 1
        else:
            api_count += 1

    if api_count:
        parts.append(f"{api_count} API call{'s' if api_count > 1 else ''}")
    if sql_count:
        parts.append(f"{sql_count} SQL step{'s' if sql_count > 1 else ''}")
    return " + ".join(parts) if parts else "no steps"


async def search_recipes(
    api_id: str,
    raw_schema: str,
    question: str,
    k: int = 3,
) -> tuple[list[dict[str, Any]], str]:
    """Search for matching recipes and build context string."""
    if not raw_schema:
        return [], ""

    schema_hash = sha256_hex(raw_schema)
    suggestions = await ASYNC_API_AGENT_STORE.suggest_recipes(
        api_id=api_id,
        schema_hash=schema_hash,
        question=question,
        k=k,
    )
    if not suggestions:
        return [], ""

    contract_suggestions: list[dict[str, Any]] = []
    for suggestion in suggestions:
        recipe = await ASYNC_API_AGENT_STORE.get_recipe(suggestion["recipe_id"])
        if not recipe or not has_recipe_contract(recipe):
            continue
        contract_suggestions.append(
            {
                **suggestion,
                "recipe": recipe,
                "params": get_recipe_tool_args(recipe),
                "tool_name": get_recipe_tool_name(recipe),
            }
        )

    return contract_suggestions, build_recipe_context(contract_suggestions)


def build_recipe_context(suggestions: list[dict[str, Any]]) -> str:
    """Build recipe context for system prompt."""
    if not suggestions:
        return ""

    lines = ["\n<recipes>", "Available recipe tools (sorted by relevance):"]

    for idx, suggestion in enumerate(suggestions, 1):
        recipe = suggestion.get("recipe")
        if not recipe:
            continue

        if not has_recipe_contract(recipe):
            continue

        params_spec = get_recipe_tool_args(recipe)
        param_list = []
        for key, spec in params_spec.items():
            if isinstance(spec, dict):
                typ = spec.get("type", "str")
                param_list.append(f"{key}: {typ}")
            else:
                param_list.append(f"{key}: str")

        tool_name = get_recipe_tool_name(recipe) or _sanitize_for_tool_name(suggestion["question"])
        score = suggestion["score"]

        lines.append(f"\n{idx}. {tool_name}({', '.join(param_list)})")
        lines.append(f'   Question: "{suggestion["question"]}"')
        lines.append(f"   Score: {score:.2f} ({_score_hint(score)})")
        lines.append(f"   Steps: {_steps_summary(get_recipe_steps(recipe))}")

    lines.append("</recipes>")
    return "\n".join(lines)
