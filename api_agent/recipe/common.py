"""Shared utilities for recipe tools in GraphQL and REST agents."""

import json
import logging
import re
from contextvars import ContextVar
from typing import Any

from agents import FunctionToolResult, RunContextWrapper
from agents.agent import ToolsToFinalOutputResult
from pydantic import BaseModel, ConfigDict, Field, create_model

from ..config import settings
from ..executor import execute_sql, truncate_for_context
from .extractor import extract_recipe
from .store import RECIPE_STORE, render_text_template, sha256_hex

logger = logging.getLogger(__name__)

# Mapping from recipe param type names to JSON Schema type names
_JSON_TYPE_NAMES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}

# Track recipe changes per-request for tool list changed notifications
_recipes_changed: ContextVar[list[str]] = ContextVar("recipes_changed")


def reset_recipe_change_flag() -> None:
    """Reset recipe change tracking for the current request."""
    _recipes_changed.set([])


def mark_recipe_changed(recipe_id: str) -> None:
    """Record that a recipe was created during the current request."""
    try:
        _recipes_changed.get().append(recipe_id)
    except LookupError:
        _recipes_changed.set([recipe_id])


def consume_recipe_changes() -> list[str]:
    """Consume and clear recipe change tracking."""
    try:
        changes = list(_recipes_changed.get())
    except LookupError:
        return []
    _recipes_changed.set([])
    return changes


def _normalize_ws_value(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return value


def _recipes_equivalent(existing: dict[str, Any], candidate: dict[str, Any], api_type: str) -> bool:
    """Check if two recipes are equivalent for the same API type."""
    if existing.get("params", {}) != candidate.get("params", {}):
        return False

    e_steps = existing.get("steps", [])
    c_steps = candidate.get("steps", [])
    e_sql = existing.get("sql_steps", [])
    c_sql = candidate.get("sql_steps", [])
    if not (isinstance(e_steps, list) and isinstance(c_steps, list)):
        return False
    if not (isinstance(e_sql, list) and isinstance(c_sql, list)):
        return False
    if len(e_steps) != len(c_steps) or len(e_sql) != len(c_sql):
        return False

    for e_step, c_step in zip(e_steps, c_steps):
        if not (isinstance(e_step, dict) and isinstance(c_step, dict)):
            return False
        if e_step.get("kind") != c_step.get("kind"):
            return False
        if e_step.get("name") != c_step.get("name"):
            return False
        if api_type == "graphql":
            if _normalize_ws_value(e_step.get("query_template")) != _normalize_ws_value(
                c_step.get("query_template")
            ):
                return False
        else:
            for key in ("method", "path", "path_params", "query_params", "body"):
                if e_step.get(key) != c_step.get(key):
                    return False

    for e_sql_step, c_sql_step in zip(e_sql, c_sql):
        if _normalize_ws_value(e_sql_step) != _normalize_ws_value(c_sql_step):
            return False

    return True


async def maybe_extract_and_save_recipe(
    api_type: str,
    api_id: str,
    question: str,
    steps: list,
    sql_steps: list[str],
    raw_schema: str,
    skip_condition: bool = False,
) -> None:
    """Extract and save recipe if conditions met.

    Args:
        api_type: "graphql" or "rest"
        api_id: API identifier for recipe storage
        question: Original user question
        steps: API call steps from agent execution
        sql_steps: SQL steps from agent execution
        raw_schema: Raw schema string for hash
        skip_condition: If True, skip extraction (e.g., polling used)
    """
    if not settings.ENABLE_RECIPES:
        return
    if skip_condition:
        logger.info("Skipping recipe extraction (skip condition)")
        return
    if not (steps and raw_schema):
        return

    try:
        schema_hash = sha256_hex(raw_schema)
        existing_recipes = RECIPE_STORE.list_recipes(api_id=api_id, schema_hash=schema_hash)
        recipe = await extract_recipe(
            api_type=api_type,
            question=question,
            steps=steps,
            sql_steps=sql_steps,
            existing_recipes=existing_recipes,
        )
        if recipe:
            # Skip if recipe already exists (same steps/sql/params)
            for existing in existing_recipes:
                if _recipes_equivalent(existing, recipe, api_type):
                    return

            # Ensure tool_name does not collide with existing recipes
            seen: set[str] = {r["tool_name"] for r in existing_recipes if r.get("tool_name")}
            recipe["tool_name"] = deduplicate_tool_name(
                recipe.get("tool_name", ""), seen_names=seen, max_len=40
            )
            tool_name = recipe.get("tool_name", "")
            recipe_id = RECIPE_STORE.save_recipe(
                api_id=api_id,
                schema_hash=schema_hash,
                question=question,
                recipe=recipe,
                tool_name=tool_name,
            )
            mark_recipe_changed(recipe_id)
    except Exception:
        logger.exception("Recipe extraction failed")


# Shared ContextVar for direct return signaling
_return_directly_flag: ContextVar[list[bool]] = ContextVar("return_directly_flag")


def _set_return_directly() -> None:
    """Signal that tool result should be returned directly (skip LLM)."""
    try:
        _return_directly_flag.get().append(True)
    except LookupError:
        pass


def _tools_to_final_output(
    context: RunContextWrapper[Any], tool_results: list[FunctionToolResult]
) -> ToolsToFinalOutputResult:
    """Check if any tool requested direct return (skip LLM processing)."""
    try:
        if _return_directly_flag.get():
            return ToolsToFinalOutputResult(is_final_output=True, final_output="__DIRECT_RETURN__")
    except LookupError:
        pass
    return ToolsToFinalOutputResult(is_final_output=False, final_output=None)


def build_recipe_docstring(
    question: str,
    steps: list,
    sql_steps: list,
    api_type: str = "rest",
    params_spec: dict[str, Any] | None = None,
) -> str:
    """Build docstring for recipe tool."""
    parts = []
    if steps:
        count = len(steps)
        if api_type == "graphql":
            parts.append(f"{count} GraphQL quer{'ies' if count > 1 else 'y'}")
        else:
            parts.append(f"{count} API call{'s' if count > 1 else ''}")
    if sql_steps:
        parts.append(f"{len(sql_steps)} SQL step{'s' if len(sql_steps) > 1 else ''}")
    steps_summary = " + ".join(parts) if parts else "No steps"

    params_section = ""
    if params_spec:
        param_lines = []
        for pname, spec in params_spec.items():
            ptype = _JSON_TYPE_NAMES.get(
                spec.get("type", "str") if isinstance(spec, dict) else "str", "string"
            )
            example = spec.get("default") if isinstance(spec, dict) else None
            hint = f" (e.g. {example})" if example is not None else ""
            param_lines.append(f"  {pname}: {ptype} REQUIRED{hint}")
        params_section = "\nRequired params:\n" + "\n".join(param_lines)

    return f"Execute recipe: {question}\nRecipe performs: {steps_summary}{params_section}"


def create_params_model(pspec: dict[str, Any], tname: str):
    """Create Pydantic model for recipe params with strict validation.

    All fields are required (no defaults). Stored defaults are example values
    from the original execution and are shown as description hints only.
    """

    class StrictBase(BaseModel):
        model_config = ConfigDict(extra="forbid")

    type_map = {"str": str, "int": int, "float": float, "bool": bool}
    field_defs = {}
    for pname, pinfo in pspec.items():
        py_type = type_map.get(pinfo.get("type", "str"), str)
        example = pinfo.get("default")
        desc = f"Required. e.g. {example}" if example is not None else "Required"
        field_defs[pname] = (py_type, Field(..., description=desc))

    return create_model(f"{tname}_Params", __base__=StrictBase, **field_defs)


def deduplicate_tool_name(base_name: str, seen_names: set[str], max_len: int = 40) -> str:
    """Ensure unique tool name within length limit."""
    base = re.sub(r"[^a-z0-9_]", "", base_name)[:max_len]
    if not base or not re.match(r"^[a-z][a-z0-9_]*$", base):
        base = "recipe"

    if base not in seen_names:
        seen_names.add(base)
        return base

    counter = 2
    while True:
        suffix = f"_{counter}"
        trimmed = base[: max_len - len(suffix)]
        candidate = f"{trimmed}{suffix}"
        if candidate not in seen_names:
            seen_names.add(candidate)
            return candidate
        counter += 1


def _execute_sql_steps(
    sql_steps: list[str],
    params: dict[str, Any],
    results: dict[str, Any],
    last_result_var: ContextVar[list[Any]],
) -> tuple[bool, list[str], str]:
    """Execute SQL steps. Returns (success, executed_sql, error_json)."""
    executed_sql: list[str] = []

    for sql_tmpl in sql_steps:
        if not isinstance(sql_tmpl, str):
            return (
                False,
                executed_sql,
                json.dumps({"success": False, "error": "invalid sql_steps"}, indent=2),
            )

        sql = render_text_template(sql_tmpl, params)
        res = execute_sql(results, sql)
        executed_sql.append(sql)

        if not res.get("success"):
            return False, executed_sql, json.dumps(res, indent=2)

        try:
            last_result_var.get()[0] = res.get("result", [])
        except LookupError:
            pass

    return True, executed_sql, ""


def format_recipe_response(
    last_result_var: ContextVar[list[Any]],
    executed_items: list[Any],
    executed_sql: list[str],
    item_key: str,
) -> str:
    """Format recipe JSON response with truncation."""
    try:
        last_rows = last_result_var.get()[0]
    except LookupError:
        last_rows = None

    base = {"success": True, item_key: executed_items, "executed_sql": executed_sql}
    if isinstance(last_rows, list):
        base.update(truncate_for_context(last_rows, "sql_result"))
    return json.dumps(base, indent=2)


def build_partial_result(
    last_data: Any,
    api_calls: list[Any],
    turn_info: str,
    call_key: str,
) -> dict[str, Any]:
    """Build partial result dict for MaxTurnsExceeded."""
    if last_data:
        return {
            "ok": True,
            "data": f"[Partial - {turn_info}] Max turns exceeded but data retrieved.",
            "result": last_data,
            call_key: api_calls,
            "error": None,
        }
    return {
        "ok": False,
        "data": None,
        "result": None,
        call_key: api_calls,
        "error": f"Max turns exceeded ({turn_info}), no data retrieved",
    }


def build_api_id(ctx, api_type: str, base_url: str = "") -> str:
    """Build api_id string for recipe matching."""
    if api_type == "graphql":
        return f"graphql:{ctx.target_url}"
    return f"rest:{ctx.target_url}|{base_url}"


def _get_results_context(query_results_var: ContextVar[dict[str, Any]]) -> dict[str, Any]:
    """Get or create results dict from ContextVar."""
    try:
        return query_results_var.get()
    except LookupError:
        results: dict[str, Any] = {}
        query_results_var.set(results)
        return results


def _score_hint(score: float) -> str:
    """Get human-readable hint for recipe match score."""
    if score >= 0.8:
        return "STRONG MATCH - highly recommended"
    if score >= 0.6:
        return "Good match - verify params"
    return "Possible match - check alignment"


def _steps_summary(steps: list, sql_steps: list) -> str:
    """Build step summary string."""
    parts = []
    if steps:
        parts.append(f"{len(steps)} API call{'s' if len(steps) > 1 else ''}")
    if sql_steps:
        parts.append(f"{len(sql_steps)} SQL step{'s' if len(sql_steps) > 1 else ''}")
    return " + ".join(parts) if parts else "no steps"


def search_recipes(
    api_id: str,
    raw_schema: str,
    question: str,
    k: int = 3,
) -> tuple[list[dict[str, Any]], str]:
    """Search for matching recipes and build context string.

    Args:
        api_id: API identifier (e.g., "graphql:url" or "rest:url|base")
        raw_schema: Raw schema JSON string
        question: User's question
        k: Max suggestions to return

    Returns:
        (suggestions_list, recipe_context_string)
    """
    if not raw_schema:
        return [], ""

    schema_hash = sha256_hex(raw_schema)
    suggestions = RECIPE_STORE.suggest_recipes(
        api_id=api_id,
        schema_hash=schema_hash,
        question=question,
        k=k,
    )
    if not suggestions:
        return [], ""

    # Enrich with recipe params for display
    for s in suggestions:
        recipe = RECIPE_STORE.get_recipe(s["recipe_id"])
        if recipe:
            s["params"] = recipe.get("params", {})

    return suggestions, build_recipe_context(suggestions)


def build_recipe_context(suggestions: list[dict[str, Any]]) -> str:
    """Build recipe context for system prompt."""
    if not suggestions:
        return ""

    lines = ["\n<recipes>", "Available recipe tools (sorted by relevance):"]

    for idx, s in enumerate(suggestions, 1):
        recipe = RECIPE_STORE.get_recipe(s["recipe_id"])
        if not recipe:
            continue

        params_spec = s.get("params", {})
        param_list = []
        for k, spec in params_spec.items():
            if isinstance(spec, dict):
                typ = spec.get("type", "str")
                default = spec.get("default")
                param_list.append(
                    f"{k}: {typ} = {default}" if default is not None else f"{k}: {typ}"
                )
            else:
                param_list.append(f"{k}: str")

        tool_name = s.get("tool_name") or _sanitize_for_tool_name(s["question"])
        score = s["score"]

        lines.append(f"\n{idx}. {tool_name}({', '.join(param_list)})")
        lines.append(f'   Question: "{s["question"]}"')
        lines.append(f"   Score: {score:.2f} ({_score_hint(score)})")
        lines.append(
            f"   Steps: {_steps_summary(recipe.get('steps', []), recipe.get('sql_steps', []))}"
        )

    lines.append("</recipes>")
    return "\n".join(lines)


def error_json(msg: str) -> str:
    """Build JSON error response."""
    return json.dumps({"success": False, "error": msg}, indent=2)


def validate_and_prepare_recipe(
    recipe_id: str,
    params_json: str,
    raw_schema_var: ContextVar[str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    """Validate recipe and prepare params. Returns (recipe, params, error_json)."""
    try:
        raw_schema = raw_schema_var.get()
    except LookupError:
        raw_schema = ""
    if not raw_schema:
        return None, None, error_json("schema not loaded")

    recipe = RECIPE_STORE.get_recipe(recipe_id)
    if not recipe:
        return None, None, error_json(f"recipe not found: {recipe_id}")

    provided: dict[str, Any] = {}
    if params_json:
        try:
            provided = json.loads(params_json)
        except json.JSONDecodeError as e:
            return None, None, error_json(f"invalid params_json: {e.msg}")

    validated, err = validate_recipe_params(recipe.get("params", {}), provided)
    if err:
        return None, None, err
    return recipe, validated, ""


def validate_recipe_params(
    params_spec: dict[str, Any],
    provided: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Validate and merge recipe params. Returns (params, error_json)."""
    # Reject unknown params
    if params_spec:
        extra = set(provided.keys()) - set(params_spec.keys())
        if extra:
            return None, error_json(f"unexpected params: {', '.join(sorted(extra))}")

    # All declared params are required (defaults are example values, not fallbacks)
    for pname, spec in params_spec.items():
        if pname not in provided:
            return None, error_json(f"missing required param: {pname}")

    return dict(provided), ""


def _sanitize_for_tool_name(question: str) -> str:
    """Convert question to valid Python identifier (max 40 chars)."""
    name = re.sub(r"[^\w\s]", "", question.lower())
    name = re.sub(r"\s+", "_", name)
    name = name[:40].strip("_")
    if name and name[0].isdigit():
        name = "r_" + name
    return name


async def execute_recipe_steps(
    recipe: dict[str, Any],
    params: dict[str, Any],
    query_results_var: ContextVar[dict[str, Any]],
    last_result_var: ContextVar[list[Any]],
    api_step_executor,
    executed_items_list,
) -> tuple[bool, Any, list[str], str]:
    """Execute recipe steps (API + SQL). Returns (success, last_data, executed_sql, error_json)."""
    results = _get_results_context(query_results_var)

    for step_idx, step in enumerate(recipe.get("steps", [])):
        success, data, error, call_rec = await api_step_executor(step_idx, step, params, results)
        if not success:
            return False, None, [], error

        if call_rec:
            executed_items_list.append(call_rec)

        if data is not None:
            try:
                last_result_var.get()[0] = data
            except LookupError:
                pass

    success, executed_sql, error = _execute_sql_steps(
        recipe.get("sql_steps", []), params, results, last_result_var
    )
    if not success:
        return False, None, executed_sql, error

    try:
        return True, last_result_var.get()[0], executed_sql, ""
    except LookupError:
        return True, None, executed_sql, ""
