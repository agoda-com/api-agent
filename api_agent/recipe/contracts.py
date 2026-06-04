"""Recipe public contract and private execution helpers."""

from __future__ import annotations

import re
from typing import Any

from .templates import _PLACEHOLDER_RE

TOOL_ARG_TYPES = {"str", "int", "float", "bool"}
SUPPORTED_TRANSFORMS = {"contains_pattern"}
SUPPORTED_INPUT_MODES = {"single", "map", "batch"}


def has_recipe_contract(recipe: dict[str, Any]) -> bool:
    return isinstance(recipe.get("public_contract"), dict) and isinstance(
        recipe.get("execution_plan"), dict
    )


def get_public_contract(recipe: dict[str, Any]) -> dict[str, Any]:
    contract = recipe.get("public_contract")
    return contract if isinstance(contract, dict) else {}


def get_execution_plan(recipe: dict[str, Any]) -> dict[str, Any]:
    plan = recipe.get("execution_plan")
    return plan if isinstance(plan, dict) else {}


def get_validation_fixture(recipe: dict[str, Any]) -> dict[str, Any]:
    fixture = recipe.get("validation_fixture")
    return fixture if isinstance(fixture, dict) else {}


def get_recipe_tool_name(recipe: dict[str, Any]) -> str:
    name = get_public_contract(recipe).get("tool_name")
    return name if isinstance(name, str) else ""


def get_recipe_description(recipe: dict[str, Any]) -> str:
    description = get_public_contract(recipe).get("description")
    return description if isinstance(description, str) else ""


def get_recipe_tool_args(recipe: dict[str, Any]) -> dict[str, Any]:
    tool_args = get_public_contract(recipe).get("tool_args")
    return tool_args if isinstance(tool_args, dict) else {}


def get_recipe_steps(recipe: dict[str, Any]) -> list[Any]:
    steps = get_execution_plan(recipe).get("steps")
    return steps if isinstance(steps, list) else []


def get_validation_tool_args(recipe: dict[str, Any]) -> dict[str, Any]:
    tool_args = get_validation_fixture(recipe).get("tool_args")
    return tool_args if isinstance(tool_args, dict) else {}


def get_step_id(step: dict[str, Any]) -> str:
    step_id = step.get("id")
    return step_id if isinstance(step_id, str) else ""


def get_step_input(step: dict[str, Any]) -> dict[str, Any]:
    step_input = step.get("input")
    return step_input if isinstance(step_input, dict) else {}


def get_step_call(step: dict[str, Any]) -> dict[str, Any]:
    call = step.get("call")
    return call if isinstance(call, dict) else {}


def get_step_output(step: dict[str, Any]) -> dict[str, Any]:
    output = step.get("output")
    return output if isinstance(output, dict) else {}


def get_step_output_name(step: dict[str, Any]) -> str:
    name = get_step_output(step).get("name")
    return name if isinstance(name, str) else ""


def find_template_vars(steps: list[Any]) -> set[str]:
    found: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        call = get_step_call(step)
        tmpl = (
            step.get("query_template") if step.get("kind") == "sql" else call.get("query_template")
        )
        if isinstance(tmpl, str):
            found.update(_PLACEHOLDER_RE.findall(tmpl))
        for key in ("path_params", "query_params", "body"):
            _find_param_refs(call.get(key), found)
    return found


def _find_param_refs(obj: Any, found: set[str]) -> None:
    if isinstance(obj, dict):
        if set(obj.keys()) == {"$var"} and isinstance(obj.get("$var"), str):
            found.add(obj["$var"])
            return
        for value in obj.values():
            _find_param_refs(value, found)
    elif isinstance(obj, list):
        for value in obj:
            _find_param_refs(value, found)


_ARG_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_tool_args(tool_args: dict[str, Any]) -> str:
    if not isinstance(tool_args, dict):
        return "invalid tool_args"
    for name, spec in tool_args.items():
        if not isinstance(name, str) or not _ARG_NAME_RE.match(name):
            return "invalid tool arg name"
        if not isinstance(spec, dict):
            return "invalid tool arg spec"
        if spec.get("type") not in TOOL_ARG_TYPES:
            return "invalid tool arg type"
        description = spec.get("description")
        if description is not None and not isinstance(description, str):
            return "invalid tool arg description"
    return ""


def validate_recipe_contract(recipe: dict[str, Any], api_type: str) -> str:
    if not has_recipe_contract(recipe):
        return "missing public contract or execution plan"

    tool_args = get_recipe_tool_args(recipe)
    if err := validate_tool_args(tool_args):
        return err

    fixture_args = get_validation_tool_args(recipe)
    if set(fixture_args.keys()) != set(tool_args.keys()):
        return "validation fixture args mismatch"

    steps = get_recipe_steps(recipe)
    if not steps:
        return "missing execution steps"

    step_ids: set[str] = set()
    available_results: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            return "invalid recipe step"

        kind = step.get("kind")
        if kind not in {"graphql", "rest", "sql"}:
            return "unsupported recipe step"
        if api_type == "rest" and kind == "graphql":
            return "recipe step api mismatch"
        if api_type == "graphql" and kind == "rest":
            return "recipe step api mismatch"

        step_id = get_step_id(step)
        if not step_id or not _ARG_NAME_RE.match(step_id):
            return "invalid step id"
        if step_id in step_ids:
            return "duplicate step id"
        step_ids.add(step_id)

        output_name = get_step_output_name(step)
        if not output_name or not _ARG_NAME_RE.match(output_name):
            return "invalid step output"
        if output_name in available_results:
            return "duplicate step output"

        if err := _validate_step_operation(step):
            return err
        if err := _validate_step_input(step, tool_args, available_results):
            return err

        available_results.add(output_name)

    return ""


def _validate_step_operation(step: dict[str, Any]) -> str:
    kind = step.get("kind")
    if kind == "sql":
        if not isinstance(step.get("query_template"), str):
            return "missing sql query template"
        return ""

    call = get_step_call(step)
    if kind == "graphql":
        if not isinstance(call.get("query_template"), str):
            return "missing graphql query template"
        return ""

    if kind == "rest":
        method = str(call.get("method") or "GET").upper()
        if method != "GET":
            return "unsafe REST recipe"
        if not isinstance(call.get("path"), str):
            return "missing rest path"
        return ""

    return "unsupported recipe step"


def _validate_step_input(
    step: dict[str, Any],
    tool_args: dict[str, Any],
    available_results: set[str],
) -> str:
    step_input = get_step_input(step)
    if not step_input:
        return "missing step input"

    mode = step_input.get("mode")
    if mode not in SUPPORTED_INPUT_MODES:
        return "unsupported step input mode"
    if step.get("kind") == "sql" and mode != "single":
        return "sql steps must be single"

    with_vars = step_input.get("with") or {}
    if not isinstance(with_vars, dict):
        return "invalid step input"
    for name, source in with_vars.items():
        if not isinstance(name, str) or not _ARG_NAME_RE.match(name):
            return "invalid step input"
        if err := _validate_input_source(source, tool_args):
            return err

    bind = step_input.get("bind") or {}
    if not isinstance(bind, dict):
        return "invalid step binding"
    for name, field in bind.items():
        if not isinstance(name, str) or not _ARG_NAME_RE.match(name):
            return "invalid step binding"
        if not isinstance(field, str) or not field:
            return "invalid step binding"

    source_name = step_input.get("from")
    if mode == "single":
        if source_name is not None or bind:
            return "single step cannot bind rows"
    else:
        if not isinstance(source_name, str) or source_name not in available_results:
            return "step input must reference prior output"
        if not bind:
            return "mapped step must bind fields"

    used_vars = find_template_vars([step])
    provided_vars = set(with_vars.keys()) | set(bind.keys())
    if used_vars != provided_vars:
        return "template vars and step input mismatch"

    attach_binding = get_step_output(step).get("attach_binding") or []
    if not isinstance(attach_binding, list):
        return "invalid output binding"
    if attach_binding and mode != "map":
        return "invalid output binding"
    for name in attach_binding:
        if not isinstance(name, str) or name not in bind:
            return "invalid output binding"

    return ""


def _validate_input_source(source: Any, tool_args: dict[str, Any]) -> str:
    if not isinstance(source, dict):
        return "invalid value source"

    arg = source.get("value")
    if not isinstance(arg, str) or arg not in tool_args:
        return "value source references missing arg"

    transform = source.get("transform")
    if transform is not None and transform not in SUPPORTED_TRANSFORMS:
        return "unsupported value transform"

    allowed_keys = {"value", "transform"}
    if set(source.keys()) - allowed_keys:
        return "invalid value source"
    return ""


def resolve_recipe_values(
    recipe: dict[str, Any],
    public_args: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    tool_args = get_recipe_tool_args(recipe)
    extra = set(public_args.keys()) - set(tool_args.keys())
    if extra:
        return None, f"unexpected params: {', '.join(sorted(extra))}"
    missing = set(tool_args.keys()) - set(public_args.keys())
    if missing:
        return None, f"missing required param: {sorted(missing)[0]}"

    return dict(public_args), ""


def resolve_step_input_values(
    step: dict[str, Any],
    public_args: dict[str, Any],
    results: dict[str, Any],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]] | None, str]:
    step_input = get_step_input(step)
    mode = step_input.get("mode")
    if mode not in SUPPORTED_INPUT_MODES:
        return None, "unsupported step input mode"

    fixed_values, err = _resolve_with_values(step_input.get("with") or {}, public_args)
    if err:
        return None, err
    assert fixed_values is not None

    if mode == "single":
        return [(fixed_values, {})], ""

    source_name = step_input.get("from")
    if not isinstance(source_name, str):
        return None, "step input must reference prior output"
    rows = results.get(source_name)
    if not isinstance(rows, list):
        return None, f"missing step result: {source_name}"

    bind = step_input.get("bind") or {}
    if not isinstance(bind, dict):
        return None, "invalid step binding"

    if mode == "batch":
        batch_values: dict[str, list[Any]] = {}
        for var_name, field in bind.items():
            values, field_error = _collect_field_values(rows, str(field), source_name)
            if field_error:
                return None, field_error
            batch_values[str(var_name)] = values
        return [({**fixed_values, **batch_values}, batch_values)] if rows else [], ""

    input_sets: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        bound_values: dict[str, Any] = {}
        for var_name, field in bind.items():
            found, value = _get_field(row, str(field))
            if not found or value is None:
                return None, f"missing step field values: {source_name}.{field}"
            bound_values[str(var_name)] = value
        input_sets.append(({**fixed_values, **bound_values}, bound_values))
    return input_sets, ""


def _resolve_with_values(
    with_vars: dict[str, Any],
    public_args: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    values: dict[str, Any] = {}
    for name, source in with_vars.items():
        if not isinstance(source, dict):
            return None, f"invalid value source: {name}"
        arg = source.get("value")
        if not isinstance(arg, str) or arg not in public_args:
            return None, f"invalid value source: {name}"
        value = public_args[arg]
        transform = source.get("transform")
        if transform == "contains_pattern":
            value = _contains_pattern(value)
        elif transform is not None:
            return None, f"invalid value source: {name}"
        values[name] = value
    return values, ""


def _contains_pattern(value: Any) -> str:
    parts = [part for part in re.split(r"[^0-9A-Za-z]+", str(value)) if part]
    return f"%{'%'.join(parts)}%" if parts else "%"


def _collect_field_values(
    rows: list[Any],
    field: str,
    source_name: str,
) -> tuple[list[Any], str]:
    values: list[Any] = []
    for row in rows:
        found, value = _get_field(row, field)
        if not found or value is None:
            return [], f"missing step field values: {source_name}.{field}"
        values.append(value)
    return values, ""


def normalize_result(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [_canonical_value(row) for row in value]
    return _canonical_value(value)


def results_equivalent(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return normalize_result(left) == normalize_result(right)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonical_value(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    return value


def _get_field(row: Any, field: str) -> tuple[bool, Any]:
    value = row
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return False, None
        value = value[part]
    return True, value
