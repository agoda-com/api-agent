"""Recipe MCP tool schema, naming, and descriptions."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

_LEADING_CONTEXT_RE = re.compile(r"^\[[^\]]+\]\s*")


def build_recipe_docstring(
    question: str,
    steps: list,
    api_type: str = "rest",
    params_spec: dict[str, Any] | None = None,
    description: str | None = None,
) -> str:
    """Build docstring for recipe tool."""
    return _normalize_description(description or "")


def _normalize_description(description: str) -> str:
    """Normalize generated tool descriptions for MCP clients."""
    return _LEADING_CONTEXT_RE.sub("", " ".join(description.split()))


def create_params_model(pspec: dict[str, Any], tname: str) -> type[BaseModel]:
    """Create Pydantic model for recipe params with strict validation."""

    class StrictBase(BaseModel):
        model_config = ConfigDict(extra="forbid")

    type_map = {"str": str, "int": int, "float": float, "bool": bool}
    field_defs: dict[str, tuple[Any, Any]] = {}
    for pname, pinfo in pspec.items():
        py_type = type_map.get(pinfo.get("type", "str"), str)
        desc = pinfo.get("description") or "Required"
        field_defs[pname] = (py_type, Field(..., description=desc))

    return create_model(f"{tname}_Params", __base__=StrictBase, **field_defs)  # ty: ignore[no-matching-overload]


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


def _sanitize_for_tool_name(question: str) -> str:
    """Convert question to valid Python identifier (max 40 chars)."""
    name = re.sub(r"[^\w\s]", "", question.lower())
    name = re.sub(r"\s+", "_", name)
    name = name[:40].strip("_")
    if name and name[0].isdigit():
        name = "r_" + name
    return name
