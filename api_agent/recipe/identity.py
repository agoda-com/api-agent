"""Recipe identity and equivalence helpers."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .contracts import get_recipe_tool_args


def sha256_hex(text: str) -> str:
    """Hash text, normalizing JSON when possible for stable schema hashes."""
    normalized = text
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if parsed is not None:
        normalized = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_ws(text: str) -> str:
    """Whitespace-normalize for template equivalence checks."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def _canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonical_value(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_canonical_value(v) for v in value]
    if isinstance(value, str):
        return normalize_ws(value)
    return value


def _stable_tool_args(recipe: dict[str, Any]) -> dict[str, Any]:
    stable: dict[str, Any] = {}
    for name, spec in get_recipe_tool_args(recipe).items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            continue
        stable[name] = {"type": spec.get("type", "str")}
    return stable


def recipe_behavior_payload(recipe: dict[str, Any]) -> dict[str, Any]:
    """Stable recipe behavior identity, excluding generated copy/names."""
    return {
        "tool_args": _canonical_value(_stable_tool_args(recipe)),
        "execution_plan": _canonical_value(recipe.get("execution_plan", {})),
    }


def recipe_fingerprint(
    *,
    api_id: str,
    schema_hash: str,
    recipe: dict[str, Any],
) -> str:
    """Canonical recipe fingerprint for idempotent saves."""
    payload = {
        "api_type": api_id.split(":", 1)[0],
        "api_id": api_id,
        "schema_hash": schema_hash,
        "behavior": recipe_behavior_payload(recipe),
    }
    return sha256_hex(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))
