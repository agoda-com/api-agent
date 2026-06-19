"""Recipe template rendering helpers."""

from __future__ import annotations

import re
from typing import Any

_PLACEHOLDER_RE = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")


def render_text_template(template: str, params: dict[str, Any]) -> str:
    """Render {{param}} placeholders using raw string insertion."""

    def _as_text(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if v is None:
            return "null"
        return str(v)

    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in params:
            raise KeyError(f"missing param: {name}")
        return _as_text(params[name])

    return _PLACEHOLDER_RE.sub(repl, template)


def render_param_refs(obj: Any, params: dict[str, Any]) -> Any:
    """Recursively replace {'$var': 'x'} nodes with params['x']."""
    if isinstance(obj, dict):
        if set(obj.keys()) == {"$var"} and isinstance(obj.get("$var"), str):
            pname = obj["$var"]
            if pname not in params:
                raise KeyError(f"missing param: {pname}")
            return params[pname]
        return {k: render_param_refs(v, params) for k, v in obj.items()}
    if isinstance(obj, list):
        return [render_param_refs(v, params) for v in obj]
    return obj
