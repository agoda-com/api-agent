"""Per-request recipe runtime state."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from agents import FunctionToolResult, RunContextWrapper
from agents.agent import ToolsToFinalOutputResult

_recipes_changed: ContextVar[list[str]] = ContextVar("recipes_changed")
_return_directly_flag: ContextVar[list[bool]] = ContextVar("return_directly_flag")
_recipe_tools_used: ContextVar[list[str]] = ContextVar("recipe_tools_used")


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


def reset_recipe_tool_usage() -> None:
    """Reset recipe tool usage tracking for the current agent run."""
    _recipe_tools_used.set([])


def mark_recipe_tool_used(recipe_id: str) -> None:
    """Record that the agent used an injected recipe tool."""
    try:
        _recipe_tools_used.get().append(recipe_id)
    except LookupError:
        _recipe_tools_used.set([recipe_id])


def recipe_tool_was_used() -> bool:
    """Return whether an injected recipe tool ran in this agent run."""
    try:
        return bool(_recipe_tools_used.get())
    except LookupError:
        return False


def _set_return_directly() -> None:
    """Signal that tool result should be returned directly."""
    try:
        _return_directly_flag.get().append(True)
    except LookupError:
        pass


def _tools_to_final_output(
    context: RunContextWrapper[Any], tool_results: list[FunctionToolResult]
) -> ToolsToFinalOutputResult:
    """Return tool output directly when a recipe requested it."""
    try:
        if _return_directly_flag.get():
            return ToolsToFinalOutputResult(is_final_output=True, final_output="__DIRECT_RETURN__")
    except LookupError:
        pass
    return ToolsToFinalOutputResult(is_final_output=False, final_output=None)
