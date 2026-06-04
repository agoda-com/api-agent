"""Query response envelope for MCP output formatting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QueryResponse:
    """Normalized agent response before transport-specific formatting."""

    ok: bool
    data: Any = None
    result: Any = None
    calls: list[Any] = field(default_factory=list)
    calls_key: str = "calls"
    error: Any = None
    trace_id: str | None = None

    @classmethod
    def from_agent_result(cls, result: dict[str, Any], calls_key: str) -> "QueryResponse":
        """Build from the existing agent result dict contract."""
        calls = result.get(calls_key, [])
        trace_id = result.get("trace_id")
        return cls(
            ok=bool(result.get("ok", False)),
            data=result.get("data"),
            result=result.get("result"),
            calls=list(calls) if isinstance(calls, list) else [],
            calls_key=calls_key,
            error=result.get("error"),
            trace_id=trace_id if isinstance(trace_id, str) else None,
        )

    @property
    def should_return_csv(self) -> bool:
        """Current direct-return behavior: result present with no text answer."""
        return self.result is not None and self.data is None

    def to_mcp_payload(
        self,
        *,
        include_result: bool = False,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        """Convert to the existing MCP response dict shape."""
        payload = {
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
        }
        if include_result and self.result is not None:
            payload["result"] = self.result
        if include_debug:
            debug: dict[str, Any] = {self.calls_key: self.calls}
            if self.trace_id:
                debug["trace_id"] = self.trace_id
            payload["debug"] = debug
        return payload
