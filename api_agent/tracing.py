"""OpenTelemetry tracing setup."""

import logging
from contextlib import contextmanager
from typing import Any, Generator

from openinference.instrumentation import using_metadata
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace

logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)


def span_trace_id(span: Any) -> str | None:
    if not span:
        return None
    trace_id = span.get_span_context().trace_id
    if trace_id == trace.INVALID_TRACE_ID:
        return None
    return trace.format_trace_id(trace_id)


def agent_span_attributes(mcp_name: str, agent_type: str) -> dict[str, str]:
    return {
        "mcp_name": mcp_name,
        "agent_type": agent_type,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
    }


@contextmanager
def trace_metadata(metadata: dict[str, Any]) -> Generator[None, None, None]:
    """Context manager for span metadata. No-op if tracing disabled."""
    with using_metadata(metadata):
        yield


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Generator[Any, None, None]:
    """Create a span with optional attributes. No-op if tracing disabled.

    Args:
        name: Span name
        attributes: Optional attributes to add to span

    Yields:
        Span object (or None if tracing disabled)
    """
    try:
        with tracer.start_as_current_span(name) as span:
            if attributes and span:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span
    except Exception:
        logger.warning("Failed to create span", exc_info=True)
        yield None
