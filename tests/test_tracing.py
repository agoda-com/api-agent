from api_agent import tracing


def test_agent_span_attributes_include_openinference_kind():
    assert tracing.agent_span_attributes("api_agent", "rest") == {
        "mcp_name": "api_agent",
        "agent_type": "rest",
        "openinference.span.kind": "AGENT",
    }


def test_span_trace_id_formats_recording_span():
    class FakeContext:
        trace_id = 0x1747C063EBC5125A0FD42E7F741F5BAB

    class FakeSpan:
        def get_span_context(self):
            return FakeContext()

    assert tracing.span_trace_id(FakeSpan()) == "1747c063ebc5125a0fd42e7f741f5bab"


def test_span_trace_id_skips_invalid_span():
    assert tracing.span_trace_id(None) is None
