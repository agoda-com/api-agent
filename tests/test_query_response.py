"""Tests for CSV response formatting."""

from types import SimpleNamespace

from api_agent.query_response import QueryResponse
from api_agent.tools.query import _should_include_result, _should_return_csv
from api_agent.utils.csv import to_csv


class TestQueryResponse:
    """Tests for normalized query response envelope."""

    def test_wrapped_payload_hides_result_and_calls_by_default(self):
        response = QueryResponse.from_agent_result(
            {
                "ok": True,
                "data": "done",
                "result": [{"id": 1}],
                "queries": ["{ users { id } }"],
                "error": None,
            },
            "queries",
        )

        assert response.to_mcp_payload() == {
            "ok": True,
            "data": "done",
            "error": None,
        }

    def test_result_can_be_included_in_payload(self):
        response = QueryResponse.from_agent_result(
            {"ok": True, "data": "done", "result": [{"id": 1}], "api_calls": [], "error": None},
            "api_calls",
        )

        assert response.to_mcp_payload(include_result=True)["result"] == [{"id": 1}]

    def test_missing_result_is_not_forced_into_payload(self):
        response = QueryResponse.from_agent_result(
            {"ok": True, "data": "done", "api_calls": [], "error": None},
            "api_calls",
        )

        assert "result" not in response.to_mcp_payload(include_result=True)

    def test_direct_return_csv_marker(self):
        response = QueryResponse.from_agent_result(
            {"ok": True, "data": None, "result": [{"id": 1}], "queries": []},
            "queries",
        )

        assert response.should_return_csv is True

    def test_debug_payload_includes_calls_and_trace_id(self):
        response = QueryResponse.from_agent_result(
            {
                "ok": True,
                "data": "done",
                "api_calls": [{"method": "GET"}],
                "trace_id": "abc123",
                "error": None,
            },
            "api_calls",
        )

        assert response.to_mcp_payload(include_debug=True)["debug"] == {
            "api_calls": [{"method": "GET"}],
            "trace_id": "abc123",
        }

    def test_debug_direct_return_includes_result(self):
        response = QueryResponse.from_agent_result(
            {"ok": True, "data": None, "result": [{"id": 1}], "api_calls": []},
            "api_calls",
        )
        req_ctx = SimpleNamespace(include_result=False, debug=True)

        assert _should_include_result(response, req_ctx) is True

    def test_query_return_directly_returns_csv_when_rows_exist(self):
        response = QueryResponse.from_agent_result(
            {"ok": True, "data": "summary", "result": [{"id": 1}], "api_calls": []},
            "api_calls",
        )
        req_ctx = SimpleNamespace(debug=False)

        assert _should_return_csv(response, req_ctx, return_directly=True) is True

    def test_query_return_directly_still_wraps_without_rows(self):
        response = QueryResponse.from_agent_result(
            {"ok": True, "data": "summary", "api_calls": []},
            "api_calls",
        )
        req_ctx = SimpleNamespace(debug=False)

        assert _should_return_csv(response, req_ctx, return_directly=True) is False

    def test_query_debug_wraps_even_when_return_directly_requested(self):
        response = QueryResponse.from_agent_result(
            {"ok": True, "data": "summary", "result": [{"id": 1}], "api_calls": []},
            "api_calls",
        )
        req_ctx = SimpleNamespace(debug=True)

        assert _should_return_csv(response, req_ctx, return_directly=True) is False


class TestToCsv:
    """Tests for to_csv function."""

    def test_list_to_csv(self):
        """List converts to CSV with header."""
        data = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        result = to_csv(data)

        lines = result.strip().splitlines()
        assert len(lines) == 3
        assert lines[0] == "id,name"
        assert lines[1] == "1,a"
        assert lines[2] == "2,b"

    def test_single_object_to_csv(self):
        """Single object converts to single row CSV."""
        data = {"id": 1, "name": "test"}
        result = to_csv(data)

        lines = result.strip().splitlines()
        assert len(lines) == 2
        assert lines[0] == "id,name"
        assert lines[1] == "1,test"

    def test_empty_list_to_csv(self):
        """Empty list returns empty string."""
        assert to_csv([]) == ""

    def test_empty_none_to_csv(self):
        """None returns empty string."""
        assert to_csv(None) == ""

    def test_nested_objects_to_csv(self):
        """Nested objects get flattened by DuckDB."""
        data = [{"user": {"id": 1, "name": "a"}}, {"user": {"id": 2, "name": "b"}}]
        result = to_csv(data)

        lines = result.strip().splitlines()
        assert len(lines) == 3
        # DuckDB creates struct column
        assert "user" in lines[0]
