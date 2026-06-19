"""Tests for executor utilities."""

import json

from api_agent.executor import (
    execute_sql,
    extract_tables_from_response,
    truncate_for_context,
)


class TestExtractTablesFromResponse:
    """Test extract_tables_from_response function."""

    def test_list_response_uses_name(self):
        """Direct list stores under explicit name, no schema."""
        data = [{"id": 1}, {"id": 2}]
        tables, schema = extract_tables_from_response(data, "users")

        assert tables == {"users": data}
        assert schema is None

    def test_single_list_key_uses_name(self):
        """Single list key extracts list under explicit name, no schema."""
        data = {"components": [{"id": 1}, {"id": 2}]}
        tables, schema = extract_tables_from_response(data, "active_users")

        assert tables == {"active_users": [{"id": 1}, {"id": 2}]}
        assert schema is None

    def test_multiple_list_keys_uses_first(self):
        """Multiple list keys: extracts first list under name, no schema."""
        data = {"users": [{"id": 1}], "posts": [{"id": 2}]}
        tables, schema = extract_tables_from_response(data, "api")

        assert "api" in tables
        assert len(tables) == 1
        assert schema is None

    def test_empty_dict_wraps_with_schema(self):
        """Empty dict wraps as single-row table with schema."""
        tables, schema = extract_tables_from_response({}, "test")

        assert tables == {"test": [{}]}
        assert schema is not None
        assert "rows" in schema

    def test_dict_without_lists_wraps_with_schema(self):
        """Dict without list values wraps whole dict with schema."""
        data = {"count": 5, "meta": {"page": 1}}
        tables, schema = extract_tables_from_response(data, "test")

        assert tables == {"test": [data]}
        assert schema is not None
        assert schema["rows"] == 1

    def test_empty_list_creates_table(self):
        """Empty list still creates table entry, no schema."""
        data = {"items": []}
        tables, schema = extract_tables_from_response(data, "empty_data")

        assert tables == {"empty_data": []}
        assert schema is None

    def test_mixed_dict_extracts_list(self):
        """Dict with mixed values extracts only the list, no schema."""
        data = {"users": [{"id": 1}], "count": 100, "meta": {"page": 1}}
        tables, schema = extract_tables_from_response(data, "api")

        assert tables == {"api": [{"id": 1}]}
        assert schema is None

    def test_non_dict_non_list_returns_empty(self):
        """Scalar values return empty tables, no schema."""
        tables, schema = extract_tables_from_response("string", "test")
        assert tables == {}
        assert schema is None

        tables, schema = extract_tables_from_response(123, "test")
        assert tables == {}
        assert schema is None

        tables, schema = extract_tables_from_response(None, "test")
        assert tables == {}
        assert schema is None

    def test_single_object_wrapped_with_schema(self):
        """Dict without list wraps whole dict with schema info."""
        data = {"user": {"id": 1, "name": "Alice"}}
        tables, schema = extract_tables_from_response(data, "user_data")

        assert tables == {"user_data": [{"user": {"id": 1, "name": "Alice"}}]}
        assert schema is not None
        assert "STRUCT" in schema["schema"]

    def test_nested_structure_stored_with_schema(self):
        """Nested dict without top-level list stores whole dict with schema."""
        data = {"response": {"data": {"id": 1, "status": "ok"}}}
        tables, schema = extract_tables_from_response(data, "api_response")

        assert tables == {"api_response": [data]}
        assert schema is not None
        assert schema["rows"] == 1


class TestExecuteSql:
    """Test execute_sql function."""

    def test_simple_select(self):
        """Basic SELECT works."""
        data = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
        result = execute_sql(data, "SELECT * FROM users")

        assert result["success"] is True
        assert len(result["result"]) == 2

    def test_select_with_where(self):
        """SELECT with WHERE clause."""
        data = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
        result = execute_sql(data, "SELECT * FROM users WHERE id = 1")

        assert result["success"] is True
        assert len(result["result"]) == 1
        assert result["result"][0]["name"] == "Alice"

    def test_select_with_limit(self):
        """SELECT with LIMIT."""
        data = {"items": [{"id": i} for i in range(100)]}
        result = execute_sql(data, "SELECT * FROM items LIMIT 10")

        assert result["success"] is True
        assert len(result["result"]) == 10

    def test_aggregation(self):
        """Aggregation functions work."""
        data = {"sales": [{"amount": 100}, {"amount": 200}, {"amount": 300}]}
        result = execute_sql(data, "SELECT SUM(amount) as total FROM sales")

        assert result["success"] is True
        assert result["result"][0]["total"] == 600

    def test_date_results_are_json_serializable(self):
        data = {"items": [{"id": 1, "due_date": "2026-05-29"}]}
        result = execute_sql(
            data,
            """
            SELECT
              CAST(due_date AS DATE) AS due_date,
              CAST(due_date || ' 12:34:56' AS TIMESTAMP) AS due_at,
              date_trunc('day', CAST(due_date AS DATE)) AS due_day
            FROM items
            """,
        )

        assert result == {
            "success": True,
            "result": [
                {
                    "due_date": "2026-05-29",
                    "due_at": "2026-05-29 12:34:56",
                    "due_day": "2026-05-29 00:00:00",
                }
            ],
        }
        json.dumps(result)

    def test_invalid_sql_returns_error(self):
        """Invalid SQL returns error."""
        data = {"users": [{"id": 1}]}
        result = execute_sql(data, "INVALID SQL SYNTAX")

        assert result["success"] is False
        assert "error" in result

    def test_missing_table_returns_error(self):
        """Query on non-existent table returns error."""
        data = {"users": [{"id": 1}]}
        result = execute_sql(data, "SELECT * FROM nonexistent")

        assert result["success"] is False
        assert "error" in result


class TestTruncateForContext:
    """Test truncate_for_context function."""

    def test_small_data_no_truncation(self):
        """Small data returns with table and rows metadata."""
        data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        result = truncate_for_context(data, "users")

        assert result["truncated"] is False
        assert result["data"] == data
        assert result["table"] == "users"
        assert result["rows"] == 2

    def test_large_data_truncated_with_schema(self):
        """Large data returns truncated with schema and complete rows."""
        # Create data larger than 32KB
        data = [{"id": i, "content": "x" * 1000} for i in range(100)]
        result = truncate_for_context(data, "big_table")

        assert result["truncated"] is True
        assert result["table"] == "big_table"
        assert result["rows"] == 100
        assert "schema" in result
        assert "showing" in result
        assert result["showing"] < 100
        # Data is always a list of complete rows
        assert isinstance(result["data"], list)
        assert len(result["data"]) == result["showing"]
        assert "hint" in result

    def test_custom_max_chars(self):
        """Respects custom max_chars limit."""
        data = [{"id": i, "content": "x" * 100} for i in range(50)]
        result = truncate_for_context(data, "test", max_chars=500)

        assert result["truncated"] is True
        assert isinstance(result["data"], list)
        # Serialized data should fit within limit
        assert len(json.dumps(result["data"])) <= 500

    def test_exact_limit_no_truncation(self):
        """Data exactly at limit doesn't truncate."""
        data = [{"id": 1}]
        data_str_len = len('[{"id": 1}]')
        result = truncate_for_context(data, "test", max_chars=data_str_len)

        assert result["truncated"] is False
        assert result["data"] == data
        assert result["table"] == "test"
        assert result["rows"] == 1

    def test_schema_contains_column_types(self):
        """Truncated result schema contains column types."""
        data = [{"id": i, "name": f"user{i}", "active": True, "score": 99.5} for i in range(100)]
        result = truncate_for_context(data, "users", max_chars=500)

        assert result["truncated"] is True
        assert "id" in result["schema"]
        assert "name" in result["schema"]
        # Should have DuckDB types
        assert "VARCHAR" in result["schema"] or "BIGINT" in result["schema"]
