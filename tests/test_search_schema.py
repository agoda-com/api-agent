"""Tests for search_schema tool."""

import json
import re
from contextvars import ContextVar

import pytest

from api_agent.agent.schema_search import create_search_schema_impl

_raw_schema: ContextVar[str] = ContextVar("test_raw_schema")
_search_schema_impl = create_search_schema_impl(_raw_schema)


class TestSearchSchema:
    """Test search_schema function on raw introspection JSON."""

    @pytest.fixture(autouse=True)
    def setup_schema(self):
        """Set up test schema in context var (raw introspection JSON)."""
        # Mimics GraphQL introspection response structure
        test_schema = {
            "queryType": {
                "fields": [
                    {"name": "users", "description": "Get all users", "args": [{"name": "limit"}]},
                    {"name": "posts", "description": "Get posts", "args": [{"name": "authorId"}]},
                ]
            },
            "types": [
                {
                    "name": "User",
                    "kind": "OBJECT",
                    "fields": [
                        {"name": "id", "type": {"name": "ID"}},
                        {"name": "name", "type": {"name": "String"}},
                        {"name": "email", "type": {"name": "String"}},
                    ],
                },
                {
                    "name": "Post",
                    "kind": "OBJECT",
                    "fields": [
                        {"name": "id", "type": {"name": "ID"}},
                        {"name": "title", "type": {"name": "String"}},
                        {"name": "author", "type": {"name": "User"}},
                    ],
                },
                {
                    "name": "Status",
                    "kind": "ENUM",
                    "enumValues": [{"name": "ACTIVE"}, {"name": "INACTIVE"}, {"name": "PENDING"}],
                },
            ],
        }
        _raw_schema.set(json.dumps(test_schema, indent=2))
        yield

    def test_simple_pattern_match(self):
        """Basic pattern matching works."""
        result = _search_schema_impl("User")

        assert "matches" in result
        assert "User" in result

    def test_regex_pattern(self):
        """Regex patterns work."""
        result = _search_schema_impl(r'"name":\s*"id"')

        # Matches field definitions with name: "id"
        assert "matches" in result

    def test_case_insensitive(self):
        """Search is case-insensitive."""
        result = _search_schema_impl("user")

        assert "matches" in result
        assert "User" in result

    def test_no_matches(self):
        """No matches returns empty."""
        result = _search_schema_impl("nonexistent_xyz")

        assert result == "(no matches)"

    def test_context_lines(self):
        """Context lines are included."""
        result = _search_schema_impl("email", context=1)

        # Should include lines around "email"
        assert "name" in result or "String" in result

    def test_invalid_regex(self):
        """Invalid regex returns error."""
        result = _search_schema_impl("[invalid")

        assert "error: invalid regex" in result

    def test_line_numbers(self):
        """Output includes line numbers."""
        result = _search_schema_impl("title")

        # Match line has ":" separator
        assert ":" in result
        # Should have numeric line number
        lines = result.split("\n")
        assert any(line.split(":")[0].split("-")[0].isdigit() for line in lines if line)

    def test_match_separator(self):
        """Match lines use : separator, context uses -."""
        result = _search_schema_impl("email", context=1)

        lines = [ln for ln in result.split("\n") if ln and not ln.startswith("(")]
        match_lines = [ln for ln in lines if "email" in ln.lower()]
        assert all(":" in ln for ln in match_lines)

    def test_enum_search(self):
        """Can find enum values."""
        result = _search_schema_impl("ACTIVE")

        assert "matches" in result
        assert "ACTIVE" in result

    def test_before_context(self):
        """before param shows lines before match."""
        result = _search_schema_impl("email", before=3, after=0)

        # Should show lines before email
        assert "name" in result or "String" in result

    def test_after_context(self):
        """after param shows lines after match."""
        result = _search_schema_impl('"name": "User"', before=0, after=3)

        # Should show lines after User type definition
        assert "OBJECT" in result or "fields" in result

    def test_description_preserved(self):
        """Raw JSON preserves descriptions (unlike DSL)."""
        result = _search_schema_impl("Get all users")

        assert "matches" in result
        assert "Get all users" in result

    def test_char_cap_truncates_and_hints(self):
        """Character cap truncates output and includes offset guidance."""
        large_schema = {"items": [{"id": i, "name": f"item_{i}"} for i in range(100)]}
        _raw_schema.set(json.dumps(large_schema, indent=2))

        result = _search_schema_impl('"id":', context=0, max_chars=400)

        assert "TRUNCATED" in result
        assert "offset=" in result
        assert len(result) <= 400

        header = result.split("\n")[0]
        match = re.search(r"showing (\d+)", header)
        assert match is not None
        showing = int(match.group(1))
        next_offset = showing
        assert f"offset={next_offset}" in result

    def test_offset_paginates_results(self):
        """Offset skips earlier matches and updates header."""
        large_schema = {"items": [{"id": i, "name": f"item_{i}"} for i in range(30)]}
        _raw_schema.set(json.dumps(large_schema, indent=2))

        result = _search_schema_impl('"id":', context=0, offset=10, max_chars=800)

        assert "starting at match 11" in result
        header = result.split("\n")[0]
        assert "showing" in header
        first_block = result.split("\n", 1)[1].split("\n--\n")[0]
        assert '"id": 10' in first_block

    def test_offset_beyond_range(self):
        """Offset larger than match count returns friendly error."""
        large_schema = {"items": [{"id": i} for i in range(5)]}
        _raw_schema.set(json.dumps(large_schema, indent=2))

        result = _search_schema_impl('"id":', offset=10)

        assert "offset 10 is beyond available results" in result

    def test_negative_offset_rejected(self):
        """Negative offsets are not allowed."""
        result = _search_schema_impl("User", offset=-1)

        assert "error: offset must be >= 0" in result

    def test_tiny_char_cap_returns_warning(self):
        """Extremely small char caps prompt guidance."""
        result = _search_schema_impl("User", context=0, max_chars=30)

        assert "Unable to show any results" in result

    def test_order_preserved_with_char_cap(self):
        """Even with char caps, matches stay in ascending order."""
        large_schema = {"items": [{"id": i} for i in range(50)]}
        _raw_schema.set(json.dumps(large_schema, indent=2))

        result = _search_schema_impl('"id":', context=0, max_chars=800)

        lines = [ln for ln in result.split("\n") if ":" in ln and not ln.startswith("(")]
        line_numbers = [int(ln.split(":")[0]) for ln in lines if ln.split(":")[0].isdigit()]
        assert line_numbers == sorted(line_numbers)


class TestSearchSchemaNoContext:
    """Test search_schema without schema loaded."""

    def test_no_schema_loaded(self):
        """Returns error when no schema."""
        _raw_schema.set("")

        result = _search_schema_impl("test")

        assert "error: schema empty" in result
