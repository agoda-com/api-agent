"""Executor for GraphQL queries and DuckDB SQL processing."""

import json
import logging
import os
import tempfile
from typing import Any

import duckdb

from .graphql import execute_query as graphql_execute

logger = logging.getLogger(__name__)


def _quote_identifier(identifier: str) -> str:
    """Return a DuckDB quoted identifier."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def extract_tables_from_response(
    data: Any, name: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Extract data from API response for DuckDB tables.

    DuckDB handles nested JSON natively:
    - STRUCT for nested objects (access via dot notation: user.name)
    - Arrays for lists (use unnest() to expand)

    Args:
        data: API response data (dict or list)
        name: User-specified table name

    Returns:
        Tuple of (tables_dict, schema_info_or_none)
        - tables_dict: {name: data} for storage
        - schema_info: Only for wrapped dicts (1-row tables), None otherwise
    """
    # Direct list response
    if isinstance(data, list):
        return {name: data}, None

    if not isinstance(data, dict):
        return {}, None

    # Dict response: prefer top-level list if exists
    for value in data.values():
        if isinstance(value, list):
            return {name: value}, None

    # No list found - wrap whole dict as single-row table + extract schema
    wrapped = [data]
    schema_info = _extract_schema(wrapped, name)
    return {name: wrapped}, schema_info


def _extract_schema(data: list[dict], table_name: str) -> dict[str, Any]:
    """Extract DuckDB schema from data (internal helper).

    Creates one temp file to get schema info for wrapped dicts.
    """
    if not data:
        return {"rows": 0, "schema": "", "hint": "Empty table"}

    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file = f.name

        with duckdb.connect() as conn:
            conn.execute(
                f"CREATE TABLE {_quote_identifier(table_name)} AS SELECT * FROM read_json_auto(?)",
                [temp_file],
            )
            schema = conn.execute(f"DESCRIBE {_quote_identifier(table_name)}").fetchall()

        schema_str = ", ".join([f"{col[0]}: {col[1]}" for col in schema])

        return {
            "rows": len(data),
            "schema": schema_str,
            "hint": "Use sql_query() to access fields. Example: "
            f"SELECT {_quote_identifier(schema[0][0])} "
            f"FROM {_quote_identifier(table_name)}",
        }
    except Exception as e:
        logger.exception("Schema extraction error")
        return {"rows": len(data), "schema": "unknown", "hint": str(e)}
    finally:
        if temp_file:
            try:
                os.unlink(temp_file)
            except OSError:
                pass


def truncate_for_context(
    data: list[dict], table_name: str, max_chars: int | None = None
) -> dict[str, Any]:
    """Truncate data for LLM context safety, include schema if truncated.

    Args:
        data: List of records
        table_name: Name for the table
        max_chars: Max chars for response (defaults to settings.MAX_TOOL_RESPONSE_CHARS)

    Returns:
        Always: {"table", "rows", "data" (list), "truncated", "showing"?, "schema"?, "hint"?}
    """
    from .config import settings

    max_chars = max_chars or settings.MAX_TOOL_RESPONSE_CHARS
    total_rows = len(data)

    # Check if full data fits
    if len(json.dumps(data)) <= max_chars:
        return {"table": table_name, "rows": total_rows, "data": data, "truncated": False}

    # Find how many complete rows fit
    preview: list[dict] = []
    current_size = 2  # "[]"
    for row in data:
        row_json = json.dumps(row)
        new_size = current_size + len(row_json) + (1 if preview else 0)
        if new_size > max_chars:
            break
        preview.append(row)
        current_size = new_size

    schema = _extract_schema(data, table_name)
    return {
        "table": table_name,
        "rows": total_rows,
        "showing": len(preview),
        "schema": schema.get("schema", ""),
        "data": preview,
        "truncated": True,
        "hint": f"Showing {len(preview)}/{total_rows}. Use sql_query to filter.",
    }


def _query_result_rows(conn: duckdb.DuckDBPyConnection, query: str) -> list[dict[str, Any]]:
    conn.sql(query).create("__sql_result")
    serialized_rows = conn.execute(
        "SELECT to_json(struct_pack(*COLUMNS(*))) FROM __sql_result"
    ).fetchall()
    return [json.loads(row[0]) for row in serialized_rows]


def execute_sql(data: Any, query: str) -> dict[str, Any]:
    """Execute SQL query on JSON data using DuckDB.

    Args:
        data: JSON data (dict or list) to query
        query: DuckDB SQL query (use table names matching top-level keys, e.g. 'posts')

    Returns:
        Dict with success/result or error
    """
    temp_files = []
    try:
        with duckdb.connect() as conn:
            # Register top-level keys as tables via temp JSON files
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list) and value:
                        # Write to temp file for DuckDB to read
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".json", delete=False
                        ) as f:
                            json.dump(value, f)
                            temp_files.append(f.name)
                        conn.execute(
                            f"CREATE TABLE {_quote_identifier(key)} AS "
                            "SELECT * FROM read_json_auto(?)",
                            [f.name],
                        )
            elif isinstance(data, list):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                    json.dump(data, f)
                    temp_files.append(f.name)
                conn.execute("CREATE TABLE data AS SELECT * FROM read_json_auto(?)", [f.name])

            conn.execute("SET enable_external_access = false")
            conn.execute("SET lock_configuration = true")

            rows = _query_result_rows(conn, query)

        return {"success": True, "result": rows}

    except duckdb.Error as e:
        return {"success": False, "error": f"SQL error: {e}"}
    except Exception as e:
        logger.exception("SQL execution error")
        return {"success": False, "error": str(e)}
    finally:
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass


async def execute_graphql(
    query: str,
    variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute GraphQL query (wrapper around graphql client).

    Args:
        query: GraphQL query string
        variables: Optional query variables

    Returns:
        Dict with success/data or error
    """
    return await graphql_execute(query, variables)
