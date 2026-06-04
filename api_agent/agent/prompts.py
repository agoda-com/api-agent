"""Shared prompt components for API agents."""

# Context section with date and limits
CONTEXT_SECTION = """<context>
Today's date: {current_date}
Max tool calls: {max_turns}
Use today's date to calculate relative dates (tomorrow, next week, etc.)
</context>"""

# Reasoning guidance for GPT-5.x low-effort runs
REASONING_GUIDANCE = """<reasoning-guidance>
- Before tool calls, internally choose the smallest API/SQL plan that can answer the request.
- Do not expose chain-of-thought; final answers should give results, assumptions, and concise method notes only.
- For multi-step tasks, verify each step's output shape before the next call.
- If evidence is insufficient, say what is missing instead of guessing.
</reasoning-guidance>"""

# SQL rules (shared)
SQL_RULES = """<sql-rules>
- API responses TRUNCATED; full data in DuckDB table
- sql_query for filtering, sorting, aggregation, joins
- Unique table names via 'name' param
- Structs: t.field.subfield (dot notation)
- Arrays: len(arr), arr[1] (1-indexed)
- UUIDs: CAST(id AS VARCHAR)
- In COALESCE, CASE, UNION, and mixed-type expressions, cast IDs/dates/enums to VARCHAR
- UNNEST: FROM t, UNNEST(t.arr) AS u(val) → t.col for original, u.val for element
- EXCLUDE: SELECT * EXCLUDE (col) FROM t (not t.* EXCLUDE)
- If joins/CTEs share column names, always qualify columns (e.g., table_alias.column)
</sql-rules>"""

# Ambiguity handling
UNCERTAINTY_SPEC = """<uncertainty>
- Ambiguous query: state your interpretation, then answer
- If missing critical inputs, ask 1-2 precise questions; otherwise state assumptions and proceed
- Never fabricate figures—only report what API returned
</uncertainty>"""

# Tool usage rules
TOOL_USAGE_RULES = """<tool-usage>
- Prefer tools for user-specific or fresh data
- Avoid redundant tool calls
- Parallelize independent reads when possible
</tool-usage>"""

# Optional parameters handling
OPTIONAL_PARAMS_SPEC = """<optional-params>
- Schema shows only required fields. Use search_schema to find optional fields.
- Don't invent values (IDs, usernames, etc.) - only use what user provides
</optional-params>"""

# Persistence on errors
PERSISTENCE_SPEC = """<persistence>
- If API call fails, analyze error and retry with corrected params
- Retry only when the error suggests a concrete fix
- Stop early when the answer is ready or the remaining path is not useful
</persistence>"""

# Effective patterns (reward good behaviors)
EFFECTIVE_PATTERNS = """<effective-patterns>
- Infer implicit params from user context
- Read schema for valid enum/type values
- Name tables descriptively
- Adapt SQL syntax on failure
- Use sensible defaults for pagination/limits
- Stop once the answer is ready; avoid extra tool calls
</effective-patterns>"""

# Decision guidance for choosing between recipes, API calls, and SQL pipelines
DECISION_GUIDANCE = """<decision-guidance>
When to use each approach:

RECIPES (if listed in <recipes> above):
- Check available recipe tools before direct API/SQL calls
- Score >= 0.7: Strong match, prefer recipe if params available
- Score < 0.7: Consider direct API/SQL instead
- Use when question very similar to past query
- SKIP if params unclear or question differs
- Recipe tools require all listed params; no implicit defaults at call-time

DIRECT API CALLS (rest_call, graphql_query):
- Simple data retrieval, no filtering needed
- User wants raw unprocessed data
- Single endpoint sufficient
- Set return_directly=true if no LLM analysis needed

SQL PIPELINES (API + sql_query):
- Filtering, sorting, aggregation required
- Joining multiple data sources
- Complex transformations
- User asks: "filtered", "sorted", "top N", "average", "grouped"

RETURN_DIRECTLY FLAG (on graphql_query, rest_call, sql_query, recipe tools):
When true: YOU still call the tool, but YOUR final response is skipped. Raw data goes directly to user.

Use return_directly=true when:
- User wants data as-is: "list", "get", "fetch", "show"
- Examples: "list users", "get order 123", "show products"

Use return_directly=false when:
- User needs YOU to answer: "why", "how many", "which", "best"
- Examples: "how many users", "why failed", "which is cheapest"

Default: recipes=true, others=false. Only on success.
</decision-guidance>"""

# Tool descriptions
REST_TOOL_DESC = """rest_call(method, path, path_params?, query_params?, body?, name?, return_directly?)
  Execute REST API call. Store result as DuckDB table for sql_query.
  - return_directly: Skip LLM analysis, return raw data directly to user
  Use for: direct reads, exploratory calls, when no recipe matches"""

SQL_TOOL_DESC = """sql_query(sql, return_directly?)
  Query DuckDB tables from previous API calls. For filtering, aggregation, joins.
  - return_directly: Skip LLM processing, return query results directly
  Use for: filtering, sorting, analytics, combining multiple API responses"""

SEARCH_TOOL_DESC = """search_schema(pattern, context=10, offset=0)
  Regex search on schema JSON. Returns matching lines with context.
  Use offset to paginate if results truncated."""

# Schema notation for REST
REST_SCHEMA_NOTATION = """<schema_notation>
METHOD /path(params) -> Type = endpoint signature
param?: Type = optional param | param: Type = required param
Type {{ field: type! }} = required field | {{ field: type }} = optional field
Type[] = array of Type
str(date-time) = ISO 8601 format: YYYY-MM-DDTHH:MM:SS
str(date) = ISO 8601 date: YYYY-MM-DD
</schema_notation>"""

# Schema notation for GraphQL
GRAPHQL_SCHEMA_NOTATION = """<schema_notation>
Type = object | Type! = non-null | [Type] = list
query(args) -> ReturnType = query signature
TypeName {{ field: Type }} = object fields
# comment = description
</schema_notation>"""
