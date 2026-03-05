# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup:**
```bash
uv sync --group dev
```

**Run server:**
```bash
uv run api-agent                    # Local dev
# Or direct (no clone): uvx --from git+https://github.com/agoda-com/api-agent api-agent
# Server starts on http://localhost:3000/mcp
```

**Tests:**
```bash
uv run pytest tests/ -v              # All tests
uv run pytest tests/test_foo.py -v   # Single test file
uv run pytest tests/test_foo.py::test_bar -v  # Single test
```

**Linting & Formatting:**
```bash
uv run ruff check api_agent/
uv run ruff check --fix api_agent/   # Auto-fix
uv run ruff format api_agent/        # Format
uv run ty check                      # Type check
```

**Docker:**
```bash
docker build -t api-agent .
docker run -p 3000:3000 -e OPENAI_API_KEY="..." api-agent
```

## Architecture

**MCP Server (FastMCP)** receives NL queries + headers → routes to **Agents** (OpenAI Agents SDK) → agents call target APIs + DuckDB for SQL processing.

### Request Flow

1. **Client** sends MCP request w/ headers (`X-Target-URL`, `X-API-Type`, `X-Target-Headers`)
2. **middleware.py**: `DynamicToolNamingMiddleware` transforms tool names per session (e.g., `_query` → `flights_api_query` based on URL)
3. **context.py**: Extracts `RequestContext` from headers
4. **tools/query.py**: Routes to GraphQL or REST agent
5. **agent/graphql_agent.py** or **agent/rest_agent.py**:
   - Fetches schema (introspection or OpenAPI)
   - Creates agent w/ dynamic tools (`graphql_query`/`rest_call`, `sql_query`, `search_schema`)
   - Runs agent loop (max 30 turns)
   - Returns results
6. **executor.py**: DuckDB integration for SQL post-processing

### Key Modules

- **api_agent/**: Main package
  - **__main__.py**: Entry point, creates FastMCP app w/ middleware
  - **config.py**: Settings via `pydantic-settings` (env vars w/ `API_AGENT_` prefix)
  - **context.py**: Header parsing → `RequestContext`, tool name generation
  - **middleware.py**: Dynamic tool naming per session
  - **tracing.py**: OpenTelemetry tracing via OTLP (uses [arize-otel](https://github.com/Arize-ai/openinference) for convenience, works with [Arize Phoenix](https://docs.arize.com/phoenix), Jaeger, Zipkin, Grafana Tempo, etc.)

- **api_agent/tools/**: MCP tool implementations
  - **query.py**: `_query` tool (NL → agent)
  - **execute.py**: `_execute` tool (direct GraphQL/REST call)

- **api_agent/agent/**: Agent logic (OpenAI Agents SDK)
  - **graphql_agent.py**: GraphQL agent w/ introspection, query building, SQL
  - **rest_agent.py**: REST agent w/ OpenAPI parsing, polling support
  - **prompts.py**: Shared system prompt fragments
  - **model.py**: LLM config (OpenAI-compatible)
  - **progress.py**: Turn tracking
  - **schema_search.py**: Grep-like schema search tool
  - **contextvar_utils.py**: Safe ContextVar access helpers

- **api_agent/recipe/**: Parameterized pipeline caching
  - **store.py**: `RecipeStore` (LRU in-memory cache, thread-safe)
  - **extractor.py**: Extract reusable recipes from agent runs
  - **runner.py**: Execute recipes outside agent context (for MCP tools)
  - **common.py**: Recipe validation, execution, parameter binding
  - **naming.py**: Tool name sanitization

- **api_agent/utils/**: Shared utilities
  - **csv.py**: CSV conversion via DuckDB (for recipe `return_directly` output)
  - **http_errors.py**: HTTP error response extraction (used by both clients)

- **api_agent/graphql/**: GraphQL client (httpx)
- **api_agent/rest/**: REST client (httpx) + OpenAPI loader (supports OpenAPI 3.x and Swagger 2.0)
- **api_agent/executor.py**: DuckDB SQL execution, table extraction, context truncation

### Context Management

All outputs capped at ~32k chars (`MAX_TOOL_RESPONSE_CHARS`) to prevent LLM overflow:
- **Query results**: Truncate by char count, show complete rows that fit
- **Schema**: Truncate large schemas, use `search_schema()` for exploration
- **Single objects**: Return DuckDB schema summary instead of full data

Agents use **ContextVar** for request isolation: `_graphql_queries`, `_query_results`, `_last_result`, `_raw_schema`. Use mutable containers (lists/dicts) since `ContextVar.set()` in child tasks doesn't propagate to parent.

### Tool Naming

Tools have internal names (`_query`, `_execute`) transformed by middleware per session:
- **Format**: `{prefix}_query`, `{prefix}_execute` — prefix from hostname or `X-API-Name` header
- Skips generic parts: TLDs, `api`, `qa`, `dev`, `internal`
- **Recipe tools**: `r_{slug}` (not API-specific), max 60 chars; `send_tool_list_changed()` notifies clients

### Safety

- **GraphQL**: Mutations blocked (queries only)
- **REST**: POST/PUT/DELETE/PATCH blocked by default, enable via `X-Allow-Unsafe-Paths` header (glob patterns)

### Polling (REST only)

Set `X-Poll-Paths` header to enable `poll_until_done` tool:
- Auto-increments `polling.count` in body between polls
- Checks `done_field` (dot-path like `"status"`, `"trips.0.isCompleted"`) against `done_value`
- Max 20 polls (configurable), default 3s delay

### Recipes

Caches parameterized API call + SQL pipelines from successful agent runs, exposed as MCP tools:

```
Query → Agent executes → Extractor LLM → Recipe stored → MCP tool `r_{name}` exposed
```

- **Storage**: LRU in-memory (default 64 entries), keyed by `(api_id, schema_hash)` - auto-invalidates on schema change
- **Deduplication**: Skips equivalent recipes, ensures unique tool names
- **Templating**: GraphQL `{{param}}`, REST `{"$param": "name"}`, SQL `{{param}}`
- **Config**: `ENABLE_RECIPES` (default: True), `RECIPE_CACHE_SIZE` (default: 64)

## After Code Changes

Always run before marking task complete:
```bash
uv run ruff check --fix api_agent/  # Lint + auto-fix
uv run ruff format api_agent/        # Format
uv run ty check                      # Type check
uv run pytest tests/ -v              # Tests
```

## Testing Notes

Tests use pytest-asyncio. Mock httpx for HTTP calls. See `tests/test_*.py` for patterns.

CI runs tests + linting on Python 3.11/3.12 (see `.github/workflows/test.yml`).
