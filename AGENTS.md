# AGENTS.md

Universal agent guidance for this repo.

## Why

`api-agent` turns one GraphQL endpoint or REST OpenAPI spec into an MCP server. Users ask natural-language questions; the server introspects schema, calls the target API, stores tabular data in DuckDB, and runs SQL post-processing when the API cannot filter, rank, join, or aggregate directly.

## What

- `api_agent/__main__.py`: ASGI app, `/mcp`, `/health`
- `api_agent/config.py`: TOML-only app settings; `API_AGENT_CONFIG` selects file path
- `api_agent/context.py`: MCP request headers -> `RequestContext`
- `api_agent/tools/query.py`: public `{prefix}_query` tool
- `api_agent/agent/`: OpenAI Agents SDK GraphQL/REST flows
- `api_agent/rest/schema_loader.py`: OpenAPI 3.x + Swagger 2.0 schema loading
- `api_agent/rest/polling.py`: REST async polling tool
- `api_agent/recipe/`: learned recipe extraction, storage, matching, tool exposure
- `api_agent/query_response.py`: wrapped vs direct CSV response formatting
- `api_agent/executor.py`: DuckDB SQL execution and result shaping
- `tests/`: pytest coverage by module and behavior

## How

Use deterministic tools first.

```bash
uv sync --group dev
uv run pytest tests/ -v
uv run pytest tests/test_foo.py::test_bar -v
uv run ruff check api_agent/ tests/
uv run ruff format --check api_agent/ tests/
uv run ruff format api_agent/ tests/
uv run ty check
```

Before pushing, run the CI lint shape:

```bash
uv sync --group dev && uv run ruff check api_agent/ && uv run ty check && uv run pytest tests/ -q
```

Run local server:

```bash
uv run api-agent
# public: http://localhost:3000/mcp
```

Run local server with tracing/debug:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
  uv run --no-sync opentelemetry-instrument api-agent
```

`start.sh` uses `opentelemetry-instrument` when an OTEL endpoint is set. Do not manually wire OpenAI Agents SDK trace exporters in app code; `openai_agents` auto-instrumentation should load `OpenAIAgentsInstrumentor`.

Useful local override:

```bash
API_AGENT_CONFIG=api-agent.toml
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

Put app levers in `api-agent.toml`. Keep `OPENAI_API_KEY` env-only. `OPENAI_BASE_URL` may be TOML or env override. Do not add other app env overrides.

## Rules

- Keep changes testable: functional core, imperative shell.
- Prefer existing patterns; read nearby code before editing.
- Update README/docs when behavior, config, deploy, or public tools change.
- Keep root guidance short; put task-specific detail in README or focused docs.
- Do not hand-roll formatting/lint checks; run `ruff`/`ty`.
- For runtime/debug, verify actual TOML path and HTTP behavior.
- For plans, end with unresolved questions, very concise.
- Be extremely concise in chat and commit messages.

## Project Facts

- Requests require `X-Target-URL` and `X-API-Type`.
- `X-API-Name` overrides tool prefix; explicit hyphens preserved.
- Public MCP exposes `{prefix}_query` plus learned `r_{slug}` tools.
- `_execute` is removed; query + recipe tools are the public surface.
- `X-Recipe-Learn-Rate: 1` forces successful-query recipe extraction.
- Recipe params are required top-level fields; examples are hints, not defaults.
- Unsafe REST writes are blocked unless allowed by `X-Allow-Unsafe-Paths`.
