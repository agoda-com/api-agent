#!/bin/sh
[ -n "${API_AGENT_CONFIG:-}" ] || API_AGENT_CONFIG=/app/api-agent.toml
export API_AGENT_CONFIG

if [ -n "${OTEL_EXPORTER_OTLP_ENDPOINT:-}" ] || [ -n "${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-}" ]; then
  exec uv run --no-sync opentelemetry-instrument api-agent
fi

exec uv run --no-sync api-agent
