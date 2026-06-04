# Python with DuckDB for data processing
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install CA certificates for HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock README.md api-agent.toml ./

# Install dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Install OpenTelemetry instrumentation libraries
RUN uv run --no-sync opentelemetry-bootstrap -a requirements > /tmp/otel-requirements.txt \
    && uv pip install -r /tmp/otel-requirements.txt \
    && rm /tmp/otel-requirements.txt

COPY api_agent ./api_agent

# Install project
RUN uv pip install --no-deps .

COPY start.sh ./

EXPOSE 3000

RUN chmod +x ./start.sh

ENTRYPOINT ["/app/start.sh"]
