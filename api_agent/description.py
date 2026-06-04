"""Downstream API description generation for MCP tools."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from agents import Agent, AgentOutputSchema, Runner
from agents.exceptions import ModelBehaviorError
from pydantic import BaseModel, ConfigDict, Field

from .config import settings
from .store import ASYNC_API_AGENT_STORE

logger = logging.getLogger(__name__)

_MAX_DESCRIPTION_CONTEXT_CHARS = 12000
_MAX_FALLBACK_DESCRIPTION_CHARS = 300
_MAX_FIELD_COUNT = 40
_MAX_PATH_COUNT = 40
_FALLBACK_CACHE_TTL_SECONDS = 300
_GENERIC_GRAPHQL_TYPE_PARTS = (
    "audit",
    "connection",
    "deprecated",
    "edge",
    "input",
    "pageinfo",
    "result",
    "setting",
)
_GRAPHQL_DOMAIN_TYPE_PRIORITY = (
    "component",
    "service",
    "team",
    "module",
    "library",
    "job",
    "dataproject",
    "platform",
    "repository",
)


class DownstreamDescriptionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=40, max_length=500)


@dataclass(frozen=True)
class DescriptionResult:
    text: str
    ttl_seconds: int | None = None


DESCRIPTION_INSTRUCTIONS = """You are a downstream API description compiler. Convert a schema summary into one MCP tool description for the general query tool.

INPUT:
- api_type: "graphql" or "rest"
- hostname: downstream API hostname
- schema_summary: compact GraphQL introspection or OpenAPI summary

OUTPUT: Structured description.
{
  "description": "<client-facing general query tool description>"
}

DESCRIPTION REQUIREMENTS:
- 1-2 short sentences, 40-300 characters.
- Write for an agent choosing whether to call this general natural-language query tool.
- Start with what the agent can ask or look up through this tool.
- Say what the downstream service actually does, not how API Agent can query it.
- Prefer the service/product/domain name from title, description, tags, operation names, type names, or field names.
- Mention concrete resources and actions from the schema, such as service accounts, tokens, team accounts, settings, emails, objectives, bookings, payments, tickets, reports, or metrics.
- Keep it broad enough for the general query tool, but grounded in this service's real capabilities.
- Make clear this is for questions about that downstream service's data or workflows.
- For REST APIs with write operations, mention actions only when the schema summary clearly exposes them.
- Use active client-facing language: "Manage...", "Look up...", "Search...", "Inspect...".

DO NOT:
- Do not mention API Agent, gateway, proxy, wrapper, schema, OpenAPI, GraphQL, MCP, tool, or generated description.
- Do not say "this API", "the API", or "the service" without concrete domain context.
- Do not mention implementation details, endpoints, paths, HTTP methods, field counts, operation counts, or schema shape.
- Do not invent domains not supported by schema_summary.
- Do not include setup instructions, auth guidance, examples, markdown, or bullet points.
- Do not mention generic query mechanics like filtering, ranking, aggregation, joins, SQL, or live API calls.
"""


async def get_downstream_description(
    *,
    api_type: str,
    hostname: str,
    raw_schema: str,
    api_id: str,
    schema_hash: str,
) -> str:
    """Return cached/generated downstream API description with deterministic fallback."""
    cached = await _get_cached_description(api_id=api_id, schema_hash=schema_hash)
    if cached:
        return cached

    fallback = fallback_downstream_description(
        api_type=api_type,
        hostname=hostname,
        raw_schema=raw_schema,
    )
    try:
        result = await asyncio.wait_for(
            _generate_downstream_description(
                api_type=api_type,
                hostname=hostname,
                raw_schema=raw_schema,
            ),
            timeout=settings.DESCRIPTION_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.info("Timed out generating downstream API description")
        await _save_fallback_description(api_id=api_id, schema_hash=schema_hash, fallback=fallback)
        return fallback
    except Exception:
        logger.exception("Failed to generate downstream API description")
        await _save_fallback_description(api_id=api_id, schema_hash=schema_hash, fallback=fallback)
        return fallback

    await _save_cached_description(
        api_id=api_id,
        schema_hash=schema_hash,
        description=result.text,
        ttl_seconds=result.ttl_seconds,
    )
    return result.text


async def _save_fallback_description(*, api_id: str, schema_hash: str, fallback: str) -> None:
    await _save_cached_description(
        api_id=api_id,
        schema_hash=schema_hash,
        description=fallback,
        ttl_seconds=_FALLBACK_CACHE_TTL_SECONDS,
    )


async def _get_cached_description(*, api_id: str, schema_hash: str) -> str | None:
    try:
        return await ASYNC_API_AGENT_STORE.get_downstream_description(
            api_id=api_id,
            schema_hash=schema_hash,
        )
    except Exception:
        logger.exception("Failed to read downstream API description cache")
        return None


async def _save_cached_description(
    *, api_id: str, schema_hash: str, description: str, ttl_seconds: int | None = None
) -> None:
    try:
        # Failure fallbacks are cached briefly to avoid repeated model calls, then retried.
        await ASYNC_API_AGENT_STORE.save_downstream_description(
            api_id=api_id,
            schema_hash=schema_hash,
            description=description,
            ttl_seconds=ttl_seconds,
        )
    except Exception:
        logger.exception("Failed to write downstream API description cache")


def fallback_downstream_description(*, api_type: str, hostname: str, raw_schema: str = "") -> str:
    if api_type != "graphql":
        schema_description = _openapi_info_description(raw_schema)
        if schema_description:
            return schema_description
    else:
        schema_description = _graphql_schema_description(raw_schema, hostname)
        if schema_description:
            return schema_description

    api_label = "GraphQL" if api_type == "graphql" else "REST"
    return (
        f"[{hostname} {api_label} API] Ask a natural-language question about the configured API.\n\n"
        "Use when the client needs fresh API data, joins, filtering, ranking, or SQL-style "
        "post-processing. The agent reads the API schema, calls the target API, and returns "
        "the answer plus calls made."
    )


def _openapi_info_description(raw_schema: str) -> str:
    if not raw_schema:
        return ""
    try:
        schema = json.loads(raw_schema)
    except ValueError:
        return ""
    if not isinstance(schema, dict):
        return ""
    info = schema.get("info")
    if not isinstance(info, dict):
        return ""
    description = info.get("description")
    return _clean_fallback_description(description) if isinstance(description, str) else ""


def _clean_fallback_description(text: str) -> str:
    normalized = " ".join(text.split())
    normalized = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", normalized)
    normalized = re.sub(r"[`*_>#]+", "", normalized).strip()
    if len(normalized) <= _MAX_FALLBACK_DESCRIPTION_CHARS:
        return normalized
    return _first_sentence(normalized)


def _graphql_schema_description(raw_schema: str, hostname: str) -> str:
    if not raw_schema:
        return ""
    try:
        context = _graphql_description_context(json.loads(raw_schema))
    except ValueError:
        return ""

    query_fields = context.get("query_fields") or []
    domain_types = context.get("domain_types") or []
    field_names = [
        str(field["name"])
        for field in query_fields[:4]
        if isinstance(field, dict) and field.get("name")
    ]
    descriptions = [
        _first_sentence(str(item["description"]))
        for item in [*query_fields, *_useful_graphql_domain_types(domain_types)]
        if isinstance(item, dict) and isinstance(item.get("description"), str)
    ]

    if descriptions:
        return ". ".join(descriptions[:2]) + "."
    if field_names:
        return f"Ask about {hostname} data including {', '.join(field_names)}."
    return ""


def _useful_graphql_domain_types(domain_types: Any) -> list[dict[str, Any]]:
    if not isinstance(domain_types, list):
        return []
    useful = []
    for item in domain_types:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").lower()
        if any(part in name for part in _GENERIC_GRAPHQL_TYPE_PARTS):
            continue
        useful.append(item)
    return sorted(useful, key=_graphql_domain_type_rank)


def _first_sentence(text: str) -> str:
    normalized = " ".join(text.split())
    sentence = normalized.split(". ", 1)[0].strip().rstrip(".")
    return sentence[:240]


def _graphql_domain_type_rank(item: dict[str, Any]) -> tuple[int, str]:
    name = str(item.get("name") or "").lower()
    for index, priority_name in enumerate(_GRAPHQL_DOMAIN_TYPE_PRIORITY):
        if name == priority_name:
            return index, name
    return len(_GRAPHQL_DOMAIN_TYPE_PRIORITY), name


def _fallback_result(*, api_type: str, hostname: str, raw_schema: str) -> DescriptionResult:
    return DescriptionResult(
        text=fallback_downstream_description(
            api_type=api_type,
            hostname=hostname,
            raw_schema=raw_schema,
        ),
        ttl_seconds=_FALLBACK_CACHE_TTL_SECONDS,
    )


async def _generate_downstream_description(
    *,
    api_type: str,
    hostname: str,
    raw_schema: str,
) -> DescriptionResult:
    from .agent.model import client, create_openai_model, get_run_config
    from .agent.progress import reset_progress

    model_name = settings.DESCRIPTION_MODEL_NAME or settings.MODEL_NAME
    agent = Agent(
        name="downstream-description-generator",
        model=create_openai_model(settings.MODEL_API, model_name, client),
        instructions=DESCRIPTION_INSTRUCTIONS,
        tools=[],
        output_type=AgentOutputSchema(DownstreamDescriptionOutput, strict_json_schema=False),
    )
    payload = {
        "api_type": api_type,
        "hostname": hostname,
        "schema_summary": build_description_context(api_type=api_type, raw_schema=raw_schema),
    }
    try:
        reset_progress()
        result = await Runner.run(
            agent,
            json.dumps(payload, indent=2),
            max_turns=1,
            run_config=get_run_config(),
        )
    except ModelBehaviorError:
        logger.info("Invalid downstream description model output")
        return _fallback_result(api_type=api_type, hostname=hostname, raw_schema=raw_schema)

    output = result.final_output
    if isinstance(output, dict):
        output = DownstreamDescriptionOutput.model_validate(output)
    if not isinstance(output, DownstreamDescriptionOutput):
        return _fallback_result(api_type=api_type, hostname=hostname, raw_schema=raw_schema)
    return _clean_description_result(
        description=output.description,
        api_type=api_type,
        hostname=hostname,
        raw_schema=raw_schema,
    )


def build_description_context(*, api_type: str, raw_schema: str) -> dict[str, Any]:
    try:
        schema = json.loads(raw_schema)
    except ValueError:
        return {"raw_schema": raw_schema[:_MAX_DESCRIPTION_CONTEXT_CHARS]}

    if api_type == "graphql":
        return _graphql_description_context(schema)
    return _openapi_description_context(schema)


def _graphql_description_context(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"raw_schema": _truncate_json(schema)}

    types = schema.get("types", [])
    type_by_name = {t.get("name"): t for t in types if isinstance(t, dict)}
    query_type = schema.get("queryType") or {}
    mutation_type = schema.get("mutationType") or {}

    return {
        "query_fields": _graphql_fields(type_by_name.get(query_type.get("name"))),
        "mutation_fields": _graphql_fields(type_by_name.get(mutation_type.get("name"))),
        "domain_types": _graphql_domain_types(
            types,
            root_names={query_type.get("name"), mutation_type.get("name")},
        ),
    }


def _graphql_fields(type_def: Any) -> list[dict[str, Any]]:
    if not isinstance(type_def, dict):
        return []
    fields = type_def.get("fields") or []
    items = []
    for field in fields[:_MAX_FIELD_COUNT]:
        if not isinstance(field, dict):
            continue
        items.append(
            {
                "name": field.get("name"),
                "description": field.get("description"),
                "args": [
                    arg.get("name")
                    for arg in field.get("args") or []
                    if isinstance(arg, dict) and arg.get("name")
                ],
            }
        )
    return items


def _graphql_domain_types(types: Any, *, root_names: set[Any]) -> list[dict[str, Any]]:
    if not isinstance(types, list):
        return []
    domain_types = []
    for type_def in types:
        if not isinstance(type_def, dict):
            continue
        name = type_def.get("name")
        if not name or name in root_names or str(name).startswith("__"):
            continue
        fields = type_def.get("fields") or []
        if not fields:
            continue
        domain_types.append(
            {
                "name": name,
                "description": type_def.get("description"),
                "fields": [
                    field.get("name")
                    for field in fields[:12]
                    if isinstance(field, dict) and field.get("name")
                ],
            }
        )
        if len(domain_types) >= _MAX_FIELD_COUNT:
            break
    return domain_types


def _openapi_description_context(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"raw_schema": _truncate_json(schema)}

    paths = []
    for path, path_spec in (schema.get("paths") or {}).items():
        if len(paths) >= _MAX_PATH_COUNT:
            break
        if not isinstance(path_spec, dict):
            continue
        for method, operation in path_spec.items():
            if len(paths) >= _MAX_PATH_COUNT:
                break
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            if not isinstance(operation, dict):
                continue
            paths.append(
                {
                    "method": method.upper(),
                    "path": path,
                    "summary": operation.get("summary"),
                    "description": operation.get("description"),
                    "operation_id": operation.get("operationId"),
                    "tags": operation.get("tags") or [],
                }
            )

    return {
        "title": (schema.get("info") or {}).get("title"),
        "description": (schema.get("info") or {}).get("description"),
        "version": (schema.get("info") or {}).get("version"),
        "tags": schema.get("tags") or [],
        "paths": paths,
    }


def _clean_description_result(
    *, description: str, api_type: str, hostname: str, raw_schema: str
) -> DescriptionResult:
    text = " ".join(description.split())
    lowered = text.lower()
    forbidden = ("api agent", "wrapper", "mcp")
    if len(text) < 40 or any(term in lowered for term in forbidden):
        return _fallback_result(api_type=api_type, hostname=hostname, raw_schema=raw_schema)
    return DescriptionResult(text=text[:500])


def _truncate_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)[:_MAX_DESCRIPTION_CONTEXT_CHARS]
