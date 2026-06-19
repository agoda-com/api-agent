import asyncio
import json

import pytest

from api_agent import description
from api_agent.store import AsyncApiAgentStore, MemoryApiAgentStore

GRAPHQL_API_ID = "graphql:https://api.example.com/graphql"
GRAPHQL_SCHEMA = '{"types":[]}'


@pytest.fixture(autouse=True)
def description_store(monkeypatch):
    store = MemoryApiAgentStore(max_size=10)
    monkeypatch.setattr(description, "ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store))
    return store


@pytest.mark.asyncio
async def test_description_generation_is_cached(monkeypatch):
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return description.DescriptionResult(
            text="Query objectives, key results, owners, status, and cycle data."
        )

    monkeypatch.setattr(description, "_generate_downstream_description", generate)

    first = await description.get_downstream_description(
        api_type="rest",
        hostname="okr.example.com",
        raw_schema='{"openapi":"3.0.0"}',
        api_id="rest:https://spec|https://api",
        schema_hash="abc",
    )
    second = await description.get_downstream_description(
        api_type="rest",
        hostname="okr.example.com",
        raw_schema='{"openapi":"3.0.0"}',
        api_id="rest:https://spec|https://api",
        schema_hash="abc",
    )

    assert first == "Query objectives, key results, owners, status, and cycle data."
    assert second == first
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_description_generation_falls_back_without_failing_list_tools(monkeypatch):
    async def generate(**_kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(description, "_generate_downstream_description", generate)

    result = await description.get_downstream_description(
        api_type="graphql",
        hostname="api.example.com",
        raw_schema=GRAPHQL_SCHEMA,
        api_id=GRAPHQL_API_ID,
        schema_hash="abc",
    )

    assert result == description.fallback_downstream_description(
        api_type="graphql",
        hostname="api.example.com",
    )


@pytest.mark.asyncio
async def test_fallback_is_cached_briefly_after_generation_failure(monkeypatch, description_store):
    async def fail(**_kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(description, "_generate_downstream_description", fail)
    result = await description.get_downstream_description(
        api_type="graphql",
        hostname="api.example.com",
        raw_schema=GRAPHQL_SCHEMA,
        api_id=GRAPHQL_API_ID,
        schema_hash="abc",
    )

    assert result == description.fallback_downstream_description(
        api_type="graphql",
        hostname="api.example.com",
    )
    assert description_store.get_downstream_description(
        api_id=GRAPHQL_API_ID,
        schema_hash="abc",
    ) == description.fallback_downstream_description(
        api_type="graphql",
        hostname="api.example.com",
        raw_schema=GRAPHQL_SCHEMA,
    )


@pytest.mark.asyncio
async def test_generated_fallback_uses_ttl_cache(monkeypatch, description_store):
    async def fallback_result(**kwargs):
        return description._fallback_result(**kwargs)

    monkeypatch.setattr(description, "_generate_downstream_description", fallback_result)

    result = await description.get_downstream_description(
        api_type="graphql",
        hostname="api.example.com",
        raw_schema=GRAPHQL_SCHEMA,
        api_id=GRAPHQL_API_ID,
        schema_hash="abc",
    )

    assert result == description.fallback_downstream_description(
        api_type="graphql",
        hostname="api.example.com",
        raw_schema=GRAPHQL_SCHEMA,
    )
    cached = description_store._downstream_descriptions[(GRAPHQL_API_ID, "abc")]
    assert cached[0] == result
    assert cached[1] is not None


@pytest.mark.asyncio
async def test_description_generation_times_out_to_fallback(monkeypatch, description_store):
    async def slow(**_kwargs):
        await asyncio.sleep(1)
        return "Query objectives, key results, owners, status, and cycle data."

    monkeypatch.setattr(description.settings, "DESCRIPTION_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(description, "_generate_downstream_description", slow)

    result = await description.get_downstream_description(
        api_type="graphql",
        hostname="api.example.com",
        raw_schema=GRAPHQL_SCHEMA,
        api_id=GRAPHQL_API_ID,
        schema_hash="abc",
    )

    assert result == description.fallback_downstream_description(
        api_type="graphql",
        hostname="api.example.com",
    )
    assert description_store.get_downstream_description(
        api_id=GRAPHQL_API_ID,
        schema_hash="abc",
    ) == description.fallback_downstream_description(
        api_type="graphql",
        hostname="api.example.com",
        raw_schema=GRAPHQL_SCHEMA,
    )


def test_rest_fallback_without_openapi_description_uses_hardcoded_query_description():
    raw_schema = json.dumps(
        {
            "openapi": "3.0.0",
            "info": {"title": "FastAPI"},
            "paths": {"/api/teams/": {"get": {"tags": ["teams"], "summary": "Get Teams"}}},
        }
    )

    result = description.fallback_downstream_description(
        api_type="rest",
        hostname="api.example-data.io",
        raw_schema=raw_schema,
    )

    assert result.startswith("[api.example-data.io REST API] Ask a natural-language question")
    assert "fresh API data" in result


def test_fallback_prefers_openapi_description():
    raw_schema = json.dumps(
        {
            "openapi": "3.0.0",
            "info": {
                "title": "Objectives API",
                "description": "Manage objectives and key results.",
            },
            "paths": {},
        }
    )

    result = description.fallback_downstream_description(
        api_type="rest",
        hostname="okr.example.com",
        raw_schema=raw_schema,
    )

    assert result == "Manage objectives and key results."


def test_rest_fallback_cleans_and_bounds_long_openapi_description():
    raw_schema = json.dumps(
        {
            "openapi": "3.0.0",
            "info": {
                "title": "Objectives API",
                "description": (
                    "# Objectives API\n\n"
                    "**Search objectives, key results, owners, and status.** "
                    "This markdown-heavy description includes long implementation notes. "
                    + "Extra details. "
                    * 80
                ),
            },
            "paths": {},
        }
    )

    result = description.fallback_downstream_description(
        api_type="rest",
        hostname="okr.example.com",
        raw_schema=raw_schema,
    )

    assert result == "Objectives API Search objectives, key results, owners, and status"
    assert len(result) <= 300
    assert "#" not in result
    assert "*" not in result


def test_graphql_fallback_uses_introspection_descriptions():
    raw_schema = json.dumps(
        {
            "queryType": {"name": "Query"},
            "types": [
                {
                    "name": "Query",
                    "fields": [
                        {
                            "name": "components",
                            "description": "Search component catalog records.",
                            "args": [],
                        }
                    ],
                },
                {
                    "name": "Component",
                    "description": "Component ownership, dependencies, and lifecycle metadata.",
                    "fields": [{"name": "id"}],
                },
            ],
        }
    )

    result = description.fallback_downstream_description(
        api_type="graphql",
        hostname="catalog.example.com",
        raw_schema=raw_schema,
    )

    assert result == (
        "Search component catalog records. "
        "Component ownership, dependencies, and lifecycle metadata."
    )


def test_graphql_fallback_uses_query_field_names_without_introspection_descriptions():
    raw_schema = json.dumps(
        {
            "queryType": {"name": "Query"},
            "types": [
                {
                    "name": "Query",
                    "fields": [{"name": "components", "args": []}],
                }
            ],
        }
    )

    result = description.fallback_downstream_description(
        api_type="graphql",
        hostname="catalog.example.com",
        raw_schema=raw_schema,
    )

    assert result == "Ask about catalog.example.com data including components."


def test_graphql_fallback_skips_generic_domain_types():
    raw_schema = json.dumps(
        {
            "queryType": {"name": "Query"},
            "types": [
                {
                    "name": "ActionResult",
                    "description": "Generic result returned by mutations.",
                    "fields": [{"name": "success"}],
                },
                {
                    "name": "AuditEvent",
                    "description": "Audit metadata recording when an action occurred.",
                    "fields": [{"name": "at"}],
                },
                {
                    "name": "Compliance",
                    "description": "Component-level compliance classification.",
                    "fields": [{"name": "id"}],
                },
                {
                    "name": "Component",
                    "description": "Component is the central catalog entity. Used for ownership.",
                    "fields": [{"name": "id"}],
                },
            ],
        }
    )

    result = description.fallback_downstream_description(
        api_type="graphql",
        hostname="catalog.example.com",
        raw_schema=raw_schema,
    )

    assert result == (
        "Component is the central catalog entity. Component-level compliance classification."
    )


def test_openapi_description_context_uses_spec_domain():
    raw_schema = json.dumps(
        {
            "openapi": "3.0.0",
            "info": {
                "title": "Objectives API",
                "description": "Manage objectives and key results.",
                "version": "1.0",
            },
            "paths": {
                "/objectives": {
                    "get": {
                        "summary": "List objectives",
                        "operationId": "listObjectives",
                        "tags": ["Objectives"],
                    }
                },
                "/objectives/{id}/key-results": {
                    "get": {
                        "summary": "List key results",
                        "operationId": "listKeyResults",
                    }
                },
            },
        }
    )

    context = description.build_description_context(api_type="rest", raw_schema=raw_schema)

    assert context["title"] == "Objectives API"
    assert context["description"] == "Manage objectives and key results."
    assert context["paths"][0]["summary"] == "List objectives"
    assert context["paths"][1]["operation_id"] == "listKeyResults"


def test_graphql_description_context_uses_query_fields_and_domain_types():
    raw_schema = json.dumps(
        {
            "queryType": {"name": "Query"},
            "mutationType": {"name": "Mutation"},
            "types": [
                {
                    "name": "Query",
                    "fields": [
                        {
                            "name": "objectives",
                            "description": "Search objectives",
                            "args": [{"name": "cycle"}],
                        }
                    ],
                },
                {
                    "name": "Objective",
                    "description": "Company objective",
                    "fields": [{"name": "id"}, {"name": "status"}],
                },
            ],
        }
    )

    context = description.build_description_context(api_type="graphql", raw_schema=raw_schema)

    assert context["query_fields"] == [
        {"name": "objectives", "description": "Search objectives", "args": ["cycle"]}
    ]
    assert context["domain_types"][0]["name"] == "Objective"


def test_description_prompt_has_strict_output_contract():
    prompt = description.DESCRIPTION_INSTRUCTIONS

    assert "OUTPUT: Structured description." in prompt
    assert '"description": "<client-facing general query tool description>"' in prompt
    assert "DESCRIPTION REQUIREMENTS:" in prompt
    assert "DO NOT:" in prompt


def test_description_prompt_forbids_internal_terms():
    prompt = description.DESCRIPTION_INSTRUCTIONS

    assert "Do not mention API Agent" in prompt
    assert "OpenAPI, GraphQL, MCP" in prompt
    assert "Do not mention implementation details" in prompt


def test_description_prompt_has_guidance_not_examples():
    prompt = description.DESCRIPTION_INSTRUCTIONS

    assert "GOOD EXAMPLES" not in prompt
    assert "BAD EXAMPLES" not in prompt
    assert "Say what the downstream service actually does" in prompt
    assert "agent choosing whether to call this general natural-language query tool" in prompt


@pytest.mark.asyncio
async def test_description_model_name_is_optional(monkeypatch):
    used = {}

    class DummyAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    async def run(*_args, **_kwargs):
        return type(
            "Result",
            (),
            {
                "final_output": description.DownstreamDescriptionOutput(
                    description="Query objectives, key results, owners, status, and cycle data."
                )
            },
        )()

    def create_model(api, model_name, client):
        used["api"] = api
        used["model_name"] = model_name
        used["client"] = client
        return object()

    monkeypatch.setattr(description.settings, "DESCRIPTION_MODEL_NAME", "")
    monkeypatch.setattr(description.settings, "MODEL_NAME", "gpt-main")
    monkeypatch.setattr(description, "Agent", DummyAgent)
    monkeypatch.setattr(description.Runner, "run", run)
    monkeypatch.setattr("api_agent.agent.model.create_openai_model", create_model)

    result = await description._generate_downstream_description(
        api_type="graphql",
        hostname="api.example.com",
        raw_schema='{"types":[]}',
    )

    assert result.text == "Query objectives, key results, owners, status, and cycle data."
    assert result.ttl_seconds is None
    assert used["model_name"] == "gpt-main"


@pytest.mark.asyncio
async def test_invalid_generated_description_uses_rest_description_fallback(monkeypatch):
    class DummyAgent:
        def __init__(self, **_kwargs):
            pass

    async def run(*_args, **_kwargs):
        return type("Result", (), {"final_output": "not structured"})()

    raw_schema = json.dumps(
        {
            "openapi": "3.0.0",
            "info": {"description": "Manage objectives and key results."},
            "paths": {},
        }
    )
    monkeypatch.setattr(description, "Agent", DummyAgent)
    monkeypatch.setattr(description.Runner, "run", run)
    monkeypatch.setattr("api_agent.agent.model.create_openai_model", lambda *_args: object())

    result = await description._generate_downstream_description(
        api_type="rest",
        hostname="okr.example.com",
        raw_schema=raw_schema,
    )

    assert result.text == "Manage objectives and key results."
    assert result.ttl_seconds == description._FALLBACK_CACHE_TTL_SECONDS


@pytest.mark.asyncio
async def test_description_generation_initializes_turn_context(monkeypatch):
    class DummyAgent:
        def __init__(self, **_kwargs):
            pass

    async def run(*_args, **_kwargs):
        from api_agent.agent.progress import get_turn_context

        assert get_turn_context(1) == "Turn 0/1"
        return type(
            "Result",
            (),
            {
                "final_output": description.DownstreamDescriptionOutput(
                    description="Query objectives, key results, owners, status, and cycle data."
                )
            },
        )()

    monkeypatch.setattr(description, "Agent", DummyAgent)
    monkeypatch.setattr(description.Runner, "run", run)
    monkeypatch.setattr("api_agent.agent.model.create_openai_model", lambda *_args: object())

    result = await description._generate_downstream_description(
        api_type="graphql",
        hostname="api.example.com",
        raw_schema='{"types":[]}',
    )

    assert result.text == "Query objectives, key results, owners, status, and cycle data."
    assert result.ttl_seconds is None
