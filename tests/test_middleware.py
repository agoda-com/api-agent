"""Tests for dynamic tool naming middleware."""

from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from fastmcp.exceptions import NotFoundError, ValidationError
from fastmcp.server.middleware import MiddlewareContext
from mcp import types as mt
from mcp.types import Tool

from api_agent.context import RequestContext
from api_agent.middleware import SPECIFIC_TOOL_HINT, DynamicToolNamingMiddleware
from api_agent.store import AsyncApiAgentStore, MemoryApiAgentStore, sha256_hex


@pytest.fixture(autouse=True)
def mock_downstream_description(monkeypatch):
    async def fake_description(**_kwargs):
        return "Query users, teams, and reporting data from the downstream service."

    monkeypatch.setattr("api_agent.middleware.get_downstream_description", fake_description)


def _dummy_list_context() -> MiddlewareContext[mt.ListToolsRequest]:
    """Create a dummy MiddlewareContext for on_list_tools tests."""
    return cast(MiddlewareContext[mt.ListToolsRequest], object())


def _recipe(
    tool_name: str = "list_users",
    description: str = "Use for listing users. Returns user ids as CSV. No required params. Do not use for different user fields, joins, or workflows.",
    tool_args: dict | None = None,
    steps: list | None = None,
) -> dict:
    return {
        "public_contract": {
            "tool_name": tool_name,
            "description": description,
            "tool_args": tool_args or {},
        },
        "execution_plan": {
            "steps": steps if steps is not None else [_graphql_step("users", "{ users { id } }")],
        },
        "validation_fixture": {"tool_args": {}},
    }


def _graphql_step(step_id: str, query_template: str, with_vars: dict | None = None) -> dict:
    return {
        "id": step_id,
        "kind": "graphql",
        "input": {"mode": "single", "with": with_vars or {}},
        "call": {"query_template": query_template},
        "output": {"name": step_id},
    }


class TestToolTransformation:
    """Test user-visible tool transformation."""

    @pytest.mark.asyncio
    async def test_explicit_api_name_preserves_hyphen_and_uses_underscore_separator(self):
        middleware = DynamicToolNamingMiddleware()
        req_ctx = RequestContext(
            target_url="https://api.example.com/graphql",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return [Tool(name="_query", description="Query", inputSchema={"type": "object"})]

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                with patch(
                    "api_agent.middleware.load_schema_and_base_url",
                    new_callable=AsyncMock,
                ) as mock_fetch:
                    mock_fetch.return_value = ('{"schema":"ok"}', "")
                    mock_headers.return_value = {
                        "x-target-url": req_ctx.target_url,
                        "x-api-type": req_ctx.api_type,
                        "x-api-name": "weather-alerts",
                    }
                    tools = await middleware.on_list_tools(_dummy_list_context(), call_next)

        names = [t.name for t in tools]
        assert "weather-alerts_query" in names
        query_tool = next(t for t in tools if t.name == "weather-alerts_query")
        assert query_tool.description == (
            "Query users, teams, and reporting data from the downstream service."
        )
        assert SPECIFIC_TOOL_HINT not in (query_tool.description or "")

    @pytest.mark.asyncio
    async def test_query_tool_mentions_specific_tools_when_available(self, monkeypatch):
        middleware = DynamicToolNamingMiddleware()
        store = MemoryApiAgentStore(max_size=10)
        monkeypatch.setattr("api_agent.middleware.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store))
        monkeypatch.setattr("api_agent.middleware.settings.ENABLE_RECIPES", True)

        raw_schema = '{"schema":"ok"}'
        schema_hash = sha256_hex(raw_schema)
        recipe = _recipe()
        store.save_recipe(
            api_id="graphql:https://api.example.com/graphql",
            schema_hash=schema_hash,
            question="List users",
            recipe=recipe,
            tool_name=recipe["public_contract"]["tool_name"],
        )
        req_ctx = RequestContext(
            target_url="https://api.example.com/graphql",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return [Tool(name="_query", description="Query", inputSchema={"type": "object"})]

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                with patch(
                    "api_agent.middleware.load_schema_and_base_url",
                    new_callable=AsyncMock,
                ) as mock_fetch:
                    mock_fetch.return_value = (raw_schema, "")
                    mock_headers.return_value = {
                        "x-target-url": req_ctx.target_url,
                        "x-api-type": req_ctx.api_type,
                    }
                    tools = await middleware.on_list_tools(_dummy_list_context(), call_next)

        query_tool = next(t for t in tools if t.name.endswith("_query"))
        assert SPECIFIC_TOOL_HINT in (query_tool.description or "")
        assert "recipe" not in (query_tool.description or "").lower()
        assert "r_*" not in (query_tool.description or "")

    @pytest.mark.asyncio
    async def test_list_tools_loads_schema_once_for_description_and_recipe_tools(self, monkeypatch):
        middleware = DynamicToolNamingMiddleware()
        store = MemoryApiAgentStore(max_size=10)
        monkeypatch.setattr("api_agent.middleware.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store))
        monkeypatch.setattr("api_agent.middleware.settings.ENABLE_RECIPES", True)

        raw_schema = '{"schema":"ok"}'
        schema_hash = sha256_hex(raw_schema)
        recipe = _recipe()
        store.save_recipe(
            api_id="graphql:https://api.example.com/graphql",
            schema_hash=schema_hash,
            question="List users",
            recipe=recipe,
            tool_name=recipe["public_contract"]["tool_name"],
        )
        req_ctx = RequestContext(
            target_url="https://api.example.com/graphql",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return [Tool(name="_query", description="Query", inputSchema={"type": "object"})]

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                with patch(
                    "api_agent.middleware.load_schema_and_base_url",
                    new_callable=AsyncMock,
                ) as mock_fetch:
                    mock_fetch.return_value = (raw_schema, "")
                    mock_headers.return_value = {
                        "x-target-url": req_ctx.target_url,
                        "x-api-type": req_ctx.api_type,
                    }
                    tools = await middleware.on_list_tools(_dummy_list_context(), call_next)

        assert mock_fetch.await_count == 1
        assert any(t.name.endswith("_query") for t in tools)
        assert any(t.name.startswith("r_") for t in tools)


class TestSchemaLoadGuards:
    """Test schema validation during tool listing."""

    @pytest.mark.asyncio
    async def test_list_tools_raises_when_rest_schema_missing(self):
        middleware = DynamicToolNamingMiddleware()

        req_ctx = RequestContext(
            target_url="https://api.example.com/openapi.json",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return []

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                with patch(
                    "api_agent.middleware.load_schema_and_base_url",
                    new_callable=AsyncMock,
                ) as mock_fetch:
                    mock_fetch.return_value = ("", "")
                    mock_headers.return_value = {
                        "x-target-url": req_ctx.target_url,
                        "x-api-type": req_ctx.api_type,
                    }
                    with pytest.raises(RuntimeError, match="Failed to load OpenAPI schema"):
                        await middleware.on_list_tools(_dummy_list_context(), call_next)

    @pytest.mark.asyncio
    async def test_list_tools_raises_when_graphql_schema_missing(self):
        middleware = DynamicToolNamingMiddleware()

        req_ctx = RequestContext(
            target_url="https://api.example.com/graphql",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return []

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                with patch(
                    "api_agent.middleware.load_schema_and_base_url",
                    new_callable=AsyncMock,
                ) as mock_fetch:
                    mock_fetch.return_value = ("", "")
                    mock_headers.return_value = {
                        "x-target-url": req_ctx.target_url,
                        "x-api-type": req_ctx.api_type,
                    }
                    with pytest.raises(RuntimeError, match="Failed to load GraphQL schema"):
                        await middleware.on_list_tools(_dummy_list_context(), call_next)


class TestRecipeToolListing:
    """Test recipe tool exposure and naming."""

    @pytest.mark.asyncio
    async def test_recipe_tool_uses_slug_without_api_prefix(self, monkeypatch):
        middleware = DynamicToolNamingMiddleware()
        store = MemoryApiAgentStore(max_size=10)
        monkeypatch.setattr("api_agent.middleware.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store))
        monkeypatch.setattr("api_agent.middleware.settings.ENABLE_RECIPES", True)

        raw_schema = '{"schema":"ok"}'
        api_id = "graphql:https://api.example.com/graphql"
        schema_hash = sha256_hex(raw_schema)
        recipe = _recipe(
            tool_name="list_users_reporting_to_manager",
            description="Use for listing users who report to one manager. Returns user names as CSV. Requires manager_name. Do not use for different reporting chains, joins, or extra fields.",
            tool_args={"manager_name": {"type": "str", "description": "Manager name"}},
            steps=[
                _graphql_step(
                    "users",
                    '{ users(manager: "{{manager_name}}") { name } }',
                    {"manager_name": {"value": "manager_name"}},
                )
            ],
        )
        store.save_recipe(
            api_id=api_id,
            schema_hash=schema_hash,
            question="List users reporting to manager",
            recipe=recipe,
            tool_name=recipe["public_contract"]["tool_name"],
        )

        req_ctx = RequestContext(
            target_url="https://api.example.com/graphql",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return [Tool(name="_query", description="Query", inputSchema={"type": "object"})]

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                with patch(
                    "api_agent.middleware.load_schema_and_base_url",
                    new_callable=AsyncMock,
                ) as mock_fetch:
                    mock_fetch.return_value = (raw_schema, "")
                    mock_headers.return_value = {
                        "x-target-url": req_ctx.target_url,
                        "x-api-type": req_ctx.api_type,
                    }
                    tools = await middleware.on_list_tools(_dummy_list_context(), call_next)

        names = [t.name for t in tools]
        assert "r_list_users_reporting_to_manager" in names
        assert all(len(name) < 50 for name in names)
        tool = next(t for t in tools if t.name == "r_list_users_reporting_to_manager")
        assert tool.description == recipe["public_contract"]["description"]
        assert "[" not in (tool.description or "")
        assert "Required params:" not in (tool.description or "")
        assert "Executes:" not in (tool.description or "")
        schema = tool.model_dump().get("parameters", {})
        assert "manager_name" in schema.get("properties", {})
        assert schema["properties"]["manager_name"]["description"] == "Manager name"

    @pytest.mark.asyncio
    async def test_recipe_tool_name_is_truncated_below_50_including_prefix(self, monkeypatch):
        middleware = DynamicToolNamingMiddleware()
        store = MemoryApiAgentStore(max_size=10)
        monkeypatch.setattr("api_agent.middleware.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store))
        monkeypatch.setattr("api_agent.middleware.settings.ENABLE_RECIPES", True)

        raw_schema = '{"schema":"ok"}'
        api_id = "graphql:https://api.example.com/graphql"
        schema_hash = sha256_hex(raw_schema)
        recipe = _recipe(
            tool_name="list_users_with_extremely_long_recipe_name_that_needs_truncation_for_limits",
            tool_args={"manager_name": {"type": "str", "description": "Manager name"}},
            steps=[
                _graphql_step(
                    "users",
                    '{ users(manager: "{{manager_name}}") { name } }',
                    {"manager_name": {"value": "manager_name"}},
                )
            ],
        )
        store.save_recipe(
            api_id=api_id,
            schema_hash=schema_hash,
            question="List users reporting to manager",
            recipe=recipe,
            tool_name=recipe["public_contract"]["tool_name"],
        )

        req_ctx = RequestContext(
            target_url="https://api.example.com/graphql",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return [Tool(name="_query", description="Query", inputSchema={"type": "object"})]

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                with patch(
                    "api_agent.middleware.load_schema_and_base_url",
                    new_callable=AsyncMock,
                ) as mock_fetch:
                    mock_fetch.return_value = (raw_schema, "")
                    mock_headers.return_value = {
                        "x-target-url": req_ctx.target_url,
                        "x-api-type": req_ctx.api_type,
                    }
                    tools = await middleware.on_list_tools(_dummy_list_context(), call_next)

        recipe_tools = [t for t in tools if t.name.startswith("r_")]
        assert recipe_tools, "expected at least one recipe tool"
        assert all(len(t.name) < 50 for t in recipe_tools)


class TestRecipeToolSchema:
    """Test recipe tool input schema."""

    @pytest.mark.asyncio
    async def test_all_tool_args_required(self, monkeypatch):
        """All public tool args must be explicitly provided."""
        middleware = DynamicToolNamingMiddleware()
        store = MemoryApiAgentStore(max_size=10)
        monkeypatch.setattr("api_agent.middleware.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store))
        monkeypatch.setattr("api_agent.middleware.settings.ENABLE_RECIPES", True)

        raw_schema = '{"schema":"ok"}'
        api_id = "graphql:https://api.example.com/graphql"
        schema_hash = sha256_hex(raw_schema)
        recipe = _recipe(
            tool_name="list_users",
            description="Use for listing users by id and active state. Returns user ids as CSV. Requires user_id and active. Do not use for different user fields, joins, or workflows.",
            tool_args={
                "user_id": {"type": "int", "description": "User id"},
                "active": {"type": "bool", "description": "Active flag"},
            },
            steps=[
                _graphql_step(
                    "users",
                    "{ users(id: {{user_id}}, active: {{active}}) { id } }",
                    {"user_id": {"value": "user_id"}, "active": {"value": "active"}},
                )
            ],
        )
        store.save_recipe(
            api_id=api_id,
            schema_hash=schema_hash,
            question="List users",
            recipe=recipe,
            tool_name=recipe["public_contract"]["tool_name"],
        )

        req_ctx = RequestContext(
            target_url="https://api.example.com/graphql",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return []

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                with patch(
                    "api_agent.middleware.load_schema_and_base_url",
                    new_callable=AsyncMock,
                ) as mock_fetch:
                    mock_fetch.return_value = (raw_schema, "")
                    mock_headers.return_value = {
                        "x-target-url": req_ctx.target_url,
                        "x-api-type": req_ctx.api_type,
                    }
                    tools = await middleware.on_list_tools(_dummy_list_context(), call_next)

        tool = next(t for t in tools if t.name == "r_list_users")
        schema = tool.model_dump().get("parameters", {})
        # Tool args are top-level, not nested under "params"
        assert "params" not in schema.get("properties", {})
        assert sorted(schema.get("required", [])) == ["active", "user_id"]
        assert "default" not in schema["properties"]["user_id"]
        assert "default" not in schema["properties"]["active"]
        assert "return_directly" not in schema["properties"]

    @pytest.mark.asyncio
    async def test_tool_arg_descriptions_from_public_contract(self, monkeypatch):
        """Tool arg descriptions come from public_contract.tool_args."""
        middleware = DynamicToolNamingMiddleware()
        store = MemoryApiAgentStore(max_size=10)
        monkeypatch.setattr("api_agent.middleware.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store))
        monkeypatch.setattr("api_agent.middleware.settings.ENABLE_RECIPES", True)

        raw_schema = '{"schema":"ok"}'
        api_id = "graphql:https://api.example.com/graphql"
        schema_hash = sha256_hex(raw_schema)
        recipe = _recipe(
            tool_name="list_users",
            description="Use for listing users by id and active state. Returns user ids as CSV. Requires user_id and active. Do not use for different user fields, joins, or workflows.",
            tool_args={
                "user_id": {"type": "int", "description": "User id"},
                "active": {"type": "bool", "description": "Active flag"},
            },
            steps=[
                _graphql_step(
                    "users",
                    "{ users(id: {{user_id}}, active: {{active}}) { id } }",
                    {"user_id": {"value": "user_id"}, "active": {"value": "active"}},
                )
            ],
        )
        store.save_recipe(
            api_id=api_id,
            schema_hash=schema_hash,
            question="List users",
            recipe=recipe,
            tool_name=recipe["public_contract"]["tool_name"],
        )

        req_ctx = RequestContext(
            target_url="https://api.example.com/graphql",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return []

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                with patch(
                    "api_agent.middleware.load_schema_and_base_url",
                    new_callable=AsyncMock,
                ) as mock_fetch:
                    mock_fetch.return_value = (raw_schema, "")
                    mock_headers.return_value = {
                        "x-target-url": req_ctx.target_url,
                        "x-api-type": req_ctx.api_type,
                    }
                    tools = await middleware.on_list_tools(_dummy_list_context(), call_next)

        tool = next(t for t in tools if t.name == "r_list_users")
        schema = tool.model_dump().get("parameters", {})
        # Tool args are top-level.
        assert "params" not in schema.get("properties", {})
        assert sorted(schema.get("required", [])) == ["active", "user_id"]
        uid_props = schema["properties"]["user_id"]
        active_props = schema["properties"]["active"]
        assert uid_props.get("description") == "User id"
        assert active_props.get("description") == "Active flag"


class TestRecipeToolErrors:
    """Ensure recipe tools signal errors via MCP exceptions."""

    @pytest.mark.asyncio
    async def test_unknown_tool_is_not_bypassed(self):
        middleware = DynamicToolNamingMiddleware()

        async def call_next(_context):
            return None

        message = mt.CallToolRequestParams(name="other_tool", arguments={})
        context = MiddlewareContext(message=message)

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            mock_headers.return_value = {
                "x-target-url": "https://api.example.com/graphql",
                "x-api-type": "graphql",
            }
            with pytest.raises(NotFoundError, match="Expected tool name starting"):
                await middleware.on_call_tool(context, call_next)

    @pytest.mark.asyncio
    async def test_recipe_tool_invalid_arguments_raises_validation_error(self):
        middleware = DynamicToolNamingMiddleware()
        req_ctx = RequestContext(
            target_url="https://api.example.com/graphql",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return None

        # Non-dict arguments should raise ValidationError
        message = mt.CallToolRequestParams.model_construct(name="r_test", arguments="not_a_dict")
        context = MiddlewareContext(message=message)

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                mock_headers.return_value = {
                    "x-target-url": req_ctx.target_url,
                    "x-api-type": req_ctx.api_type,
                }
                with pytest.raises(ValidationError, match="Invalid arguments"):
                    await middleware.on_call_tool(context, call_next)

    @pytest.mark.asyncio
    async def test_recipe_tool_invalid_param_type_raises_validation_error(self, monkeypatch):
        middleware = DynamicToolNamingMiddleware()
        store = MemoryApiAgentStore(max_size=10)
        monkeypatch.setattr("api_agent.middleware.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store))
        monkeypatch.setattr(
            "api_agent.recipe.runner.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store)
        )

        raw_schema = '{"schema":"ok"}'
        api_id = "graphql:https://api.example.com/graphql"
        schema_hash = sha256_hex(raw_schema)
        recipe = _recipe(
            tool_name="list_users",
            tool_args={"limit": {"type": "int", "description": "Limit"}},
            steps=[
                _graphql_step(
                    "users",
                    "{ users(limit: {{limit}}) { id } }",
                    {"limit": {"value": "limit"}},
                )
            ],
        )
        store.save_recipe(
            api_id=api_id,
            schema_hash=schema_hash,
            question="List users",
            recipe=recipe,
            tool_name=recipe["public_contract"]["tool_name"],
        )
        req_ctx = RequestContext(
            target_url="https://api.example.com/graphql",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return None

        message = mt.CallToolRequestParams(name="r_list_users", arguments={"limit": "ten"})
        context = MiddlewareContext(message=message)

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                with patch(
                    "api_agent.middleware.load_schema_and_base_url",
                    new_callable=AsyncMock,
                ) as mock_fetch:
                    mock_fetch.return_value = (raw_schema, "")
                    mock_headers.return_value = {
                        "x-target-url": req_ctx.target_url,
                        "x-api-type": req_ctx.api_type,
                    }
                    with pytest.raises(ValidationError, match="invalid param type: limit"):
                        await middleware.on_call_tool(context, call_next)

    @pytest.mark.asyncio
    async def test_recipe_tool_not_found_raises_not_found(self):
        middleware = DynamicToolNamingMiddleware()
        req_ctx = RequestContext(
            target_url="https://api.example.com/graphql",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return None

        message = mt.CallToolRequestParams(name="r_missing", arguments={})
        context = MiddlewareContext(message=message)

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                with patch(
                    "api_agent.middleware.load_schema_and_base_url",
                    new_callable=AsyncMock,
                ) as mock_fetch:
                    mock_fetch.return_value = ('{"schema":"ok"}', "")
                    mock_headers.return_value = {
                        "x-target-url": req_ctx.target_url,
                        "x-api-type": req_ctx.api_type,
                    }
                    with patch(
                        "api_agent.middleware.ASYNC_API_AGENT_STORE.find_recipe_by_tool_slug",
                        return_value=None,
                    ):
                        with pytest.raises(NotFoundError, match="recipe not found"):
                            await middleware.on_call_tool(context, call_next)


class TestCallToolNameValidation:
    """Validate explicit vs implicit tool name prefixes."""

    @pytest.mark.asyncio
    async def test_explicit_api_name_with_hyphen_accepts_underscore_separator(self):
        middleware = DynamicToolNamingMiddleware()

        async def call_next(context):
            return context.message.name

        message = mt.CallToolRequestParams(name="weather-alerts_query", arguments={})
        context = MiddlewareContext(message=message)

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            mock_headers.return_value = {
                "x-api-name": "weather-alerts",
                "x-target-url": "https://api.example.com/graphql",
                "x-api-type": "graphql",
            }
            internal = await middleware.on_call_tool(context, call_next)
            assert internal == "_query"
