import pytest

from api_agent.context import RequestContext
from api_agent.recipe.runner import execute_recipe_tool
from api_agent.recipe.search import build_api_id
from api_agent.store import sha256_hex
from api_agent.utils.csv import to_csv


def _contract_recipe() -> dict:
    return {
        "public_contract": {
            "tool_name": "get_users",
            "description": "Use for listing users. Returns user ids as CSV. No required params. Do not use for different user fields, joins, or workflows.",
            "tool_args": {},
        },
        "execution_plan": {
            "steps": [
                {
                    "id": "users",
                    "kind": "graphql",
                    "input": {"mode": "single", "with": {}},
                    "call": {"query_template": "{ users { id } }"},
                    "output": {"name": "users"},
                }
            ],
        },
        "validation_fixture": {"tool_args": {}},
    }


@pytest.mark.asyncio
async def test_execute_recipe_tool_return_directly_graphql(monkeypatch):
    ctx = RequestContext(
        target_url="https://example.com/graphql",
        target_headers={},
        api_type="graphql",
        base_url="",
        include_result=False,
        allow_unsafe_paths=(),
        poll_paths=(),
    )

    raw_schema = "schema"
    schema_hash = sha256_hex(raw_schema)
    api_id = build_api_id(ctx, ctx.api_type, "")

    async def fake_load_schema_and_base_url(_ctx):
        return raw_schema, ""

    async def fake_execute_recipe_steps(
        recipe,
        params,
        query_results_var,
        last_result_var,
        api_step_executor,
        executed_items_list,
    ):
        executed_items_list.append("q1")
        return True, {"ok": 1}, ["select 1"], ""

    async def fake_get_recipe_meta(_recipe_id):
        return {
            "schema_hash": schema_hash,
            "api_id": api_id,
            "recipe": _contract_recipe(),
        }

    monkeypatch.setattr(
        "api_agent.recipe.runner.load_schema_and_base_url", fake_load_schema_and_base_url
    )
    monkeypatch.setattr("api_agent.recipe.runner.execute_recipe_steps", fake_execute_recipe_steps)
    monkeypatch.setattr(
        "api_agent.recipe.runner.ASYNC_API_AGENT_STORE.get_recipe_meta",
        fake_get_recipe_meta,
    )

    result = await execute_recipe_tool(ctx, "r_test", params=None, return_directly=True)
    assert result == to_csv({"ok": 1})


@pytest.mark.asyncio
async def test_execute_recipe_tool_return_directly_rest(monkeypatch):
    ctx = RequestContext(
        target_url="https://example.com/rest",
        target_headers={},
        api_type="rest",
        base_url="/v1",
        include_result=False,
        allow_unsafe_paths=(),
        poll_paths=(),
    )

    raw_schema = "schema"
    schema_hash = sha256_hex(raw_schema)
    api_id = build_api_id(ctx, ctx.api_type, ctx.base_url or "")

    async def fake_load_schema_and_base_url(_ctx):
        return raw_schema, "/v1"

    async def fake_execute_recipe_steps(
        recipe,
        params,
        query_results_var,
        last_result_var,
        api_step_executor,
        executed_items_list,
    ):
        executed_items_list.append({"method": "GET", "path": "/users/1"})
        return True, {"id": 1}, [], ""

    async def fake_get_recipe_meta(_recipe_id):
        return {
            "schema_hash": schema_hash,
            "api_id": api_id,
            "recipe": _contract_recipe(),
        }

    monkeypatch.setattr(
        "api_agent.recipe.runner.load_schema_and_base_url", fake_load_schema_and_base_url
    )
    monkeypatch.setattr("api_agent.recipe.runner.execute_recipe_steps", fake_execute_recipe_steps)
    monkeypatch.setattr(
        "api_agent.recipe.runner.ASYNC_API_AGENT_STORE.get_recipe_meta",
        fake_get_recipe_meta,
    )

    result = await execute_recipe_tool(ctx, "r_test", params=None, return_directly=True)
    assert result == to_csv({"id": 1})
