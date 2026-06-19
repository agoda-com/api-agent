"""Tests for recipe extraction and deduplication."""

import json
import logging
from copy import deepcopy
from types import SimpleNamespace

import pytest
from agents import AgentOutputSchema
from agents.exceptions import ModelBehaviorError

from api_agent.executor import execute_sql
from api_agent.recipe.contracts import resolve_step_input_values
from api_agent.recipe.extractor import extract_recipe
from api_agent.recipe.extractor_models import ExtractedRecipeOutput
from api_agent.recipe.extractor_prompts import build_extractor_instructions
from api_agent.recipe.learning import maybe_extract_and_save_recipe
from api_agent.store import AsyncApiAgentStore, MemoryApiAgentStore, sha256_hex


def _graphql_recipe() -> dict:
    return {
        "public_contract": {
            "tool_name": "list_users",
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


def _graphql_step(step_id: str, query_template: str, with_vars: dict | None = None) -> dict:
    return {
        "id": step_id,
        "kind": "graphql",
        "input": {"mode": "single", "with": with_vars or {}},
        "call": {"query_template": query_template},
        "output": {"name": step_id},
    }


def _sql_step(step_id: str, query_template: str, with_vars: dict | None = None) -> dict:
    return {
        "id": step_id,
        "kind": "sql",
        "input": {"mode": "single", "with": with_vars or {}},
        "query_template": query_template,
        "output": {"name": step_id},
    }


def _rest_step(
    step_id: str,
    path: str,
    *,
    input_spec: dict | None = None,
    path_params: dict | None = None,
    query_params: dict | None = None,
    body: dict | None = None,
    output: dict | None = None,
    method: str = "GET",
) -> dict:
    return {
        "id": step_id,
        "kind": "rest",
        "input": input_spec or {"mode": "single", "with": {}},
        "call": {
            "method": method,
            "path": path,
            "path_params": path_params or {},
            "query_params": query_params or {},
            "body": body or {},
        },
        "output": output or {"name": step_id},
    }


async def _async_result(value):
    return value


def test_build_extractor_instructions_uses_graphql_section():
    instructions = build_extractor_instructions("graphql")

    assert "GRAPHQL RECIPE RULES" in instructions
    assert "GRAPHQL MAP EXAMPLE" in instructions
    assert "REST RECIPE RULES" not in instructions
    assert '"kind": "graphql"' in instructions
    assert '"kind": "rest"' not in instructions


def test_build_extractor_instructions_uses_rest_section():
    instructions = build_extractor_instructions("rest")

    assert "REST RECIPE RULES" in instructions
    assert "REST MAP EXAMPLE" in instructions
    assert "GRAPHQL RECIPE RULES" not in instructions
    assert '"kind": "rest"' in instructions


def test_contains_pattern_matches_separator_variants():
    step = _sql_step(
        "filtered_objectives",
        "SELECT id FROM objectives WHERE lower(assignee.team.name) LIKE '{{team_pattern}}'",
        {"team_pattern": {"value": "team", "transform": "contains_pattern"}},
    )
    input_sets, error = resolve_step_input_values(step, {"team": "alpha-suite"}, {})
    assert error == ""
    assert input_sets is not None

    sql = step["query_template"].replace("{{team_pattern}}", input_sets[0][0]["team_pattern"])
    result = execute_sql(
        {"objectives": [{"id": 1, "assignee": {"team": {"name": "Alpha Suite"}}}]},
        sql,
    )

    assert result["result"] == [{"id": 1}]


def test_contains_pattern_splits_underscores():
    step = _sql_step(
        "filtered_objectives",
        "SELECT id FROM objectives WHERE lower(assignee.team.name) LIKE '{{team_pattern}}'",
        {"team_pattern": {"value": "team", "transform": "contains_pattern"}},
    )
    input_sets, error = resolve_step_input_values(step, {"team": "alpha_suite"}, {})
    assert error == ""
    assert input_sets is not None

    sql = step["query_template"].replace("{{team_pattern}}", input_sets[0][0]["team_pattern"])
    result = execute_sql(
        {"objectives": [{"id": 1, "assignee": {"team": {"name": "Alpha Suite"}}}]},
        sql,
    )

    assert result["result"] == [{"id": 1}]


@pytest.mark.asyncio
async def test_extract_recipe_relaxes_public_string_sql_equality(monkeypatch):
    recipe = {
        "public_contract": {
            "tool_name": "list_team_okrs",
            "description": "Use for listing OKRs for a specific team in a specific cycle. Returns objectives with their key results, owners, and status as rows. Requires team and cycle. Do not use for different fields, joins, or workflow.",
            "tool_args": {
                "team": {"type": "str", "description": "Team name"},
                "cycle": {"type": "str", "description": "Cycle"},
            },
        },
        "execution_plan": {
            "steps": [
                _rest_step(
                    "objectives",
                    "/api/v2/objectives",
                    input_spec={"mode": "single", "with": {"cycle": {"value": "cycle"}}},
                    query_params={"cycle": {"$var": "cycle"}},
                ),
                _sql_step(
                    "filtered_objectives",
                    (
                        "SELECT id FROM objectives "
                        "WHERE lower(assignee.team.name) = lower('{{team_name}}')"
                    ),
                    {"team_name": {"value": "team"}},
                ),
            ],
        },
        "validation_fixture": {"tool_args": {"team": "alpha-suite", "cycle": "Y2026Q2"}},
    }

    async def run_agent(_agent, _input, max_turns, run_config):
        return SimpleNamespace(final_output=ExtractedRecipeOutput.model_validate(recipe))

    monkeypatch.setattr("api_agent.recipe.extractor.Runner.run", run_agent)

    result = await extract_recipe(
        api_type="rest",
        question="List alpha-suite OKRs for Q2 2026",
        steps=[],
    )

    assert result is not None
    sql_step = result["execution_plan"]["steps"][1]
    assert "LIKE lower('{{team_name}}')" in sql_step["query_template"]
    assert sql_step["input"]["with"]["team_name"]["transform"] == "contains_pattern"


@pytest.mark.asyncio
async def test_extract_recipe_keeps_canonical_string_sql_equality(monkeypatch):
    recipe = {
        "public_contract": {
            "tool_name": "list_team_okrs",
            "description": "Use for listing OKRs for a specific team in a specific cycle. Returns objective rows for that team. Requires team and cycle. Do not use for different fields, joins, or workflow.",
            "tool_args": {
                "team": {"type": "str", "description": "Team name"},
                "cycle": {"type": "str", "description": "Cycle"},
            },
        },
        "execution_plan": {
            "steps": [
                _rest_step(
                    "objectives",
                    "/api/v2/objectives",
                    input_spec={"mode": "single", "with": {"cycle": {"value": "cycle"}}},
                    query_params={"cycle": {"$var": "cycle"}},
                ),
                _sql_step(
                    "filtered_objectives",
                    (
                        "SELECT id FROM objectives "
                        "WHERE lower(assignee.team.name) = lower('{{team_name}}')"
                    ),
                    {"team_name": {"value": "team"}},
                ),
            ],
        },
        "validation_fixture": {"tool_args": {"team": "Alpha Suite", "cycle": "Y2026Q2"}},
    }

    async def run_agent(_agent, _input, max_turns, run_config):
        return SimpleNamespace(final_output=ExtractedRecipeOutput.model_validate(recipe))

    monkeypatch.setattr("api_agent.recipe.extractor.Runner.run", run_agent)

    result = await extract_recipe(
        api_type="rest",
        question="List Alpha Suite OKRs for Q2 2026",
        steps=[],
    )

    assert result is not None
    sql_step = result["execution_plan"]["steps"][1]
    assert " = lower('{{team_name}}')" in sql_step["query_template"]
    assert "transform" not in sql_step["input"]["with"]["team_name"]


@pytest.mark.asyncio
async def test_extract_recipe_accepts_structured_output(monkeypatch):
    recipe = _graphql_recipe()

    async def run_agent(agent, _input, max_turns, run_config):
        assert max_turns == 2
        assert isinstance(agent.output_type, AgentOutputSchema)
        assert agent.output_type.output_type is ExtractedRecipeOutput
        assert not agent.output_type.is_strict_json_schema()
        assert "GRAPHQL RECIPE RULES" in agent.instructions
        assert "REST RECIPE RULES" not in agent.instructions
        return SimpleNamespace(final_output=ExtractedRecipeOutput.model_validate(recipe))

    monkeypatch.setattr("api_agent.recipe.extractor.Runner.run", run_agent)

    result = await extract_recipe(
        api_type="graphql",
        question="List users",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { id } }"}],
    )

    assert result == recipe


@pytest.mark.asyncio
async def test_extract_recipe_passes_validation_feedback(monkeypatch):
    recipe = _graphql_recipe()
    feedback = {
        "reason": "candidate_result_mismatch",
        "candidate_result": {"row_count": 0, "sample": []},
        "expected_result": {"row_count": 1, "sample": [{"id": 1}]},
    }

    async def run_agent(_agent, input_payload, max_turns, run_config):
        assert max_turns == 2
        payload = json.loads(input_payload)
        assert payload["validation_feedback"] == feedback
        return SimpleNamespace(final_output=ExtractedRecipeOutput.model_validate(recipe))

    monkeypatch.setattr("api_agent.recipe.extractor.Runner.run", run_agent)

    result = await extract_recipe(
        api_type="graphql",
        question="List users",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { id } }"}],
        validation_feedback=feedback,
    )

    assert result == recipe


@pytest.mark.asyncio
async def test_extract_recipe_accepts_interleaved_sql_steps(monkeypatch):
    recipe: dict = {
        "public_contract": {
            "tool_name": "get_user_order_total",
            "description": "Use for calculating total order amount for one user. Returns summed order amount as CSV. Requires userId. Do not use for unrelated order fields or joins.",
            "tool_args": {"userId": {"type": "int", "description": "User id"}},
        },
        "execution_plan": {
            "steps": [
                _graphql_step(
                    "orders",
                    "{ orders(userId: {{userId}}) { id amount } }",
                    {"userId": {"value": "userId"}},
                ),
                _sql_step("order_total", "SELECT SUM(amount) AS total FROM orders"),
            ],
        },
        "validation_fixture": {"tool_args": {"userId": 42}},
    }

    async def run_agent(_agent, _input, max_turns, run_config):
        return SimpleNamespace(final_output=ExtractedRecipeOutput.model_validate(recipe))

    monkeypatch.setattr("api_agent.recipe.extractor.Runner.run", run_agent)

    result = await extract_recipe(
        api_type="graphql",
        question="Total orders for user 42",
        steps=[
            {
                "kind": "graphql",
                "name": "orders",
                "query": "{ orders(userId: 42) { id amount } }",
            },
            {
                "kind": "sql",
                "query": "SELECT SUM(amount) AS total FROM orders",
            },
        ],
    )

    assert result == recipe
    assert result is not None
    assert "sql_steps" not in result
    assert [step["kind"] for step in result["execution_plan"]["steps"]] == ["graphql", "sql"]


@pytest.mark.asyncio
async def test_extract_recipe_rejects_invalid_structured_output(monkeypatch):
    async def run_agent(_agent, _input, max_turns, run_config):
        return SimpleNamespace(final_output={"tool_name": "list_users"})

    monkeypatch.setattr("api_agent.recipe.extractor.Runner.run", run_agent)

    result = await extract_recipe(
        api_type="graphql",
        question="List users",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { id } }"}],
    )

    assert result is None


@pytest.mark.asyncio
async def test_extract_recipe_rejects_model_output_validation_error(monkeypatch):
    async def run_agent(_agent, _input, max_turns, run_config):
        raise ModelBehaviorError("invalid final output")

    monkeypatch.setattr("api_agent.recipe.extractor.Runner.run", run_agent)

    result = await extract_recipe(
        api_type="graphql",
        question="List users",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { id } }"}],
    )

    assert result is None


@pytest.mark.asyncio
async def test_extract_recipe_rejects_missing_validation_fixture_for_public_args(monkeypatch):
    recipe = _graphql_recipe()
    recipe["public_contract"]["tool_args"] = {"userId": {"type": "int", "description": "User id"}}
    recipe["execution_plan"]["steps"][0]["call"]["query_template"] = (
        "{ users(id: {{userId}}) { id } }"
    )
    recipe["execution_plan"]["steps"][0]["input"]["with"] = {"userId": {"value": "userId"}}
    del recipe["validation_fixture"]

    async def run_agent(_agent, _input, max_turns, run_config):
        return SimpleNamespace(final_output=recipe)

    monkeypatch.setattr("api_agent.recipe.extractor.Runner.run", run_agent)

    result = await extract_recipe(
        api_type="graphql",
        question="List user 1",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { id } }"}],
    )

    assert result is None


@pytest.mark.asyncio
async def test_extract_recipe_rejects_reserved_tool_prefix(monkeypatch):
    recipe = _graphql_recipe()
    recipe["public_contract"]["tool_name"] = "r_list_users"

    async def run_agent(_agent, _input, max_turns, run_config):
        return SimpleNamespace(final_output=ExtractedRecipeOutput.model_validate(recipe))

    monkeypatch.setattr("api_agent.recipe.extractor.Runner.run", run_agent)

    result = await extract_recipe(
        api_type="graphql",
        question="List users",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { id } }"}],
    )

    assert result is None


@pytest.mark.asyncio
async def test_extract_recipe_rejects_unknown_template_var(monkeypatch):
    recipe = _graphql_recipe()
    recipe["execution_plan"]["steps"][0]["call"]["query_template"] = (
        "{ users(id: {{userId}}) { id } }"
    )

    async def run_agent(_agent, _input, max_turns, run_config):
        return SimpleNamespace(final_output=ExtractedRecipeOutput.model_validate(recipe))

    monkeypatch.setattr("api_agent.recipe.extractor.Runner.run", run_agent)

    result = await extract_recipe(
        api_type="graphql",
        question="List users",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { id } }"}],
    )

    assert result is None


@pytest.mark.asyncio
async def test_extract_recipe_rejects_missing_input_arg(monkeypatch):
    recipe = _graphql_recipe()
    recipe["public_contract"]["tool_args"] = {"userId": {"type": "int", "description": "User id"}}
    recipe["execution_plan"]["steps"][0]["call"]["query_template"] = (
        "{ users(id: {{userId}}) { id } }"
    )
    recipe["execution_plan"]["steps"][0]["input"]["with"] = {"userId": {"value": "missing"}}
    recipe["validation_fixture"] = {"tool_args": {"userId": 1}}

    async def run_agent(_agent, _input, max_turns, run_config):
        return SimpleNamespace(final_output=ExtractedRecipeOutput.model_validate(recipe))

    monkeypatch.setattr("api_agent.recipe.extractor.Runner.run", run_agent)

    result = await extract_recipe(
        api_type="graphql",
        question="List user 1",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { id } }"}],
    )

    assert result is None


@pytest.mark.asyncio
async def test_extract_recipe_rejects_hidden_step_result_rest_param(monkeypatch):
    recipe = {
        "public_contract": {
            "tool_name": "get_team_okrs",
            "description": "Use for looking up team OKRs by team and cycle. Returns objective and key result rows as CSV. Requires team_id and cycle. Do not use for different fields, joins, or workflows.",
            "tool_args": {
                "team_id": {"type": "int", "description": "Team id"},
                "cycle": {"type": "str", "description": "Cycle"},
            },
        },
        "execution_plan": {"steps": [{"kind": "rest", "name": "objectives_by_cycle"}]},
        "validation_fixture": {"tool_args": {"team_id": 3, "cycle": "Y2026Q2"}},
    }

    async def run_agent(_agent, _input, max_turns, run_config):
        return SimpleNamespace(final_output=recipe)

    monkeypatch.setattr("api_agent.recipe.extractor.Runner.run", run_agent)

    result = await extract_recipe(
        api_type="rest",
        question="Team 3 OKRs for Y2026Q2",
        steps=[{"kind": "rest", "method": "GET", "path": "/objectives"}],
    )

    assert result is None


@pytest.mark.asyncio
async def test_extract_recipe_accepts_mapped_binding_source(monkeypatch):
    recipe = {
        "public_contract": {
            "tool_name": "get_team_okrs",
            "description": "Use for looking up team OKRs by team and cycle. Returns objective and key result rows as CSV. Requires team_id and cycle. Do not use for different fields, joins, or workflows.",
            "tool_args": {
                "team_id": {"type": "int", "description": "Team id"},
                "cycle": {"type": "str", "description": "Cycle"},
            },
        },
        "execution_plan": {
            "steps": [
                _rest_step(
                    "objectives_by_cycle",
                    "/objectives",
                    input_spec={"mode": "single", "with": {"cycle": {"value": "cycle"}}},
                    query_params={"cycle": {"$var": "cycle"}},
                ),
                _sql_step(
                    "filtered_objectives",
                    (
                        "SELECT id FROM objectives_by_cycle "
                        "WHERE assignee.team.id = {{team_id}} ORDER BY id"
                    ),
                    {"team_id": {"value": "team_id"}},
                ),
                _rest_step(
                    "key_results_for_objective",
                    "/objectives/{objective_id}/key-results",
                    input_spec={
                        "mode": "map",
                        "from": "filtered_objectives",
                        "bind": {"objective_id": "id"},
                        "with": {},
                    },
                    path_params={"objective_id": {"$var": "objective_id"}},
                    output={
                        "name": "key_results_for_objective",
                        "attach_binding": ["objective_id"],
                    },
                ),
            ],
        },
        "validation_fixture": {"tool_args": {"team_id": 3, "cycle": "Y2026Q2"}},
    }

    async def run_agent(_agent, _input, max_turns, run_config):
        return SimpleNamespace(final_output=ExtractedRecipeOutput.model_validate(recipe))

    monkeypatch.setattr("api_agent.recipe.extractor.Runner.run", run_agent)

    result = await extract_recipe(
        api_type="rest",
        question="Team 3 OKRs for Y2026Q2",
        steps=[{"kind": "rest", "method": "GET", "path": "/objectives"}],
    )

    assert result == recipe


@pytest.mark.asyncio
async def test_extract_recipe_rejects_unsupported_transform(monkeypatch):
    recipe = _graphql_recipe()
    recipe["public_contract"]["tool_args"] = {"team": {"type": "str", "description": "Team"}}
    recipe["execution_plan"]["steps"][0]["input"]["with"] = {
        "teamPattern": {"value": "team", "transform": "regex"}
    }
    recipe["execution_plan"]["steps"][0]["call"]["query_template"] = (
        '{ users(team: "{{teamPattern}}") { id } }'
    )
    recipe["validation_fixture"] = {"tool_args": {"team": "alpha-suite"}}

    async def run_agent(_agent, _input, max_turns, run_config):
        return SimpleNamespace(final_output=ExtractedRecipeOutput.model_validate(recipe))

    monkeypatch.setattr("api_agent.recipe.extractor.Runner.run", run_agent)

    result = await extract_recipe(
        api_type="graphql",
        question="List alpha-suite users",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { id } }"}],
    )

    assert result is None


@pytest.mark.asyncio
async def test_extract_recipe_rejects_rest_write_candidate(monkeypatch):
    recipe = {
        "public_contract": {
            "tool_name": "create_user",
            "description": "Use for creating a user by name. Returns created user rows as CSV. Requires name. Do not use for different fields, joins, or workflows.",
            "tool_args": {"name": {"type": "str", "description": "Name"}},
        },
        "execution_plan": {
            "steps": [
                _rest_step(
                    "users",
                    "/users",
                    input_spec={"mode": "single", "with": {"name": {"value": "name"}}},
                    body={"name": {"$var": "name"}},
                    method="POST",
                )
            ],
        },
        "validation_fixture": {"tool_args": {"name": "Ada"}},
    }

    async def run_agent(_agent, _input, max_turns, run_config):
        return SimpleNamespace(final_output=ExtractedRecipeOutput.model_validate(recipe))

    monkeypatch.setattr("api_agent.recipe.extractor.Runner.run", run_agent)

    result = await extract_recipe(
        api_type="rest",
        question="Create Ada",
        steps=[{"kind": "rest", "method": "POST", "path": "/users"}],
    )

    assert result is None


@pytest.mark.asyncio
async def test_skip_duplicate_recipe(monkeypatch):
    store = MemoryApiAgentStore(max_size=10)
    monkeypatch.setattr(
        "api_agent.recipe.learning.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store)
    )
    monkeypatch.setattr("api_agent.recipe.learning.settings.ENABLE_RECIPES", True)

    raw_schema = '{"schema":"ok"}'
    api_id = "graphql:https://api.example.com/graphql"
    schema_hash = sha256_hex(raw_schema)
    recipe = {
        **_graphql_recipe(),
    }
    store.save_recipe(
        api_id=api_id,
        schema_hash=schema_hash,
        question="List users",
        recipe=recipe,
        tool_name=recipe["public_contract"]["tool_name"],
    )

    async def fake_extract_recipe(**_kwargs):
        return dict(recipe)

    monkeypatch.setattr("api_agent.recipe.learning.extract_recipe", fake_extract_recipe)

    await maybe_extract_and_save_recipe(
        api_type="graphql",
        api_id=api_id,
        question="List users",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { id } }"}],
        raw_schema=raw_schema,
        learn_rate=1,
        original_result=[{"id": 1}],
        validate_candidate=lambda _recipe, _args: _async_result([{"id": 1}]),
    )

    assert len(store.list_recipes(api_id=api_id, schema_hash=schema_hash)) == 1


@pytest.mark.asyncio
async def test_skip_duplicate_recipe_with_different_description(monkeypatch):
    store = MemoryApiAgentStore(max_size=10)
    monkeypatch.setattr(
        "api_agent.recipe.learning.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store)
    )
    monkeypatch.setattr("api_agent.recipe.learning.settings.ENABLE_RECIPES", True)

    raw_schema = '{"schema":"ok"}'
    api_id = "graphql:https://api.example.com/graphql"
    schema_hash = sha256_hex(raw_schema)
    existing = _graphql_recipe()
    store.save_recipe(
        api_id=api_id,
        schema_hash=schema_hash,
        question="List users",
        recipe=existing,
        tool_name=existing["public_contract"]["tool_name"],
    )

    candidate = deepcopy(existing)
    candidate["public_contract"]["description"] = (
        "Use for listing user identifiers. Returns user ids as CSV. No required params. Do not use for different fields, joins, or workflows."
    )

    async def fake_extract_recipe(**_kwargs):
        return candidate

    monkeypatch.setattr("api_agent.recipe.learning.extract_recipe", fake_extract_recipe)

    await maybe_extract_and_save_recipe(
        api_type="graphql",
        api_id=api_id,
        question="List users",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { id } }"}],
        raw_schema=raw_schema,
        learn_rate=1,
        original_result=[{"id": 1}],
        validate_candidate=lambda _recipe, _args: _async_result([{"id": 1}]),
    )

    recipes = store.list_recipes(api_id=api_id, schema_hash=schema_hash)
    assert len(recipes) == 1
    assert recipes[0]["tool_name"] == "list_users"


@pytest.mark.asyncio
async def test_deduplicate_tool_name_on_collision(monkeypatch):
    store = MemoryApiAgentStore(max_size=10)
    monkeypatch.setattr(
        "api_agent.recipe.learning.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store)
    )
    monkeypatch.setattr("api_agent.recipe.learning.settings.ENABLE_RECIPES", True)

    raw_schema = '{"schema":"ok"}'
    api_id = "graphql:https://api.example.com/graphql"
    schema_hash = sha256_hex(raw_schema)

    recipe_existing = {
        **_graphql_recipe(),
    }
    store.save_recipe(
        api_id=api_id,
        schema_hash=schema_hash,
        question="List users",
        recipe=recipe_existing,
        tool_name=recipe_existing["public_contract"]["tool_name"],
    )

    recipe_new = _graphql_recipe()
    recipe_new["public_contract"] = {
        "tool_name": "list_users",
        "description": "Use for listing users by name. Returns user names as CSV. No required params. Do not use for different user fields, joins, or workflows.",
        "tool_args": {},
    }
    recipe_new["execution_plan"] = {
        "steps": [_graphql_step("users", "{ users { name } }")],
    }

    async def fake_extract_recipe(**_kwargs):
        return dict(recipe_new)

    monkeypatch.setattr("api_agent.recipe.learning.extract_recipe", fake_extract_recipe)

    await maybe_extract_and_save_recipe(
        api_type="graphql",
        api_id=api_id,
        question="List users by name",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { name } }"}],
        raw_schema=raw_schema,
        learn_rate=1,
        original_result=[{"name": "Ada"}],
        validate_candidate=lambda _recipe, _args: _async_result([{"name": "Ada"}]),
    )

    tool_names = {
        r["tool_name"] for r in store.list_recipes(api_id=api_id, schema_hash=schema_hash)
    }
    assert "list_users" in tool_names
    assert "list_users_2" in tool_names


@pytest.mark.asyncio
async def test_result_mismatch_rejects_recipe(monkeypatch, caplog):
    store = MemoryApiAgentStore(max_size=10)
    monkeypatch.setattr(
        "api_agent.recipe.learning.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store)
    )
    monkeypatch.setattr("api_agent.recipe.learning.settings.ENABLE_RECIPES", True)
    caplog.set_level(logging.INFO, logger="api_agent.recipe.learning")
    span = SimpleNamespace(name="", attrs={}, events=[])

    def set_attribute(key, value):
        span.attrs[key] = value

    def add_event(name, attrs):
        span.events.append((name, attrs))

    class FakeTraceSpan:
        def __enter__(self):
            return span

        def __exit__(self, *_args):
            return False

    def fake_trace_span(name, attributes=None):
        span.name = name
        span.attrs.update(attributes or {})
        span.set_attribute = set_attribute
        span.add_event = add_event
        return FakeTraceSpan()

    async def fake_extract_recipe(**_kwargs):
        return _graphql_recipe()

    monkeypatch.setattr("api_agent.recipe.learning.trace_span", fake_trace_span)
    monkeypatch.setattr("api_agent.recipe.learning.extract_recipe", fake_extract_recipe)

    await maybe_extract_and_save_recipe(
        api_type="graphql",
        api_id="graphql:https://api.example.com/graphql",
        question="List users",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { id } }"}],
        raw_schema='{"schema":"ok"}',
        learn_rate=1,
        original_result=[{"id": 1}],
        validate_candidate=lambda _recipe, _args: _async_result([{"id": 2}]),
    )

    assert (
        store.list_recipes(
            api_id="graphql:https://api.example.com/graphql",
            schema_hash=sha256_hex('{"schema":"ok"}'),
        )
        == []
    )
    assert span.name == "recipe.learning"
    assert span.attrs["recipe.learning.outcome"] == "skipped"
    assert span.attrs["recipe.learning.skip_reason"] == "candidate_result_mismatch"
    assert span.attrs["recipe.candidate_rows"] == 1
    assert span.attrs["recipe.expected_rows"] == 1
    event_name, event_attrs = span.events[-1]
    assert event_name == "recipe.learning.skipped"
    assert event_attrs["recipe.learning.skip_reason"] == "candidate_result_mismatch"
    assert (
        "Skipping recipe extraction (candidate result mismatch) candidate_rows=1 expected_rows=1"
    ) in caplog.text


@pytest.mark.asyncio
async def test_result_mismatch_repairs_and_saves_recipe(monkeypatch):
    store = MemoryApiAgentStore(max_size=10)
    monkeypatch.setattr(
        "api_agent.recipe.learning.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store)
    )
    monkeypatch.setattr("api_agent.recipe.learning.settings.ENABLE_RECIPES", True)

    bad_recipe = _graphql_recipe()
    bad_recipe["execution_plan"]["steps"][0]["call"]["query_template"] = "{ users { name } }"
    good_recipe = _graphql_recipe()
    extract_calls = []

    async def fake_extract_recipe(**kwargs):
        extract_calls.append(kwargs)
        return bad_recipe if len(extract_calls) == 1 else good_recipe

    async def validate_candidate(recipe, _args):
        query = recipe["execution_plan"]["steps"][0]["call"]["query_template"]
        return [{"id": 1}] if "{ users { id } }" in query else []

    monkeypatch.setattr("api_agent.recipe.learning.extract_recipe", fake_extract_recipe)

    await maybe_extract_and_save_recipe(
        api_type="graphql",
        api_id="graphql:https://api.example.com/graphql",
        question="List users",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { id } }"}],
        raw_schema='{"schema":"ok"}',
        learn_rate=1,
        original_result=[{"id": 1}],
        validate_candidate=validate_candidate,
    )

    recipes = store.list_recipes(
        api_id="graphql:https://api.example.com/graphql",
        schema_hash=sha256_hex('{"schema":"ok"}'),
    )
    assert len(recipes) == 1
    assert len(extract_calls) == 2
    assert extract_calls[0]["validation_feedback"] is None
    feedback = extract_calls[1]["validation_feedback"]
    assert feedback["reason"] == "candidate_result_mismatch"
    assert feedback["candidate_result"]["row_count"] == 0
    assert feedback["expected_result"]["row_count"] == 1
    assert feedback["previous_candidate"] == bad_recipe


@pytest.mark.asyncio
async def test_large_results_are_sampled_for_extractor_but_validated_fully(monkeypatch):
    store = MemoryApiAgentStore(max_size=10)
    monkeypatch.setattr(
        "api_agent.recipe.learning.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store)
    )
    monkeypatch.setattr("api_agent.recipe.learning.settings.ENABLE_RECIPES", True)
    full_result = [{"id": i} for i in range(8)]

    async def fake_extract_recipe(**kwargs):
        assert kwargs["result"] == {
            "row_count": 8,
            "sample": full_result[:5],
            "truncated": True,
        }
        assert kwargs["steps"][0]["result"] == {
            "row_count": 8,
            "sample": full_result[:5],
            "truncated": True,
        }
        return _graphql_recipe()

    monkeypatch.setattr("api_agent.recipe.learning.extract_recipe", fake_extract_recipe)

    await maybe_extract_and_save_recipe(
        api_type="graphql",
        api_id="graphql:https://api.example.com/graphql",
        question="List users",
        steps=[
            {
                "kind": "graphql",
                "name": "users",
                "query": "{ users { id } }",
                "result": full_result,
            }
        ],
        raw_schema='{"schema":"ok"}',
        learn_rate=1,
        original_result=full_result,
        validate_candidate=lambda _recipe, _args: _async_result(full_result),
    )

    recipes = store.list_recipes(
        api_id="graphql:https://api.example.com/graphql",
        schema_hash=sha256_hex('{"schema":"ok"}'),
    )
    assert len(recipes) == 1
    assert "validation_fixture" not in recipes[0]


@pytest.mark.asyncio
async def test_existing_recipes_context_drops_validation_results(monkeypatch):
    store = MemoryApiAgentStore(max_size=10)
    monkeypatch.setattr(
        "api_agent.recipe.learning.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store)
    )
    monkeypatch.setattr("api_agent.recipe.learning.settings.ENABLE_RECIPES", True)

    raw_schema = '{"schema":"ok"}'
    api_id = "graphql:https://api.example.com/graphql"
    existing = _graphql_recipe()
    existing["validation_fixture"]["result"] = [{"secret": "do-not-send"}]
    store.save_recipe(
        api_id=api_id,
        schema_hash=sha256_hex(raw_schema),
        question="List users",
        recipe=existing,
        tool_name=existing["public_contract"]["tool_name"],
    )

    async def fake_extract_recipe(**kwargs):
        existing_context = kwargs["existing_recipes"]
        assert len(existing_context) == 1
        assert "validation_fixture" not in existing_context[0]
        assert "execution_plan" not in existing_context[0]
        assert "do-not-send" not in json.dumps(existing_context)
        assert existing_context[0]["tool_name"] == "list_users"
        assert existing_context[0]["public_contract"]["tool_args"] == {}
        return _graphql_recipe()

    monkeypatch.setattr("api_agent.recipe.learning.extract_recipe", fake_extract_recipe)

    await maybe_extract_and_save_recipe(
        api_type="graphql",
        api_id=api_id,
        question="List users",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { id } }"}],
        raw_schema=raw_schema,
        learn_rate=1,
        original_result=[{"id": 1}],
        validate_candidate=lambda _recipe, _args: _async_result([{"id": 1}]),
    )


@pytest.mark.asyncio
async def test_repeated_terminal_rest_results_are_validation_baseline(monkeypatch):
    store = MemoryApiAgentStore(max_size=10)
    monkeypatch.setattr(
        "api_agent.recipe.learning.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store)
    )
    monkeypatch.setattr("api_agent.recipe.learning.settings.ENABLE_RECIPES", True)

    recipe = {
        "public_contract": {
            "tool_name": "get_team_key_results",
            "description": "Use for listing key results for a team in one cycle. Returns key result rows. Requires team and cycle. Do not use for different fields, joins, or workflow.",
            "tool_args": {
                "team": {"type": "str", "description": "Team"},
                "cycle": {"type": "str", "description": "Cycle"},
            },
        },
        "execution_plan": {"steps": [_rest_step("key_results", "/key-results")]},
        "validation_fixture": {"tool_args": {"team": "alpha-suite", "cycle": "Y2026Q2"}},
    }
    expected_result = [{"id": 1}, {"id": 2}, {"id": 3}]

    async def fake_extract_recipe(**kwargs):
        assert kwargs["result"] == expected_result
        return recipe

    monkeypatch.setattr("api_agent.recipe.learning.extract_recipe", fake_extract_recipe)

    await maybe_extract_and_save_recipe(
        api_type="rest",
        api_id="rest:https://spec|https://api",
        question="List alpha-suite key results",
        steps=[
            {
                "kind": "rest",
                "method": "GET",
                "path": "/objectives/{objective_id}/key-results",
                "path_params": {"objective_id": 1},
                "result": [{"id": 1}],
            },
            {
                "kind": "rest",
                "method": "GET",
                "path": "/objectives/{objective_id}/key-results",
                "path_params": {"objective_id": 2},
                "result": [{"id": 2}, {"id": 3}],
            },
        ],
        raw_schema='{"schema":"ok"}',
        learn_rate=1,
        original_result=[{"id": 2}, {"id": 3}],
        validate_candidate=lambda _recipe, _args: _async_result(expected_result),
    )

    assert (
        len(
            store.list_recipes(
                api_id="rest:https://spec|https://api",
                schema_hash=sha256_hex('{"schema":"ok"}'),
            )
        )
        == 1
    )


@pytest.mark.asyncio
async def test_empty_intermediate_probe_steps_are_not_sent_to_extractor(monkeypatch, caplog):
    store = MemoryApiAgentStore(max_size=10)
    monkeypatch.setattr(
        "api_agent.recipe.learning.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store)
    )
    monkeypatch.setattr("api_agent.recipe.learning.settings.ENABLE_RECIPES", True)
    caplog.set_level(logging.INFO, logger="api_agent.recipe.learning")

    recipe = {
        "public_contract": {
            "tool_name": "get_team_key_results",
            "description": "Use for listing key results for a team in one cycle. Returns key result rows. Requires team and cycle. Do not use for different fields, joins, or workflow.",
            "tool_args": {
                "team": {"type": "str", "description": "Team"},
                "cycle": {"type": "str", "description": "Cycle"},
            },
        },
        "execution_plan": {"steps": [_rest_step("key_results", "/key-results")]},
        "validation_fixture": {"tool_args": {"team": "Example Team", "cycle": "Y2026Q2"}},
    }
    expected_result = [{"id": 1}, {"id": 2}]

    async def fake_extract_recipe(**kwargs):
        sql_steps = [step for step in kwargs["steps"] if step["kind"] == "sql"]
        assert [step["query"] for step in sql_steps] == [
            "SELECT id FROM objectives_q2 WHERE assignee.team.id=1087"
        ]
        assert kwargs["result"] == expected_result
        return recipe

    monkeypatch.setattr("api_agent.recipe.learning.extract_recipe", fake_extract_recipe)

    await maybe_extract_and_save_recipe(
        api_type="rest",
        api_id="rest:https://spec|https://api",
        question="List Example Team key results for Q2 2026",
        steps=[
            {
                "kind": "rest",
                "method": "GET",
                "path": "/api/v2/objectives",
                "query_params": {"cycle": "Y2026Q2"},
                "result": [{"id": 1}],
            },
            {
                "kind": "sql",
                "query": (
                    "SELECT id FROM objectives_q2 "
                    "WHERE lower(assignee.team.name) LIKE '%exampleteam%'"
                ),
                "result": [],
            },
            {
                "kind": "rest",
                "method": "GET",
                "path": "/api/v2/teams",
                "result": [{"id": 1087, "name": "Example Team"}],
            },
            {
                "kind": "sql",
                "query": "SELECT id FROM objectives_q2 WHERE assignee.team.id=1087",
                "result": [{"id": 22380}, {"id": 22381}],
            },
            {
                "kind": "rest",
                "method": "GET",
                "path": "/api/v2/objectives/{objective_id}/key-results",
                "path_params": {"objective_id": 22380},
                "result": [{"id": 1}],
            },
            {
                "kind": "rest",
                "method": "GET",
                "path": "/api/v2/objectives/{objective_id}/key-results",
                "path_params": {"objective_id": 22381},
                "result": [{"id": 2}],
            },
        ],
        raw_schema='{"schema":"ok"}',
        learn_rate=1,
        original_result=[{"id": 2}],
        validate_candidate=lambda _recipe, _args: _async_result(expected_result),
    )

    assert "Pruned recipe learning dead-end steps count=1" in caplog.text
    assert (
        len(
            store.list_recipes(
                api_id="rest:https://spec|https://api",
                schema_hash=sha256_hex('{"schema":"ok"}'),
            )
        )
        == 1
    )


@pytest.mark.asyncio
async def test_empty_final_result_can_be_learned(monkeypatch):
    store = MemoryApiAgentStore(max_size=10)
    monkeypatch.setattr(
        "api_agent.recipe.learning.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store)
    )
    monkeypatch.setattr("api_agent.recipe.learning.settings.ENABLE_RECIPES", True)

    recipe = {
        "public_contract": {
            "tool_name": "find_team_objectives",
            "description": "Use for listing objectives for a team in one cycle. Returns objective rows. Requires team and cycle. Do not use for different fields, joins, or workflow.",
            "tool_args": {
                "team": {"type": "str", "description": "Team"},
                "cycle": {"type": "str", "description": "Cycle"},
            },
        },
        "execution_plan": {"steps": [_rest_step("objectives", "/api/v2/objectives")]},
        "validation_fixture": {"tool_args": {"team": "No Match", "cycle": "Y2026Q2"}},
    }

    async def fake_extract_recipe(**kwargs):
        assert kwargs["result"] == []
        assert kwargs["steps"][-1]["result"] == []
        return recipe

    monkeypatch.setattr("api_agent.recipe.learning.extract_recipe", fake_extract_recipe)

    await maybe_extract_and_save_recipe(
        api_type="rest",
        api_id="rest:https://spec|https://api",
        question="List No Match objectives for Q2 2026",
        steps=[
            {
                "kind": "rest",
                "method": "GET",
                "path": "/api/v2/objectives",
                "query_params": {"cycle": "Y2026Q2"},
                "result": [{"id": 1}],
            },
            {
                "kind": "sql",
                "query": "SELECT id FROM objectives WHERE assignee.team.name = 'No Match'",
                "result": [],
            },
        ],
        raw_schema='{"schema":"ok"}',
        learn_rate=1,
        original_result=[],
        validate_candidate=lambda _recipe, _args: _async_result([]),
    )

    recipes = store.list_recipes(
        api_id="rest:https://spec|https://api",
        schema_hash=sha256_hex('{"schema":"ok"}'),
    )
    assert len(recipes) == 1
    assert "validation_fixture" not in recipes[0]


@pytest.mark.asyncio
async def test_result_order_mismatch_rejects_recipe(monkeypatch):
    store = MemoryApiAgentStore(max_size=10)
    monkeypatch.setattr(
        "api_agent.recipe.learning.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store)
    )
    monkeypatch.setattr("api_agent.recipe.learning.settings.ENABLE_RECIPES", True)

    async def fake_extract_recipe(**_kwargs):
        return _graphql_recipe()

    monkeypatch.setattr("api_agent.recipe.learning.extract_recipe", fake_extract_recipe)

    await maybe_extract_and_save_recipe(
        api_type="graphql",
        api_id="graphql:https://api.example.com/graphql",
        question="List users",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { id } }"}],
        raw_schema='{"schema":"ok"}',
        learn_rate=1,
        original_result=[{"id": 1}, {"id": 2}],
        validate_candidate=lambda _recipe, _args: _async_result([{"id": 2}, {"id": 1}]),
    )

    assert (
        store.list_recipes(
            api_id="graphql:https://api.example.com/graphql",
            schema_hash=sha256_hex('{"schema":"ok"}'),
        )
        == []
    )


@pytest.mark.asyncio
async def test_optimized_candidate_can_use_different_steps_when_result_matches(monkeypatch):
    store = MemoryApiAgentStore(max_size=10)
    monkeypatch.setattr(
        "api_agent.recipe.learning.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store)
    )
    monkeypatch.setattr("api_agent.recipe.learning.settings.ENABLE_RECIPES", True)
    recipe = _graphql_recipe()
    recipe["execution_plan"]["steps"] = [_graphql_step("users", "{ users { id name } }")]

    async def fake_extract_recipe(**_kwargs):
        return recipe

    monkeypatch.setattr("api_agent.recipe.learning.extract_recipe", fake_extract_recipe)

    await maybe_extract_and_save_recipe(
        api_type="graphql",
        api_id="graphql:https://api.example.com/graphql",
        question="List users",
        steps=[
            {"kind": "graphql", "name": "users", "query": "{ users { id } }"},
            {"kind": "sql", "query": "SELECT id FROM users"},
        ],
        raw_schema='{"schema":"ok"}',
        learn_rate=1,
        original_result=[{"id": 1}],
        validate_candidate=lambda _recipe, _args: _async_result([{"id": 1}]),
    )

    saved = store.list_recipes(
        api_id="graphql:https://api.example.com/graphql",
        schema_hash=sha256_hex('{"schema":"ok"}'),
    )
    assert len(saved) == 1
    assert saved[0]["steps"] == recipe["execution_plan"]["steps"]


@pytest.mark.asyncio
async def test_learn_rate_one_overrides_strong_match_skip(monkeypatch):
    store = MemoryApiAgentStore(max_size=10)
    monkeypatch.setattr(
        "api_agent.recipe.learning.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store)
    )
    monkeypatch.setattr("api_agent.recipe.learning.settings.ENABLE_RECIPES", True)
    monkeypatch.setattr("api_agent.recipe.learning.settings.RECIPE_LEARN_RATE", 0.2)

    raw_schema = '{"schema":"ok"}'
    api_id = "graphql:https://api.example.com/graphql"
    recipe = _graphql_recipe()
    called = False

    async def fake_extract_recipe(**_kwargs):
        nonlocal called
        called = True
        return dict(recipe)

    monkeypatch.setattr("api_agent.recipe.learning.extract_recipe", fake_extract_recipe)

    await maybe_extract_and_save_recipe(
        api_type="graphql",
        api_id=api_id,
        question="List users",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { id } }"}],
        raw_schema=raw_schema,
        learn_rate=1,
        strong_recipe_match=True,
        original_result=[{"id": 1}],
        validate_candidate=lambda _recipe, _args: _async_result([{"id": 1}]),
    )

    assert called is True
    assert len(store.list_recipes(api_id=api_id, schema_hash=sha256_hex(raw_schema))) == 1


@pytest.mark.asyncio
async def test_strong_match_uses_default_rate_to_skip_extraction(monkeypatch):
    store = MemoryApiAgentStore(max_size=10)
    monkeypatch.setattr(
        "api_agent.recipe.learning.ASYNC_API_AGENT_STORE", AsyncApiAgentStore(store)
    )
    monkeypatch.setattr("api_agent.recipe.learning.settings.ENABLE_RECIPES", True)
    monkeypatch.setattr("api_agent.recipe.learning.settings.RECIPE_LEARN_RATE", 0.2)

    async def fake_extract_recipe(**_kwargs):
        raise AssertionError("extractor should not run")

    monkeypatch.setattr("api_agent.recipe.learning.extract_recipe", fake_extract_recipe)

    await maybe_extract_and_save_recipe(
        api_type="graphql",
        api_id="graphql:https://api.example.com/graphql",
        question="List users",
        steps=[{"kind": "graphql", "name": "users", "query": "{ users { id } }"}],
        raw_schema='{"schema":"ok"}',
        strong_recipe_match=True,
    )

    assert (
        store.list_recipes(
            api_id="graphql:https://api.example.com/graphql",
            schema_hash=sha256_hex('{"schema":"ok"}'),
        )
        == []
    )
