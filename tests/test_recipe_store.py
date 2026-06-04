"""Unit tests for API Agent store utilities."""

import pytest

from api_agent.recipe.contracts import validate_recipe_contract
from api_agent.recipe.execution import render_rest_call_sets
from api_agent.recipe.identity import recipe_fingerprint
from api_agent.recipe.learning import should_learn_recipe
from api_agent.recipe.search import build_recipe_context
from api_agent.recipe.templates import render_param_refs, render_text_template
from api_agent.store import API_AGENT_STORE, MemoryApiAgentStore, sha256_hex


def _recipe(
    tool_name: str = "test_recipe",
    tool_args: dict | None = None,
    steps: list | None = None,
) -> dict:
    return {
        "public_contract": {
            "tool_name": tool_name,
            "description": "Use for a tested recipe. Returns rows as CSV. No required params. Do not use for different fields, joins, or workflows.",
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
    output: dict | None = None,
) -> dict:
    return {
        "id": step_id,
        "kind": "rest",
        "input": input_spec or {"mode": "single", "with": {}},
        "call": {
            "method": "GET",
            "path": path,
            "path_params": path_params or {},
            "query_params": query_params or {},
            "body": {},
        },
        "output": output or {"name": step_id},
    }


def test_render_text_template_basic():
    assert render_text_template("limit {{n}}", {"n": 10}) == "limit 10"
    assert render_text_template("active={{flag}}", {"flag": True}) == "active=true"
    assert render_text_template("v={{x}}", {"x": None}) == "v=null"


def test_sha256_hex_normalizes_json():
    a = '{"b": 1, "a": 2}'
    b = '{"a": 2, "b": 1}'
    assert sha256_hex(a) == sha256_hex(b)


def test_memory_store_caches_downstream_description_by_api_and_schema():
    store = MemoryApiAgentStore(max_size=10)

    store.save_downstream_description(
        api_id="rest:https://spec|https://api",
        schema_hash="schema-a",
        description="Query objectives and key results.",
    )

    assert (
        store.get_downstream_description(
            api_id="rest:https://spec|https://api",
            schema_hash="schema-a",
        )
        == "Query objectives and key results."
    )
    assert (
        store.get_downstream_description(
            api_id="rest:https://spec|https://api",
            schema_hash="schema-b",
        )
        is None
    )


def test_memory_store_expires_downstream_description(monkeypatch):
    now = 1000.0
    monkeypatch.setattr("api_agent.store.time.time", lambda: now)
    store = MemoryApiAgentStore(max_size=10)

    store.save_downstream_description(
        api_id="rest:https://spec|https://api",
        schema_hash="schema-a",
        description="Query objectives and key results.",
        ttl_seconds=10,
    )

    assert (
        store.get_downstream_description(
            api_id="rest:https://spec|https://api",
            schema_hash="schema-a",
        )
        == "Query objectives and key results."
    )
    now = 1011.0
    assert (
        store.get_downstream_description(
            api_id="rest:https://spec|https://api",
            schema_hash="schema-a",
        )
        is None
    )


def test_render_param_refs_nested():
    obj = {"a": {"$var": "x"}, "b": [{"$var": "y"}], "c": 3}
    out = render_param_refs(obj, {"x": 1, "y": "foo"})
    assert out == {"a": 1, "b": ["foo"], "c": 3}


def test_render_rest_call_sets_maps_binding_rows():
    step = _rest_step(
        "key_results",
        "/objectives/{id}/key-results",
        input_spec={
            "mode": "map",
            "from": "objectives",
            "bind": {"objective_id": "id"},
            "with": {"cycle": {"value": "cycle"}},
        },
        path_params={"id": {"$var": "objective_id"}},
        query_params={"cycle": {"$var": "cycle"}},
    )

    rendered, error = render_rest_call_sets(
        step,
        {"cycle": "Y2026Q2"},
        {"objectives": [{"id": 10}, {"id": 11}]},
    )

    assert error == ""
    assert [(r.path_params, r.query_params, r.body, r.binding) for r in rendered] == [
        ({"id": 10}, {"cycle": "Y2026Q2"}, None, {"objective_id": 10}),
        ({"id": 11}, {"cycle": "Y2026Q2"}, None, {"objective_id": 11}),
    ]


def test_render_rest_call_sets_zips_fields_from_one_binding_rowset():
    step = _rest_step(
        "key_results",
        "/objectives/{id}/key-results",
        input_spec={
            "mode": "map",
            "from": "objective_owner_pairs",
            "bind": {"objective_id": "id", "owner_id": "owner_id"},
        },
        path_params={"id": {"$var": "objective_id"}},
        query_params={"owner": {"$var": "owner_id"}},
    )

    rendered, error = render_rest_call_sets(
        step,
        {},
        {"objective_owner_pairs": [{"id": 10, "owner_id": 20}, {"id": 11, "owner_id": 21}]},
    )

    assert error == ""
    assert [(r.path_params, r.query_params, r.binding) for r in rendered] == [
        ({"id": 10}, {"owner": 20}, {"objective_id": 10, "owner_id": 20}),
        ({"id": 11}, {"owner": 21}, {"objective_id": 11, "owner_id": 21}),
    ]


def test_render_rest_call_sets_batches_binding_rows():
    step = _rest_step(
        "key_results",
        "/key-results",
        input_spec={
            "mode": "batch",
            "from": "objectives",
            "bind": {"objective_ids": "id"},
        },
        query_params={"ids": {"$var": "objective_ids"}},
    )

    rendered, error = render_rest_call_sets(
        step,
        {},
        {"objectives": [{"id": 10}, {"id": 11}]},
    )

    assert error == ""
    assert [(r.query_params, r.binding) for r in rendered] == [
        ({"ids": [10, 11]}, {"objective_ids": [10, 11]})
    ]


def test_validate_recipe_contract_rejects_missing_binding_rowset():
    step = _rest_step(
        "key_results",
        "/objectives/{id}/key-results",
        input_spec={"mode": "map", "from": "objectives", "bind": {"objective_id": "id"}},
        path_params={"objective_id": {"$var": "objective_id"}},
    )
    recipe = _recipe(
        steps=[step],
    )

    assert validate_recipe_contract(recipe, "rest") == "step input must reference prior output"


def test_validate_recipe_contract_rejects_duplicate_step_output():
    recipe = _recipe(
        steps=[
            _rest_step("users", "/users", output={"name": "rows"}),
            _rest_step("posts", "/posts", output={"name": "rows"}),
        ],
    )

    assert validate_recipe_contract(recipe, "rest") == "duplicate step output"


def test_validate_recipe_contract_rejects_batch_output_binding_attachment():
    recipe = _recipe(
        steps=[
            _rest_step("objectives", "/objectives"),
            _rest_step(
                "key_results",
                "/key-results",
                input_spec={
                    "mode": "batch",
                    "from": "objectives",
                    "bind": {"objective_ids": "id"},
                },
                query_params={"ids": {"$var": "objective_ids"}},
                output={"name": "key_results", "attach_binding": ["objective_ids"]},
            ),
        ],
    )

    assert validate_recipe_contract(recipe, "rest") == "invalid output binding"


def test_render_rest_call_sets_allows_empty_map():
    step = _rest_step(
        "key_results",
        "/objectives/{id}/key-results",
        input_spec={"mode": "map", "from": "objectives", "bind": {"objective_id": "id"}},
        path_params={"id": {"$var": "objective_id"}},
    )
    rendered, error = render_rest_call_sets(step, {}, {"objectives": []})

    assert error == ""
    assert rendered == []


def test_recipe_store_preserves_defaults():
    """Defaults are preserved as-is (no sensitivity filtering)."""
    store = MemoryApiAgentStore(max_size=10)
    recipe = _recipe(
        tool_args={
            "user_id": {"type": "str", "default": "123e4567-e89b-12d3-a456-426614174000"},
            "limit": {"type": "int", "default": 10},
        },
        steps=[],
    )
    recipe_id = store.save_recipe(
        api_id="rest:https://spec|https://api",
        schema_hash="s",
        question="q",
        recipe=recipe,
        tool_name="test_recipe",
    )
    saved = store.get_recipe(recipe_id)
    assert saved is not None
    # Defaults preserved exactly as provided
    assert saved["public_contract"]["tool_args"]["user_id"]["default"] == (
        "123e4567-e89b-12d3-a456-426614174000"
    )
    assert saved["public_contract"]["tool_args"]["limit"]["default"] == 10


def test_recipe_store_save_is_idempotent_by_fingerprint():
    store = MemoryApiAgentStore(max_size=10)
    recipe = _recipe(steps=[_graphql_step("users", "{ users { id } }")])

    first_id = store.save_recipe(
        api_id="graphql:https://api.example.com/graphql",
        schema_hash="s",
        question="list users",
        recipe=recipe,
        tool_name="list_users",
    )
    second_id = store.save_recipe(
        api_id="graphql:https://api.example.com/graphql",
        schema_hash="s",
        question="list users again",
        recipe=recipe,
        tool_name="list_users_duplicate",
    )

    assert second_id == first_id
    assert (
        len(store.list_recipes(api_id="graphql:https://api.example.com/graphql", schema_hash="s"))
        == 1
    )


def test_recipe_store_fifo_eviction_does_not_promote_duplicates():
    store = MemoryApiAgentStore(max_size=2)
    api_id = "graphql:https://api.example.com/graphql"
    recipes = [
        _recipe(tool_name="a", steps=[_graphql_step("a", "{ a }")]),
        _recipe(tool_name="b", steps=[_graphql_step("b", "{ b }")]),
        _recipe(tool_name="c", steps=[_graphql_step("c", "{ c }")]),
    ]

    first_id = store.save_recipe(
        api_id=api_id, schema_hash="s", question="a", recipe=recipes[0], tool_name="a"
    )
    store.save_recipe(
        api_id=api_id, schema_hash="s", question="a again", recipe=recipes[0], tool_name="a"
    )
    second_id = store.save_recipe(
        api_id=api_id, schema_hash="s", question="b", recipe=recipes[1], tool_name="b"
    )
    third_id = store.save_recipe(
        api_id=api_id, schema_hash="s", question="c", recipe=recipes[2], tool_name="c"
    )

    ids = {r["recipe_id"] for r in store.list_recipes(api_id=api_id, schema_hash="s")}
    assert first_id not in ids
    assert ids == {second_id, third_id}


def test_recipe_store_disable_hides_recipe():
    store = MemoryApiAgentStore(max_size=10)
    api_id = "rest:https://spec|https://api"
    recipe = _recipe(tool_name="list_users", steps=[_rest_step("users", "/users")])
    recipe_id = store.save_recipe(
        api_id=api_id, schema_hash="s", question="users", recipe=recipe, tool_name="list_users"
    )

    assert store.disable_recipe(recipe_id)
    assert store.list_recipes(api_id=api_id, schema_hash="s") == []
    assert (
        store.list_recipes(api_id=api_id, schema_hash="s", include_disabled=True)[0]["enabled"]
        is False
    )


def test_recipe_fingerprint_normalizes_whitespace():
    left = {
        **_recipe(
            steps=[
                _graphql_step("users", "{ users { id } }"),
                _sql_step("filtered_users", "SELECT * FROM users"),
            ]
        )
    }
    right = {
        **_recipe(
            steps=[
                _graphql_step("users", "{   users { id } }"),
                _sql_step("filtered_users", "SELECT  *   FROM users"),
            ]
        )
    }

    assert recipe_fingerprint(
        api_id="graphql:x", schema_hash="s", recipe=left
    ) == recipe_fingerprint(api_id="graphql:x", schema_hash="s", recipe=right)


def test_recipe_fingerprint_ignores_generated_copy_and_name():
    left = _recipe(tool_name="list_users")
    right = _recipe(tool_name="fetch_users")
    right["public_contract"]["description"] = (
        "Use for fetching user ids. Returns rows as CSV. No required params. Do not use for other workflows."
    )

    assert recipe_fingerprint(
        api_id="graphql:x", schema_hash="s", recipe=left
    ) == recipe_fingerprint(api_id="graphql:x", schema_hash="s", recipe=right)


def test_should_learn_recipe_rates():
    assert not should_learn_recipe(api_id="a", schema_hash="s", question="q", learn_rate=0)
    assert should_learn_recipe(api_id="a", schema_hash="s", question="q", learn_rate=1)
    assert should_learn_recipe(
        api_id="a", schema_hash="s", question="q", learn_rate=0.5
    ) == should_learn_recipe(api_id="a", schema_hash="s", question="q", learn_rate=0.5)


def test_recipe_store_scoring_prefers_closer_match():
    store = MemoryApiAgentStore(max_size=10)
    r1 = _recipe(tool_name="top_hotels", steps=[])
    r2 = _recipe(tool_name="list_users", steps=[])
    id1 = store.save_recipe(
        api_id="rest:a|b",
        schema_hash="s",
        question="top hotels by rating",
        recipe=r1,
        tool_name="top_hotels",
    )
    id2 = store.save_recipe(
        api_id="rest:a|b",
        schema_hash="s",
        question="list users by age",
        recipe=r2,
        tool_name="list_users",
    )
    assert id1 and id2
    suggestions = store.suggest_recipes(
        api_id="rest:a|b", schema_hash="s", question="best hotels", k=2
    )
    assert suggestions
    assert suggestions[0]["recipe_id"] == id1


def test_recipe_store_scoring_handles_token_order():
    store = MemoryApiAgentStore(max_size=10)
    r1 = _recipe(tool_name="find_hotels_in_nyc", steps=[])
    r2 = _recipe(tool_name="find_users_in_nyc", steps=[])
    id1 = store.save_recipe(
        api_id="rest:a|b",
        schema_hash="s",
        question="find hotels in nyc",
        recipe=r1,
        tool_name="find_hotels_in_nyc",
    )
    _id2 = store.save_recipe(
        api_id="rest:a|b",
        schema_hash="s",
        question="find users in nyc",
        recipe=r2,
        tool_name="find_users_in_nyc",
    )
    suggestions = store.suggest_recipes(
        api_id="rest:a|b", schema_hash="s", question="nyc hotels find", k=2
    )
    assert suggestions
    assert suggestions[0]["recipe_id"] == id1


def test_render_text_template_missing_param_raises():
    import pytest

    with pytest.raises(KeyError):
        render_text_template("limit {{n}}", {})


def test_validate_recipe_params_requires_all_params():
    """All declared params are required (defaults are examples, not fallbacks)."""
    from api_agent.recipe.execution import validate_recipe_params

    params_spec = {
        "manager_name": {"type": "str", "default": "Alice Smith"},
        "manager_email": {"type": "str", "default": "alice.smith@example.com"},
    }
    params, error = validate_recipe_params(params_spec, {"manager_name": "Bob Jones"})
    assert params is None
    assert "missing required param: manager_email" in error


def test_validate_recipe_params_rejects_extra():
    """Extra params are rejected even when spec is non-empty."""
    from api_agent.recipe.execution import validate_recipe_params

    params_spec = {"limit": {"type": "int", "default": 10}}
    params, error = validate_recipe_params(params_spec, {"limit": 5, "extra": "bad"})
    assert params is None
    assert "unexpected params: extra" in error


def test_validate_recipe_params_enforces_declared_types():
    from api_agent.recipe.execution import validate_recipe_params

    params_spec = {"limit": {"type": "int", "description": "Limit"}}
    params, error = validate_recipe_params(params_spec, {"limit": "ten"})
    assert params is None
    assert "invalid param type: limit must be int" in error


def test_validate_recipe_params_coerces_declared_types():
    from api_agent.recipe.execution import validate_recipe_params

    params_spec = {"limit": {"type": "int", "description": "Limit"}}
    params, error = validate_recipe_params(params_spec, {"limit": "10"})
    assert error == ""
    assert params == {"limit": 10}


def test_global_recipe_store_available():
    # Basic smoke test to ensure singleton is constructed
    assert API_AGENT_STORE is not None


def test_build_recipe_context_empty():
    """Empty suggestions returns empty string."""
    assert build_recipe_context([]) == ""


def _recipe_suggestion(
    *,
    recipe_id: str,
    score: float,
    question: str,
    recipe: dict,
    params: dict | None = None,
    tool_name: str | None = None,
) -> dict:
    suggestion = {
        "recipe_id": recipe_id,
        "score": score,
        "question": question,
        "params": params or {},
        "recipe": recipe,
    }
    if tool_name:
        suggestion["tool_name"] = tool_name
    return suggestion


def test_build_recipe_context_with_suggestions():
    """Suggestions are formatted correctly for prompt injection."""
    r1 = _recipe(
        tool_name="get_users_starting_with_a",
        tool_args={"prefix": {"type": "str", "description": "Prefix"}},
        steps=[],
    )
    r2 = _recipe(tool_name="list_all_users", steps=[])

    rid1 = API_AGENT_STORE.save_recipe(
        api_id="rest:test|test",
        schema_hash="s",
        question="get users starting with A",
        recipe=r1,
        tool_name="get_users_starting_with_a",
    )
    rid2 = API_AGENT_STORE.save_recipe(
        api_id="rest:test|test",
        schema_hash="s",
        question="list all users",
        recipe=r2,
        tool_name="list_all_users",
    )

    suggestions = [
        _recipe_suggestion(
            recipe_id=rid1,
            score=0.85,
            question="get users starting with A",
            params={"prefix": {"type": "str", "description": "Prefix"}},
            recipe=r1,
            tool_name="get_users_starting_with_a",
        ),
        _recipe_suggestion(
            recipe_id=rid2,
            score=0.72,
            question="list all users",
            recipe=r2,
            tool_name="list_all_users",
        ),
    ]
    result = build_recipe_context(suggestions)

    assert "<recipes>" in result
    assert "</recipes>" in result
    assert "Score: 0.85" in result
    assert "get users starting with A" in result
    assert "prefix: str" in result
    assert "Score: 0.72" in result
    assert "list all users" in result


def test_build_recipe_context_no_params():
    """Recipes without params show empty param list."""
    r = _recipe(tool_name="simple_query", steps=[])
    rid = API_AGENT_STORE.save_recipe(
        api_id="rest:test|test",
        schema_hash="s",
        question="simple query",
        recipe=r,
        tool_name="simple_query",
    )

    suggestions = [
        _recipe_suggestion(recipe_id=rid, score=0.90, question="simple query", recipe=r),
    ]
    result = build_recipe_context(suggestions)
    # Tool name with no params should have empty signature
    assert "simple_query()" in result or "simple_query\n" in result


def test_build_recipe_context_enhanced_format():
    """Enhanced context shows tool names, score hints, step summaries."""
    # Create mock recipe in store
    # Using global API_AGENT_STORE
    recipe = _recipe(
        tool_name="get_users_recent_posts",
        tool_args={"user_id": {"type": "int", "description": "User id"}},
        steps=[
            _rest_step("users", "/users"),
            _sql_step("active_users", "SELECT * FROM users WHERE active = true"),
        ],
    )
    recipe_id = API_AGENT_STORE.save_recipe(
        api_id="rest:test|test",
        schema_hash="test_hash",
        question="Get user's recent posts",
        recipe=recipe,
        tool_name="get_users_recent_posts",
    )

    suggestions = [
        _recipe_suggestion(
            recipe_id=recipe_id,
            score=0.85,
            question="Get user's recent posts",
            params={"user_id": {"type": "int", "description": "User id"}},
            recipe=recipe,
        )
    ]

    result = build_recipe_context(suggestions)

    # Check new format elements
    assert "Available recipe tools" in result
    assert "get_users_recent_posts" in result  # Sanitized tool name
    assert "user_id: int" in result  # Typed param signature
    assert "Score: 0.85" in result
    assert "STRONG MATCH" in result  # Score >= 0.8
    assert "1 API call + 1 SQL step" in result  # Step summary


def test_build_recipe_context_score_hints():
    """Test different score interpretation hints."""
    # Using global API_AGENT_STORE
    recipe = _recipe(steps=[])

    # High score
    rid1 = API_AGENT_STORE.save_recipe(
        api_id="rest:a|b",
        schema_hash="s",
        question="high score query",
        recipe=recipe,
        tool_name="high_score_query",
    )
    # Medium score
    rid2 = API_AGENT_STORE.save_recipe(
        api_id="rest:a|b",
        schema_hash="s",
        question="medium score query",
        recipe=recipe,
        tool_name="medium_score_query",
    )
    # Low score
    rid3 = API_AGENT_STORE.save_recipe(
        api_id="rest:a|b",
        schema_hash="s",
        question="low score query",
        recipe=recipe,
        tool_name="low_score_query",
    )

    suggestions = [
        _recipe_suggestion(recipe_id=rid1, score=0.92, question="high score query", recipe=recipe),
        _recipe_suggestion(
            recipe_id=rid2, score=0.68, question="medium score query", recipe=recipe
        ),
        _recipe_suggestion(recipe_id=rid3, score=0.45, question="low score query", recipe=recipe),
    ]

    result = build_recipe_context(suggestions)

    assert "STRONG MATCH - highly recommended" in result
    assert "Good match - verify params" in result
    assert "Possible match - check alignment" in result


def test_build_recipe_context_step_summaries():
    """Test step summary formatting."""
    # Using global API_AGENT_STORE

    # API only
    r1 = _recipe(tool_name="api_only", steps=[_rest_step("users", "/users")])
    # SQL only
    r2 = _recipe(tool_name="sql_only", steps=[_sql_step("rows", "SELECT * FROM t")])
    # Both
    r3 = _recipe(
        tool_name="both",
        steps=[
            _rest_step("users", "/users"),
            _rest_step("posts", "/posts"),
            _sql_step("sql1", "SQL1"),
            _sql_step("sql2", "SQL2"),
        ],
    )
    # Neither
    r4 = _recipe(tool_name="neither", steps=[])

    rid1 = API_AGENT_STORE.save_recipe(
        api_id="rest:a|b",
        schema_hash="s",
        question="api only",
        recipe=r1,
        tool_name="api_only",
    )
    rid2 = API_AGENT_STORE.save_recipe(
        api_id="rest:a|b",
        schema_hash="s",
        question="sql only",
        recipe=r2,
        tool_name="sql_only",
    )
    rid3 = API_AGENT_STORE.save_recipe(
        api_id="rest:a|b",
        schema_hash="s",
        question="both",
        recipe=r3,
        tool_name="both",
    )
    rid4 = API_AGENT_STORE.save_recipe(
        api_id="rest:a|b",
        schema_hash="s",
        question="neither",
        recipe=r4,
        tool_name="neither",
    )

    suggestions = [
        _recipe_suggestion(recipe_id=rid1, score=0.7, question="api only", recipe=r1),
        _recipe_suggestion(recipe_id=rid2, score=0.7, question="sql only", recipe=r2),
        _recipe_suggestion(recipe_id=rid3, score=0.7, question="both", recipe=r3),
        _recipe_suggestion(recipe_id=rid4, score=0.7, question="neither", recipe=r4),
    ]

    result = build_recipe_context(suggestions)

    assert "1 API call" in result
    assert "1 SQL step" in result
    assert "2 API calls + 2 SQL steps" in result
    assert "no steps" in result


@pytest.mark.asyncio
async def test_validate_and_prepare_recipe_success():
    """async_validate_and_prepare_recipe returns recipe and params."""
    from contextvars import ContextVar

    from api_agent.recipe.learning import async_validate_and_prepare_recipe
    from api_agent.store import API_AGENT_STORE

    schema_var: ContextVar[str] = ContextVar("schema")
    schema_var.set('{"type": "test"}')

    recipe = _recipe(
        tool_name="get_users",
        tool_args={"limit": {"type": "int", "description": "Limit"}},
        steps=[
            _graphql_step(
                "users",
                "{ users(limit: {{limit}}) { id } }",
                {"limit": {"value": "limit"}},
            )
        ],
    )
    rid = API_AGENT_STORE.save_recipe(
        api_id="graphql:test",
        schema_hash="abc",
        question="get users",
        recipe=recipe,
        tool_name="get_users",
    )

    result, params, error = await async_validate_and_prepare_recipe(rid, '{"limit": 5}', schema_var)
    assert error == ""
    assert result is not None
    assert params == {"limit": 5}


@pytest.mark.asyncio
async def test_validate_and_prepare_recipe_not_found():
    """async_validate_and_prepare_recipe returns error for missing recipe."""
    from contextvars import ContextVar

    from api_agent.recipe.learning import async_validate_and_prepare_recipe

    schema_var: ContextVar[str] = ContextVar("schema")
    schema_var.set('{"type": "test"}')

    result, params, error = await async_validate_and_prepare_recipe("nonexistent", "{}", schema_var)
    assert result is None
    assert params is None
    assert "not found" in error


@pytest.mark.asyncio
async def test_validate_and_prepare_recipe_no_schema():
    """async_validate_and_prepare_recipe returns error when schema not loaded."""
    from contextvars import ContextVar

    from api_agent.recipe.learning import async_validate_and_prepare_recipe

    schema_var: ContextVar[str] = ContextVar("schema")  # Not set

    result, params, error = await async_validate_and_prepare_recipe("r_123", "{}", schema_var)
    assert result is None
    assert "schema not loaded" in error


@pytest.mark.asyncio
async def test_execute_recipe_steps_returns_executed_sql():
    """execute_recipe_steps returns executed SQL list."""
    from contextvars import ContextVar

    from api_agent.recipe.execution import execute_recipe_steps

    query_results: ContextVar[dict] = ContextVar("qr")
    last_result: ContextVar[list] = ContextVar("lr")
    query_results.set({"data": [{"id": 1, "name": "test"}]})
    last_result.set([None])

    recipe = _recipe(
        steps=[
            _sql_step("data_rows", "SELECT * FROM data"),
            _sql_step("first_data_row", "SELECT id FROM data WHERE id = 1"),
        ],
    )

    executed_items: list = []

    async def mock_executor(idx, step, params, results):
        return True, {"mock": "data"}, "", {"call": idx}

    success, last_data, executed_sql, error = await execute_recipe_steps(
        recipe,
        {},
        query_results,
        last_result,
        mock_executor,
        executed_items,
    )

    assert success is True
    assert error == ""
    assert len(executed_sql) == 2
    assert executed_sql[0] == "SELECT * FROM data"
    assert executed_sql[1] == "SELECT id FROM data WHERE id = 1"


@pytest.mark.asyncio
async def test_execute_recipe_steps_with_api_and_sql():
    """execute_recipe_steps executes both API and SQL steps."""
    from contextvars import ContextVar

    from api_agent.recipe.execution import execute_recipe_steps

    query_results: ContextVar[dict] = ContextVar("qr")
    last_result: ContextVar[list] = ContextVar("lr")
    query_results.set({})
    last_result.set([None])

    recipe = _recipe(
        steps=[
            {"kind": "test", "name": "step1"},
            _sql_step("step1_rows", "SELECT * FROM step1"),
        ],
    )

    executed_items: list = []
    executor_calls: list = []

    async def mock_executor(idx, step, params, results):
        executor_calls.append(step)
        results["step1"] = [{"id": 1}, {"id": 2}]
        return True, [{"id": 1}, {"id": 2}], "", {"call": "step1"}

    success, last_data, executed_sql, error = await execute_recipe_steps(
        recipe,
        {},
        query_results,
        last_result,
        mock_executor,
        executed_items,
    )

    assert success is True
    assert len(executor_calls) == 1
    assert len(executed_items) == 1
    assert len(executed_sql) == 1
    assert last_data == [{"id": 1}, {"id": 2}]  # SQL result


@pytest.mark.asyncio
async def test_execute_recipe_steps_preserves_interleaved_order():
    """SQL steps can run between API steps and expose named SQL results."""
    from contextvars import ContextVar

    from api_agent.recipe.execution import execute_recipe_steps

    query_results: ContextVar[dict] = ContextVar("qr")
    last_result: ContextVar[list] = ContextVar("lr")
    query_results.set({})
    last_result.set([None])

    recipe = _recipe(
        steps=[
            {"kind": "test", "name": "source"},
            _sql_step("filtered", "SELECT id FROM source WHERE id = 2"),
            {"kind": "test", "name": "after_sql"},
        ],
    )

    executed_items: list = []
    executor_calls: list = []

    async def mock_executor(idx, step, params, results):
        executor_calls.append(step["name"])
        if step["name"] == "source":
            results["source"] = [{"id": 1}, {"id": 2}]
            return True, results["source"], "", {"call": "source"}
        assert results["filtered"] == [{"id": 2}]
        results["after_sql"] = [{"ok": True}]
        return True, results["after_sql"], "", {"call": "after_sql"}

    success, last_data, executed_sql, error = await execute_recipe_steps(
        recipe,
        {},
        query_results,
        last_result,
        mock_executor,
        executed_items,
    )

    assert success is True
    assert error == ""
    assert executor_calls == ["source", "after_sql"]
    assert executed_items == [{"call": "source"}, {"call": "after_sql"}]
    assert executed_sql == ["SELECT id FROM source WHERE id = 2"]
    assert last_data == [{"ok": True}]


@pytest.mark.asyncio
async def test_execute_recipe_steps_allows_mapped_step_executor_to_resolve_bindings():
    from contextvars import ContextVar

    from api_agent.recipe.execution import build_step_input_sets, execute_recipe_steps

    query_results: ContextVar[dict] = ContextVar("qr")
    last_result: ContextVar[list] = ContextVar("lr")
    query_results.set({})
    last_result.set([None])

    mapped_step = {
        "kind": "test",
        "name": "key_results",
        "input": {
            "mode": "map",
            "from": "filtered_objectives",
            "bind": {"objective_id": "id"},
        },
    }
    recipe = _recipe(
        steps=[
            {"kind": "test", "name": "objectives"},
            _sql_step("filtered_objectives", "SELECT id FROM objectives WHERE id > 1 ORDER BY id"),
            mapped_step,
        ],
    )

    executor_params: list[dict] = []

    async def mock_executor(_idx, step, params, results):
        if step["name"] == "objectives":
            results["objectives"] = [{"id": 1}, {"id": 2}, {"id": 3}]
            return True, results["objectives"], "", {"call": "objectives"}
        input_sets, input_error = build_step_input_sets(step, params, results)
        assert input_error == ""
        executor_params.extend([input_set.params for input_set in input_sets])
        return True, [{"objective_id": 2}, {"objective_id": 3}], "", {"call": "key_results"}

    success, last_data, executed_sql, error = await execute_recipe_steps(
        recipe,
        {},
        query_results,
        last_result,
        mock_executor,
        [],
    )

    assert success is True
    assert error == ""
    assert executed_sql == ["SELECT id FROM objectives WHERE id > 1 ORDER BY id"]
    assert executor_params == [{"objective_id": 2}, {"objective_id": 3}]
    assert last_data == [{"objective_id": 2}, {"objective_id": 3}]


@pytest.mark.asyncio
async def test_execute_recipe_steps_api_failure():
    """execute_recipe_steps returns empty sql on API failure."""
    from contextvars import ContextVar

    from api_agent.recipe.execution import execute_recipe_steps

    query_results: ContextVar[dict] = ContextVar("qr")
    last_result: ContextVar[list] = ContextVar("lr")
    query_results.set({})
    last_result.set([None])

    recipe = _recipe(steps=[{"kind": "test"}])

    async def failing_executor(idx, step, params, results):
        return False, None, '{"error": "api failed"}', None

    success, last_data, executed_sql, error = await execute_recipe_steps(
        recipe,
        {},
        query_results,
        last_result,
        failing_executor,
        [],
    )

    assert success is False
    assert executed_sql == []
    assert "api failed" in error


@pytest.mark.asyncio
async def test_execute_recipe_steps_api_failure_after_sql_keeps_executed_sql():
    from contextvars import ContextVar

    from api_agent.recipe.execution import execute_recipe_steps

    query_results: ContextVar[dict] = ContextVar("qr")
    last_result: ContextVar[list] = ContextVar("lr")
    query_results.set({"data": [{"id": 1}]})
    last_result.set([None])

    recipe = _recipe(
        steps=[
            _sql_step("data_ids", "SELECT id FROM data"),
            {"kind": "test"},
        ],
    )

    async def failing_executor(idx, step, params, results):
        return False, None, '{"error": "api failed"}', None

    success, last_data, executed_sql, error = await execute_recipe_steps(
        recipe,
        {},
        query_results,
        last_result,
        failing_executor,
        [],
    )

    assert success is False
    assert last_data is None
    assert executed_sql == ["SELECT id FROM data"]
    assert "api failed" in error


# --- sanitize_tool_name tests ---


class TestSanitizeToolName:
    def test_basic(self):
        from api_agent.recipe.naming import sanitize_tool_name

        assert sanitize_tool_name("get_users") == "get_users"

    def test_special_chars_stripped(self):
        from api_agent.recipe.naming import sanitize_tool_name

        assert sanitize_tool_name("get-users!@#beta") == "get_users_beta"

    def test_spaces_to_underscores(self):
        from api_agent.recipe.naming import sanitize_tool_name

        assert sanitize_tool_name("get users by name") == "get_users_by_name"

    def test_empty_returns_recipe(self):
        from api_agent.recipe.naming import sanitize_tool_name

        assert sanitize_tool_name("") == "recipe"
        assert sanitize_tool_name(None) == "recipe"

    def test_uppercase_lowered(self):
        from api_agent.recipe.naming import sanitize_tool_name

        assert sanitize_tool_name("GetUsers") == "getusers"

    def test_leading_trailing_underscores_stripped(self):
        from api_agent.recipe.naming import sanitize_tool_name

        assert sanitize_tool_name("  _hello_  ") == "hello"

    def test_reserved_prefixes_removed(self):
        from api_agent.recipe.naming import sanitize_tool_name

        assert sanitize_tool_name("r_list_users") == "list_users"
        assert sanitize_tool_name("api_get_users") == "get_users"

    def test_digit_prefix_gets_safe_slug(self):
        from api_agent.recipe.naming import sanitize_tool_name

        assert sanitize_tool_name("123 hotels") == "recipe_123_hotels"
