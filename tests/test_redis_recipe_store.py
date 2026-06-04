import threading

import pytest

from api_agent.store import AsyncApiAgentStore, RedisApiAgentStore


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.calls = []

    def set(self, *args):
        self.calls.append(("set", args))
        return self

    def sadd(self, *args):
        self.calls.append(("sadd", args))
        return self

    def rpush(self, *args):
        self.calls.append(("rpush", args))
        return self

    def delete(self, *args):
        self.calls.append(("delete", args))
        return self

    def srem(self, *args):
        self.calls.append(("srem", args))
        return self

    def lrem(self, *args):
        self.calls.append(("lrem", args))
        return self

    def execute(self):
        for name, args in self.calls:
            getattr(self.client, name)(*args)


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.expiries = {}
        self.sets = {}
        self.lists = {}
        self.counts = {}

    def pipeline(self):
        return FakePipeline(self)

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ex=None):
        self.values[key] = value
        if ex is not None:
            self.expiries[key] = ex

    def incr(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def srem(self, key, value):
        self.sets.setdefault(key, set()).discard(value)

    def scard(self, key):
        return len(self.sets.get(key, set()))

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    def lpop(self, key):
        values = self.lists.setdefault(key, [])
        return values.pop(0) if values else None

    def lrem(self, key, _count, value):
        self.lists[key] = [v for v in self.lists.get(key, []) if v != value]

    def delete(self, key):
        self.values.pop(key, None)


def _recipe(tool_name: str, query_template: str) -> dict:
    return {
        "public_contract": {
            "tool_name": tool_name,
            "description": f"Use for {tool_name}. Returns rows as CSV. No required params.",
            "tool_args": {},
        },
        "execution_plan": {
            "steps": [
                {
                    "id": tool_name,
                    "kind": "graphql",
                    "input": {"mode": "single", "with": {}},
                    "call": {"query_template": query_template},
                    "output": {"name": tool_name},
                }
            ],
        },
        "validation_fixture": {"tool_args": {}},
    }


def _redis_store(monkeypatch) -> tuple[RedisApiAgentStore, FakeRedis]:
    fake = FakeRedis()

    import redis

    monkeypatch.setattr(redis.Redis, "from_url", staticmethod(lambda *_args, **_kwargs: fake))
    return RedisApiAgentStore("redis://test", namespace="test", max_size=2), fake


def test_redis_recipe_store_idempotent_and_fifo(monkeypatch):
    store, _fake = _redis_store(monkeypatch)
    api_id = "graphql:https://api.example.com/graphql"
    recipes = [
        _recipe("a", "{ a }"),
        _recipe("b", "{ b }"),
        _recipe("c", "{ c }"),
    ]

    first_id = store.save_recipe(
        api_id=api_id, schema_hash="s", question="a", recipe=recipes[0], tool_name="a"
    )
    assert (
        store.save_recipe(
            api_id=api_id, schema_hash="s", question="a again", recipe=recipes[0], tool_name="a"
        )
        == first_id
    )
    second_id = store.save_recipe(
        api_id=api_id, schema_hash="s", question="b", recipe=recipes[1], tool_name="b"
    )
    third_id = store.save_recipe(
        api_id=api_id, schema_hash="s", question="c", recipe=recipes[2], tool_name="c"
    )

    ids = {r["recipe_id"] for r in store.list_recipes(api_id=api_id, schema_hash="s")}
    assert ids == {second_id, third_id}


def test_redis_recipe_store_caches_downstream_description(monkeypatch):
    store, _fake = _redis_store(monkeypatch)

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


def test_redis_recipe_store_sets_downstream_description_ttl(monkeypatch):
    store, fake = _redis_store(monkeypatch)

    store.save_downstream_description(
        api_id="rest:https://spec|https://api",
        schema_hash="schema-a",
        description="Query objectives and key results.",
        ttl_seconds=300,
    )

    key = store._downstream_description_key("rest:https://spec|https://api", "schema-a")
    assert fake.expiries[key] == 300


@pytest.mark.asyncio
async def test_redis_recipe_store_calls_are_offloaded(monkeypatch):
    store, _fake = _redis_store(monkeypatch)
    event_loop_thread_id = threading.get_ident()

    def list_recipes(*, api_id, schema_hash, include_disabled=False):
        return [
            {
                "api_id": api_id,
                "schema_hash": schema_hash,
                "include_disabled": include_disabled,
                "thread_id": threading.get_ident(),
            }
        ]

    monkeypatch.setattr(store, "list_recipes", list_recipes)

    result = await AsyncApiAgentStore(store).list_recipes(api_id="api", schema_hash="schema")

    assert result[0]["thread_id"] != event_loop_thread_id
