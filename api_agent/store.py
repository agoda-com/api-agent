"""Storage for API Agent recipes and derived API metadata."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Protocol, TypeVar, cast

from rapidfuzz import fuzz

from .config import settings
from .recipe.contracts import (
    get_recipe_description,
    get_recipe_steps,
    get_recipe_tool_args,
    get_recipe_tool_name,
)
from .recipe.identity import recipe_fingerprint, sha256_hex
from .recipe.naming import sanitize_tool_name

logger = logging.getLogger(__name__)
StoreResult = TypeVar("StoreResult")


def _log_recipe(msg: str) -> None:
    if settings.DEBUG:
        logger.info(f"[Recipe] {msg}")


def _normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def _tokens(q: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _normalize_question(q)))


def _similarity(query: str, signature: str) -> float:
    """Similarity score using RapidFuzz token-based matching."""
    q_norm = _normalize_question(query)
    s_norm = _normalize_question(signature)
    if not q_norm or not s_norm:
        return 0.0
    if q_norm == s_norm:
        return 1.0

    q_tokens = _tokens(query)
    s_tokens = _tokens(signature)
    if not q_tokens or not s_tokens:
        return 0.0

    q_text = " ".join(sorted(q_tokens))
    s_text = " ".join(sorted(s_tokens))
    base = fuzz.token_set_ratio(q_text, s_text)

    partial_fn = getattr(fuzz, "partial_token_set_ratio", None)
    extra: float = (
        partial_fn(q_text, s_text) if callable(partial_fn) else fuzz.WRatio(q_text, s_text)
    )

    overlap = len(q_tokens & s_tokens) / max(len(q_tokens), 1)
    coverage = len(s_tokens & q_tokens) / max(len(s_tokens), 1)
    token_balance = min(overlap, coverage) * 100.0

    return (0.55 * base + 0.25 * extra + 0.20 * token_balance) / 100.0


@dataclass
class RecipeRecord:
    recipe_id: str
    api_id: str
    schema_hash: str
    question: str
    question_sig: str
    question_tokens: set[str]
    recipe: dict[str, Any]
    tool_name: str
    fingerprint: str
    created_at: float
    last_used_at: float
    insertion_index: int
    enabled: bool = True


class ApiAgentStoreProtocol(Protocol):
    def save_or_touch(
        self,
        *,
        api_id: str,
        schema_hash: str,
        question: str,
        recipe: dict[str, Any],
        tool_name: str,
    ) -> str: ...

    def save_recipe(
        self,
        *,
        api_id: str,
        schema_hash: str,
        question: str,
        recipe: dict[str, Any],
        tool_name: str,
    ) -> str: ...

    def get_recipe(self, recipe_id: str) -> dict[str, Any] | None: ...

    def get_recipe_meta(self, recipe_id: str) -> dict[str, Any] | None: ...

    def list_recipes(
        self, *, api_id: str, schema_hash: str, include_disabled: bool = False
    ) -> list[dict[str, Any]]: ...

    def suggest_recipes(
        self, *, api_id: str, schema_hash: str, question: str, k: int = 3
    ) -> list[dict[str, Any]]: ...

    def find_recipe_by_tool_slug(
        self, *, api_id: str, schema_hash: str, tool_slug: str, max_slug_len: int | None = None
    ) -> dict[str, Any] | None: ...

    def get_downstream_description(self, *, api_id: str, schema_hash: str) -> str | None: ...

    def save_downstream_description(
        self, *, api_id: str, schema_hash: str, description: str, ttl_seconds: int | None = None
    ) -> None: ...

    def disable_recipe(self, recipe_id: str) -> bool: ...

    def delete_recipe(self, recipe_id: str) -> bool: ...

    def merge_recipes(self, source_id: str, target_id: str) -> bool: ...


class MemoryApiAgentStore:
    """Thread-safe API Agent store with recipe and downstream description caches."""

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max(1, int(max_size))
        self._lock = threading.Lock()
        self._records: dict[str, RecipeRecord] = {}
        self._by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._by_fingerprint: dict[str, str] = {}
        self._downstream_descriptions: dict[tuple[str, str], tuple[str, float | None]] = {}
        self._fifo: list[str] = []
        self._next_insertion = 0

    def save_or_touch(
        self,
        *,
        api_id: str,
        schema_hash: str,
        question: str,
        recipe: dict[str, Any],
        tool_name: str,
    ) -> str:
        fingerprint = recipe_fingerprint(api_id=api_id, schema_hash=schema_hash, recipe=recipe)
        now = time.time()

        with self._lock:
            existing_id = self._by_fingerprint.get(fingerprint)
            if existing_id and existing_id in self._records:
                rec = self._records[existing_id]
                rec.question = question
                rec.question_sig = _normalize_question(question)
                rec.question_tokens = _tokens(question)
                rec.recipe = dict(recipe)
                rec.tool_name = tool_name
                rec.last_used_at = now
                _log_recipe(f"TOUCH {existing_id} fingerprint={fingerprint[:8]}")
                return existing_id

            recipe_id = f"r_{uuid.uuid4().hex[:8]}"
            self._next_insertion += 1
            record = RecipeRecord(
                recipe_id=recipe_id,
                api_id=api_id,
                schema_hash=schema_hash,
                question=question,
                question_sig=_normalize_question(question),
                question_tokens=_tokens(question),
                recipe=dict(recipe),
                tool_name=tool_name,
                fingerprint=fingerprint,
                created_at=now,
                last_used_at=now,
                insertion_index=self._next_insertion,
            )
            self._records[recipe_id] = record
            self._by_key[(api_id, schema_hash)].add(recipe_id)
            self._by_fingerprint[fingerprint] = recipe_id
            self._fifo.append(recipe_id)
            self._evict_if_needed()

        params = list(get_recipe_tool_args(recipe).keys())
        _log_recipe(f"SAVE {recipe_id} params={params} q={question[:40]}")
        return recipe_id

    def save_recipe(
        self,
        *,
        api_id: str,
        schema_hash: str,
        question: str,
        recipe: dict[str, Any],
        tool_name: str,
    ) -> str:
        return self.save_or_touch(
            api_id=api_id,
            schema_hash=schema_hash,
            question=question,
            recipe=recipe,
            tool_name=tool_name,
        )

    def get_recipe(self, recipe_id: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self._records.get(recipe_id)
            if not rec:
                return None
            rec.last_used_at = time.time()
            return dict(rec.recipe)

    def get_recipe_meta(self, recipe_id: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self._records.get(recipe_id)
            if not rec:
                return None
            rec.last_used_at = time.time()
            return _record_to_meta(rec)

    def suggest_recipes(
        self,
        *,
        api_id: str,
        schema_hash: str,
        question: str,
        k: int = 3,
    ) -> list[dict[str, Any]]:
        q_sig = _normalize_question(question)
        with self._lock:
            recs = self._records_for_key(api_id, schema_hash, include_disabled=False)

        scored: list[tuple[float, RecipeRecord]] = []
        for r in recs:
            score = _similarity(q_sig, r.question_sig)
            if score > 0:
                scored.append((score, r))

        scored.sort(key=lambda t: (t[0], t[1].last_used_at), reverse=True)
        out = [_record_to_suggestion(r, score) for score, r in scored[: max(0, int(k))]]

        if out:
            matches = " ".join(f"{s['recipe_id']}({s['score']:.2f})" for s in out)
            _log_recipe(f"SUGGEST found={len(out)} [{matches}]")
        return out

    def list_recipes(
        self,
        *,
        api_id: str,
        schema_hash: str,
        include_disabled: bool = False,
    ) -> list[dict[str, Any]]:
        with self._lock:
            recs = self._records_for_key(api_id, schema_hash, include_disabled=include_disabled)
        recs.sort(key=lambda r: r.last_used_at, reverse=True)
        return [_record_to_public(r) for r in recs]

    def find_recipe_by_tool_slug(
        self,
        *,
        api_id: str,
        schema_hash: str,
        tool_slug: str,
        max_slug_len: int | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            recs = self._records_for_key(api_id, schema_hash, include_disabled=False)

        matches = []
        for rec in recs:
            slug = sanitize_tool_name(rec.tool_name)
            if isinstance(max_slug_len, int) and max_slug_len > 0:
                slug = slug[:max_slug_len]
            if slug == tool_slug:
                matches.append(rec)

        if not matches:
            return None
        matches.sort(key=lambda r: (r.last_used_at, r.created_at), reverse=True)
        return _record_to_public(matches[0])

    def get_downstream_description(self, *, api_id: str, schema_hash: str) -> str | None:
        key = (api_id, schema_hash)
        with self._lock:
            cached = self._downstream_descriptions.get(key)
            if not cached:
                return None
            description, expires_at = cached
            if expires_at is not None and expires_at <= time.time():
                self._downstream_descriptions.pop(key, None)
                return None
            return description

    def save_downstream_description(
        self, *, api_id: str, schema_hash: str, description: str, ttl_seconds: int | None = None
    ) -> None:
        expires_at = time.time() + ttl_seconds if ttl_seconds is not None else None
        with self._lock:
            self._downstream_descriptions[(api_id, schema_hash)] = (description, expires_at)

    def disable_recipe(self, recipe_id: str) -> bool:
        with self._lock:
            rec = self._records.get(recipe_id)
            if not rec:
                return False
            rec.enabled = False
            rec.last_used_at = time.time()
            return True

    def delete_recipe(self, recipe_id: str) -> bool:
        with self._lock:
            return self._delete(recipe_id)

    def merge_recipes(self, source_id: str, target_id: str) -> bool:
        if source_id == target_id:
            return False
        with self._lock:
            if source_id not in self._records or target_id not in self._records:
                return False
            self._records[target_id].last_used_at = time.time()
            return self._delete(source_id)

    def _records_for_key(
        self, api_id: str, schema_hash: str, *, include_disabled: bool
    ) -> list[RecipeRecord]:
        ids = list(self._by_key.get((api_id, schema_hash), set()))
        return [
            self._records[i]
            for i in ids
            if i in self._records and (include_disabled or self._records[i].enabled)
        ]

    def _evict_if_needed(self) -> None:
        while len(self._records) > self._max_size and self._fifo:
            oldest_id = self._fifo.pop(0)
            self._delete(oldest_id)

    def _delete(self, recipe_id: str) -> bool:
        rec = self._records.pop(recipe_id, None)
        if not rec:
            return False
        self._by_fingerprint.pop(rec.fingerprint, None)
        self._fifo = [rid for rid in self._fifo if rid != recipe_id]
        key = (rec.api_id, rec.schema_hash)
        ids = self._by_key.get(key)
        if ids:
            ids.discard(recipe_id)
            if not ids:
                self._by_key.pop(key, None)
        return True


class RedisApiAgentStore:
    """Redis-backed API Agent store with the same contract as MemoryApiAgentStore."""

    def __init__(self, url: str, *, namespace: str = "api-agent", max_size: int = 1000) -> None:
        if not url:
            raise ValueError("Redis API Agent store requires [redis].url")
        import redis

        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._namespace = namespace
        self._max_size = max(1, int(max_size))

    def save_or_touch(
        self,
        *,
        api_id: str,
        schema_hash: str,
        question: str,
        recipe: dict[str, Any],
        tool_name: str,
    ) -> str:
        fingerprint = recipe_fingerprint(api_id=api_id, schema_hash=schema_hash, recipe=recipe)
        now = time.time()
        existing_id = cast(str | None, self._client.get(self._fingerprint_key(fingerprint)))
        if existing_id:
            rec = self._load_record(existing_id)
            if rec:
                rec.question = question
                rec.question_sig = _normalize_question(question)
                rec.question_tokens = _tokens(question)
                rec.recipe = dict(recipe)
                rec.tool_name = tool_name
                rec.last_used_at = now
                self._save_record(rec)
                return str(existing_id)

        recipe_id = f"r_{uuid.uuid4().hex[:8]}"
        insertion_index = int(self._client.incr(self._key("next_insertion")))
        rec = RecipeRecord(
            recipe_id=recipe_id,
            api_id=api_id,
            schema_hash=schema_hash,
            question=question,
            question_sig=_normalize_question(question),
            question_tokens=_tokens(question),
            recipe=dict(recipe),
            tool_name=tool_name,
            fingerprint=fingerprint,
            created_at=now,
            last_used_at=now,
            insertion_index=insertion_index,
        )
        pipe = self._client.pipeline()
        pipe.set(self._record_key(recipe_id), json.dumps(_record_to_json(rec), sort_keys=True))
        pipe.set(self._fingerprint_key(fingerprint), recipe_id)
        pipe.sadd(self._all_key(), recipe_id)
        pipe.sadd(self._index_key(api_id, schema_hash), recipe_id)
        pipe.rpush(self._fifo_key(), recipe_id)
        pipe.execute()
        self._evict_if_needed()
        return recipe_id

    def save_recipe(
        self,
        *,
        api_id: str,
        schema_hash: str,
        question: str,
        recipe: dict[str, Any],
        tool_name: str,
    ) -> str:
        return self.save_or_touch(
            api_id=api_id,
            schema_hash=schema_hash,
            question=question,
            recipe=recipe,
            tool_name=tool_name,
        )

    def get_recipe(self, recipe_id: str) -> dict[str, Any] | None:
        rec = self._load_record(recipe_id)
        if not rec:
            return None
        rec.last_used_at = time.time()
        self._save_record(rec)
        return dict(rec.recipe)

    def get_recipe_meta(self, recipe_id: str) -> dict[str, Any] | None:
        rec = self._load_record(recipe_id)
        if not rec:
            return None
        rec.last_used_at = time.time()
        self._save_record(rec)
        return _record_to_meta(rec)

    def suggest_recipes(
        self,
        *,
        api_id: str,
        schema_hash: str,
        question: str,
        k: int = 3,
    ) -> list[dict[str, Any]]:
        q_sig = _normalize_question(question)
        scored: list[tuple[float, RecipeRecord]] = []
        for rec in self._records_for_key(api_id, schema_hash, include_disabled=False):
            score = _similarity(q_sig, rec.question_sig)
            if score > 0:
                scored.append((score, rec))
        scored.sort(key=lambda t: (t[0], t[1].last_used_at), reverse=True)
        return [_record_to_suggestion(r, score) for score, r in scored[: max(0, int(k))]]

    def list_recipes(
        self,
        *,
        api_id: str,
        schema_hash: str,
        include_disabled: bool = False,
    ) -> list[dict[str, Any]]:
        recs = self._records_for_key(api_id, schema_hash, include_disabled=include_disabled)
        recs.sort(key=lambda r: r.last_used_at, reverse=True)
        return [_record_to_public(r) for r in recs]

    def find_recipe_by_tool_slug(
        self,
        *,
        api_id: str,
        schema_hash: str,
        tool_slug: str,
        max_slug_len: int | None = None,
    ) -> dict[str, Any] | None:
        matches = []
        for rec in self._records_for_key(api_id, schema_hash, include_disabled=False):
            slug = sanitize_tool_name(rec.tool_name)
            if isinstance(max_slug_len, int) and max_slug_len > 0:
                slug = slug[:max_slug_len]
            if slug == tool_slug:
                matches.append(rec)
        if not matches:
            return None
        matches.sort(key=lambda r: (r.last_used_at, r.created_at), reverse=True)
        return _record_to_public(matches[0])

    def get_downstream_description(self, *, api_id: str, schema_hash: str) -> str | None:
        return cast(
            str | None,
            self._client.get(self._downstream_description_key(api_id, schema_hash)),
        )

    def save_downstream_description(
        self, *, api_id: str, schema_hash: str, description: str, ttl_seconds: int | None = None
    ) -> None:
        key = self._downstream_description_key(api_id, schema_hash)
        if ttl_seconds is None:
            self._client.set(key, description)
        else:
            self._client.set(key, description, ex=ttl_seconds)

    def disable_recipe(self, recipe_id: str) -> bool:
        rec = self._load_record(recipe_id)
        if not rec:
            return False
        rec.enabled = False
        rec.last_used_at = time.time()
        self._save_record(rec)
        return True

    def delete_recipe(self, recipe_id: str) -> bool:
        rec = self._load_record(recipe_id)
        if not rec:
            return False
        pipe = self._client.pipeline()
        pipe.delete(self._record_key(recipe_id))
        pipe.delete(self._fingerprint_key(rec.fingerprint))
        pipe.srem(self._all_key(), recipe_id)
        pipe.srem(self._index_key(rec.api_id, rec.schema_hash), recipe_id)
        pipe.lrem(self._fifo_key(), 0, recipe_id)
        pipe.execute()
        return True

    def merge_recipes(self, source_id: str, target_id: str) -> bool:
        if source_id == target_id:
            return False
        target = self._load_record(target_id)
        if not target or not self._load_record(source_id):
            return False
        target.last_used_at = time.time()
        self._save_record(target)
        return self.delete_recipe(source_id)

    def _records_for_key(
        self, api_id: str, schema_hash: str, *, include_disabled: bool
    ) -> list[RecipeRecord]:
        ids = list(
            cast(set[str], self._client.smembers(self._index_key(api_id, schema_hash)) or set())
        )
        recs = []
        for recipe_id in ids:
            rec = self._load_record(str(recipe_id))
            if rec and (include_disabled or rec.enabled):
                recs.append(rec)
        return recs

    def _evict_if_needed(self) -> None:
        while int(cast(int, self._client.scard(self._all_key()) or 0)) > self._max_size:
            oldest_id = cast(str | None, self._client.lpop(self._fifo_key()))
            if not oldest_id:
                return
            self.delete_recipe(str(oldest_id))

    def _load_record(self, recipe_id: str) -> RecipeRecord | None:
        raw = cast(str | None, self._client.get(self._record_key(recipe_id)))
        if not raw:
            return None
        try:
            return _record_from_json(json.loads(raw))
        except Exception:
            logger.exception("Invalid recipe record in Redis: %s", recipe_id)
            return None

    def _save_record(self, rec: RecipeRecord) -> None:
        self._client.set(self._record_key(rec.recipe_id), json.dumps(_record_to_json(rec)))

    def _key(self, suffix: str) -> str:
        return f"{self._namespace}:{suffix}"

    def _record_key(self, recipe_id: str) -> str:
        return self._key(f"recipe:{recipe_id}")

    def _fingerprint_key(self, fingerprint: str) -> str:
        return self._key(f"recipe_fingerprint:{fingerprint}")

    def _index_key(self, api_id: str, schema_hash: str) -> str:
        digest = sha256_hex(api_id)
        return self._key(f"recipes:{digest}:{schema_hash}")

    def _downstream_description_key(self, api_id: str, schema_hash: str) -> str:
        digest = sha256_hex(api_id)
        return self._key(f"downstream_description:{digest}:{schema_hash}")

    def _all_key(self) -> str:
        return self._key("recipes:all")

    def _fifo_key(self) -> str:
        return self._key("recipes:fifo")


class AsyncApiAgentStore:
    """Async request-path adapter for synchronous store implementations."""

    def __init__(self, store: ApiAgentStoreProtocol) -> None:
        self._store = store

    async def save_or_touch(
        self,
        *,
        api_id: str,
        schema_hash: str,
        question: str,
        recipe: dict[str, Any],
        tool_name: str,
    ) -> str:
        return await self._call(
            self._store.save_or_touch,
            api_id=api_id,
            schema_hash=schema_hash,
            question=question,
            recipe=recipe,
            tool_name=tool_name,
        )

    async def save_recipe(
        self,
        *,
        api_id: str,
        schema_hash: str,
        question: str,
        recipe: dict[str, Any],
        tool_name: str,
    ) -> str:
        return await self._call(
            self._store.save_recipe,
            api_id=api_id,
            schema_hash=schema_hash,
            question=question,
            recipe=recipe,
            tool_name=tool_name,
        )

    async def get_recipe(self, recipe_id: str) -> dict[str, Any] | None:
        return await self._call(self._store.get_recipe, recipe_id)

    async def get_recipe_meta(self, recipe_id: str) -> dict[str, Any] | None:
        return await self._call(self._store.get_recipe_meta, recipe_id)

    async def list_recipes(
        self,
        *,
        api_id: str,
        schema_hash: str,
        include_disabled: bool = False,
    ) -> list[dict[str, Any]]:
        return await self._call(
            self._store.list_recipes,
            api_id=api_id,
            schema_hash=schema_hash,
            include_disabled=include_disabled,
        )

    async def suggest_recipes(
        self,
        *,
        api_id: str,
        schema_hash: str,
        question: str,
        k: int = 3,
    ) -> list[dict[str, Any]]:
        return await self._call(
            self._store.suggest_recipes,
            api_id=api_id,
            schema_hash=schema_hash,
            question=question,
            k=k,
        )

    async def find_recipe_by_tool_slug(
        self,
        *,
        api_id: str,
        schema_hash: str,
        tool_slug: str,
        max_slug_len: int | None = None,
    ) -> dict[str, Any] | None:
        return await self._call(
            self._store.find_recipe_by_tool_slug,
            api_id=api_id,
            schema_hash=schema_hash,
            tool_slug=tool_slug,
            max_slug_len=max_slug_len,
        )

    async def get_downstream_description(self, *, api_id: str, schema_hash: str) -> str | None:
        return await self._call(
            self._store.get_downstream_description,
            api_id=api_id,
            schema_hash=schema_hash,
        )

    async def save_downstream_description(
        self, *, api_id: str, schema_hash: str, description: str, ttl_seconds: int | None = None
    ) -> None:
        await self._call(
            self._store.save_downstream_description,
            api_id=api_id,
            schema_hash=schema_hash,
            description=description,
            ttl_seconds=ttl_seconds,
        )

    async def disable_recipe(self, recipe_id: str) -> bool:
        return await self._call(self._store.disable_recipe, recipe_id)

    async def delete_recipe(self, recipe_id: str) -> bool:
        return await self._call(self._store.delete_recipe, recipe_id)

    async def merge_recipes(self, source_id: str, target_id: str) -> bool:
        return await self._call(self._store.merge_recipes, source_id, target_id)

    async def _call(
        self,
        method: Callable[..., StoreResult],
        *args: Any,
        **kwargs: Any,
    ) -> StoreResult:
        if isinstance(self._store, RedisApiAgentStore):
            return await asyncio.to_thread(partial(method, *args, **kwargs))
        return method(*args, **kwargs)


def _record_to_json(rec: RecipeRecord) -> dict[str, Any]:
    data = asdict(rec)
    data["question_tokens"] = sorted(rec.question_tokens)
    return data


def _record_from_json(data: dict[str, Any]) -> RecipeRecord:
    return RecipeRecord(
        recipe_id=str(data["recipe_id"]),
        api_id=str(data["api_id"]),
        schema_hash=str(data["schema_hash"]),
        question=str(data["question"]),
        question_sig=str(data.get("question_sig") or _normalize_question(data["question"])),
        question_tokens=set(data.get("question_tokens") or _tokens(str(data["question"]))),
        recipe=dict(data.get("recipe") or {}),
        tool_name=str(data.get("tool_name") or "recipe"),
        fingerprint=str(data["fingerprint"]),
        created_at=float(data.get("created_at") or time.time()),
        last_used_at=float(data.get("last_used_at") or time.time()),
        insertion_index=int(data.get("insertion_index") or 0),
        enabled=bool(data.get("enabled", True)),
    )


def _record_to_meta(rec: RecipeRecord) -> dict[str, Any]:
    return {
        "recipe_id": rec.recipe_id,
        "api_id": rec.api_id,
        "schema_hash": rec.schema_hash,
        "question": rec.question,
        "tool_name": rec.tool_name,
        "description": get_recipe_description(rec.recipe),
        "recipe": dict(rec.recipe),
        "fingerprint": rec.fingerprint,
        "enabled": rec.enabled,
        "created_at": rec.created_at,
        "last_used_at": rec.last_used_at,
    }


def _record_to_suggestion(rec: RecipeRecord, score: float) -> dict[str, Any]:
    return {
        "recipe_id": rec.recipe_id,
        "score": round(score, 4),
        "created_at": rec.created_at,
        "last_used_at": rec.last_used_at,
        "question": rec.question,
        "tool_name": rec.tool_name,
        "description": get_recipe_description(rec.recipe),
    }


def _record_to_public(rec: RecipeRecord) -> dict[str, Any]:
    tool_args = get_recipe_tool_args(rec.recipe)
    steps = get_recipe_steps(rec.recipe)
    return {
        "recipe_id": rec.recipe_id,
        "question": rec.question,
        "tool_name": get_recipe_tool_name(rec.recipe) or rec.tool_name,
        "description": get_recipe_description(rec.recipe),
        "created_at": rec.created_at,
        "last_used_at": rec.last_used_at,
        "fingerprint": rec.fingerprint,
        "enabled": rec.enabled,
        "public_contract": dict(rec.recipe.get("public_contract", {})),
        "execution_plan": dict(rec.recipe.get("execution_plan", {})),
        "tool_args": dict(tool_args),
        "steps": list(steps),
    }


def create_api_agent_store() -> ApiAgentStoreProtocol:
    if settings.STORAGE_BACKEND.lower() == "redis":
        return RedisApiAgentStore(
            settings.REDIS_URL,
            namespace=settings.STORAGE_NAMESPACE,
            max_size=settings.RECIPE_CACHE_SIZE,
        )
    return MemoryApiAgentStore(max_size=settings.RECIPE_CACHE_SIZE)


API_AGENT_STORE = create_api_agent_store()
ASYNC_API_AGENT_STORE = AsyncApiAgentStore(API_AGENT_STORE)
