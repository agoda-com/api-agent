"""Process-local cache for externally loaded API schemas."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class CachedSchema:
    value: Any
    stale: bool


@dataclass(frozen=True)
class _Entry:
    value: Any
    stored_at: float


class SchemaCache:
    """Bounded schema cache with stale reads and per-key failure backoff."""

    def __init__(
        self, max_entries: int = 1000, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._max_entries = max(1, int(max_entries))
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._failures: OrderedDict[str, float] = OrderedDict()

    def get(
        self, key: str, *, ttl_seconds: float, stale_max_age_seconds: float
    ) -> CachedSchema | None:
        now = self._clock()
        ttl = max(0.0, ttl_seconds)
        stale_age = max(0.0, stale_max_age_seconds)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            age = max(0.0, now - entry.stored_at)
            if ttl > 0 and age < ttl:
                self._entries.move_to_end(key)
                return CachedSchema(deepcopy(entry.value), stale=False)
            if stale_age > 0 and age < ttl + stale_age:
                self._entries.move_to_end(key)
                return CachedSchema(deepcopy(entry.value), stale=True)
            self._entries.pop(key, None)
            return None

    def is_circuit_open(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            retry_at = self._failures.get(key)
            if retry_at is None:
                return False
            if retry_at <= now:
                self._failures.pop(key, None)
                return False
            self._failures.move_to_end(key)
            return True

    def record_success(self, key: str, value: Any) -> None:
        entry = _Entry(value=deepcopy(value), stored_at=self._clock())
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            self._failures.pop(key, None)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            while len(self._failures) > self._max_entries:
                self._failures.popitem(last=False)

    def record_failure(self, key: str, *, backoff_seconds: float) -> None:
        backoff = max(0.0, backoff_seconds)
        if backoff == 0:
            return
        with self._lock:
            self._failures[key] = self._clock() + backoff
            self._failures.move_to_end(key)
            while len(self._failures) > self._max_entries:
                self._failures.popitem(last=False)

    def clear(self) -> None:
        """Clear cached state; primarily useful for isolated tests."""
        with self._lock:
            self._entries.clear()
            self._failures.clear()


def schema_cache_key(api_type: str, url: str, headers: dict[str, Any] | None) -> str:
    """Hash the endpoint and effective auth headers without retaining their values."""
    normalized_headers = sorted(
        (
            str(name).lower(),
            json.dumps(value, sort_keys=True, separators=(",", ":"), default=str),
        )
        for name, value in (headers or {}).items()
    )
    material = json.dumps(
        [api_type, url, normalized_headers],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


schema_cache = SchemaCache()
