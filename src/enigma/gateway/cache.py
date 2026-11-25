"""SQLite response cache keyed by the canonical request hash + resolved provider/model.

Kept in its own database file (not the control DB): it is disposable derived data, and
deleting it never loses institutional records (the cost ledger holds the call history).
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from enigma.core.hashing import sha256_of
from enigma.gateway.types import ModelResponse

_SCHEMA = """
CREATE TABLE IF NOT EXISTS response_cache (
    key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def cache_key(request_hash: str, provider: str, model_id: str, registry_version: str) -> str:
    return sha256_of(
        {
            "request": request_hash,
            "provider": provider,
            "model_id": model_id,
            "registry": registry_version,
        }
    )


class ResponseCache:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)

    def get(self, key: str) -> ModelResponse | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM response_cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return ModelResponse.model_validate_json(row[0])

    def put(self, key: str, response: ModelResponse, now: datetime) -> None:
        body = response.model_copy(update={"parsed": None}).model_dump_json()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO response_cache(key, provider, model_id, response,"
                " created_at) VALUES (?,?,?,?,?)",
                (key, response.provider, response.model_id, body, now.isoformat()),
            )

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
