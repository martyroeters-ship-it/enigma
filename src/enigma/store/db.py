"""SQLite control database (WAL mode) with forward-only migrations.

ADR-0002: SQLite is the M1 system of record (zero infrastructure, durable, single-file backups).
All access goes through the repositories in `enigma.store`, so a PostgreSQL backend can replace
it without touching callers.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MIGRATIONS: tuple[str, ...] = (
    # 1: core tables
    """
    CREATE TABLE documents (
        kind TEXT NOT NULL,
        id TEXT NOT NULL,
        version INTEGER NOT NULL,
        plane TEXT NOT NULL,
        body TEXT NOT NULL,
        body_hash TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        recorded_by TEXT NOT NULL,
        PRIMARY KEY (id, version)
    );
    CREATE INDEX documents_kind ON documents(kind, id);
    CREATE VIEW documents_latest AS
        SELECT d.* FROM documents d
        JOIN (SELECT id, MAX(version) AS v FROM documents GROUP BY id) m
          ON d.id = m.id AND d.version = m.v;

    CREATE TABLE audit_log (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        actor TEXT NOT NULL,
        action TEXT NOT NULL,
        subject_id TEXT,
        details TEXT NOT NULL,
        prev_hash TEXT NOT NULL,
        entry_hash TEXT NOT NULL
    );

    CREATE TABLE fsm_events (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        machine TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        from_state TEXT,
        to_state TEXT NOT NULL,
        actor TEXT NOT NULL,
        reason_code TEXT NOT NULL,
        artifact_hashes TEXT NOT NULL,
        ts TEXT NOT NULL
    );
    CREATE INDEX fsm_events_subject ON fsm_events(machine, subject_id, seq);

    CREATE TABLE outbox (
        message_id TEXT PRIMARY KEY,
        correlation_id TEXT NOT NULL,
        to_agent TEXT NOT NULL,
        from_agent TEXT NOT NULL,
        message_type TEXT NOT NULL,
        envelope TEXT NOT NULL,
        created_at TEXT NOT NULL,
        acked_at TEXT,
        attempts INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX outbox_pending ON outbox(to_agent, acked_at, created_at);

    CREATE TABLE task_queue (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        assigned_to TEXT NOT NULL,
        correlation_id TEXT NOT NULL,
        status TEXT NOT NULL,
        priority INTEGER NOT NULL,
        body TEXT NOT NULL,
        lease_owner TEXT,
        lease_expires_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX task_queue_ready ON task_queue(assigned_to, status, priority, created_at);

    CREATE TABLE costs (
        id TEXT PRIMARY KEY,
        actor TEXT NOT NULL,
        task_id TEXT,
        category TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT,
        input_tokens INTEGER NOT NULL,
        output_tokens INTEGER NOT NULL,
        usd REAL NOT NULL,
        http_requests INTEGER NOT NULL,
        cpu_seconds REAL NOT NULL,
        cached INTEGER NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX costs_actor_time ON costs(actor, created_at);

    CREATE TABLE checkpoints (
        workflow_id TEXT PRIMARY KEY,
        workflow TEXT NOT NULL,
        step TEXT NOT NULL,
        state TEXT NOT NULL,
        status TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE flags (
        name TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        set_by TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE VIRTUAL TABLE search_index USING fts5(
        id UNINDEXED, kind UNINDEXED, plane UNINDEXED, title, body, topics,
        tokenize = 'porter unicode61'
    );
    """,
)


class Database:
    """Thread-safe SQLite wrapper. One writer at a time; WAL allows concurrent readers."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path) if str(path) != ":memory:" else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(path), check_same_thread=False, isolation_level=None, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._depth = 0
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            current = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            for index, script in enumerate(MIGRATIONS[current:], start=current + 1):
                self._conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version={index};\nCOMMIT;")

    @property
    def schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Re-entrant transaction: nested calls join the outermost one."""
        with self._lock:
            outermost = self._depth == 0
            if outermost:
                self._conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield self._conn
            except BaseException:
                self._depth -= 1
                if outermost:
                    self._conn.execute("ROLLBACK")
                raise
            self._depth -= 1
            if outermost:
                self._conn.execute("COMMIT")

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            row: sqlite3.Row | None = self._conn.execute(sql, params).fetchone()
            return row

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self.transaction() as conn:
            return conn.execute(sql, params).rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
