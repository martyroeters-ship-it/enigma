"""Durable task queue with leases. A crashed worker's lease expires and the task is re-claimed."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from enigma.core.hashing import canonical_json
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import Task, TaskStatus
from enigma.store.db import Database


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


class TaskQueue:
    def __init__(self, db: Database) -> None:
        self.db = db

    def enqueue(self, task: Task) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO task_queue(id, kind, assigned_to, correlation_id, status, priority,"
                " body, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    task.id,
                    task.kind,
                    task.assigned_to.value,
                    task.correlation_id,
                    task.status.value,
                    task.priority,
                    canonical_json(task),
                    _iso(task.created_at),
                    _iso(task.updated_at),
                ),
            )

    def get(self, task_id: str) -> Task | None:
        row = self.db.query_one("SELECT body FROM task_queue WHERE id = ?", (task_id,))
        return None if row is None else Task.model_validate_json(row["body"])

    def claim(
        self,
        worker: str,
        roles: tuple[AgentRole, ...] | None = None,
        lease: timedelta = timedelta(minutes=10),
        now: datetime | None = None,
    ) -> Task | None:
        """Atomically claim the highest-priority ready task (pending, or expired lease)."""
        now = now or datetime.now(UTC)
        sql = (
            "SELECT id, body FROM task_queue WHERE (status = 'pending' OR "
            "(status = 'running' AND lease_expires_at < ?))"
        )
        params: list[Any] = [_iso(now)]
        if roles:
            sql += f" AND assigned_to IN ({','.join('?' * len(roles))})"
            params.extend(r.value for r in roles)
        sql += " ORDER BY priority ASC, created_at ASC LIMIT 1"
        with self.db.transaction() as conn:
            row = conn.execute(sql, params).fetchone()
            if row is None:
                return None
            task = Task.model_validate_json(row["body"])
            task = task.model_copy(
                update={
                    "status": TaskStatus.RUNNING,
                    "attempts": task.attempts + 1,
                    "updated_at": now,
                }
            )
            conn.execute(
                "UPDATE task_queue SET status = ?, body = ?, lease_owner = ?,"
                " lease_expires_at = ?, updated_at = ? WHERE id = ?",
                (
                    task.status.value,
                    canonical_json(task),
                    worker,
                    _iso(now + lease),
                    _iso(now),
                    task.id,
                ),
            )
        return task

    def heartbeat(
        self, task_id: str, worker: str, lease: timedelta = timedelta(minutes=10)
    ) -> None:
        now = datetime.now(UTC)
        self.db.execute(
            "UPDATE task_queue SET lease_expires_at = ? WHERE id = ? AND lease_owner = ?",
            (_iso(now + lease), task_id, worker),
        )

    def complete(self, task_id: str, outputs: dict[str, Any]) -> Task:
        return self._finish(task_id, TaskStatus.SUCCEEDED, outputs=outputs)

    def fail(self, task_id: str, error: str, retry: bool = True) -> Task:
        """Fail a task. If attempts remain and `retry`, it returns to pending."""
        task = self.get(task_id)
        if task is None:
            raise KeyError(task_id)
        if retry and task.attempts < task.max_attempts:
            return self._finish(task_id, TaskStatus.PENDING, error=error)
        return self._finish(task_id, TaskStatus.FAILED, error=error)

    def cancel(self, task_id: str, reason: str) -> Task:
        return self._finish(task_id, TaskStatus.CANCELLED, error=reason)

    def _finish(
        self,
        task_id: str,
        status: TaskStatus,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> Task:
        now = datetime.now(UTC)
        with self.db.transaction() as conn:
            row = conn.execute("SELECT body FROM task_queue WHERE id = ?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(task_id)
            task = Task.model_validate_json(row["body"])
            update: dict[str, Any] = {"status": status, "updated_at": now}
            if outputs is not None:
                update["outputs"] = outputs
            if error is not None:
                update["error"] = error
            task = task.model_copy(update=update)
            conn.execute(
                "UPDATE task_queue SET status = ?, body = ?, lease_owner = NULL,"
                " lease_expires_at = NULL, updated_at = ? WHERE id = ?",
                (status.value, canonical_json(task), _iso(now), task_id),
            )
        return task

    def requeue_orphans(self, is_alive: Callable[[str], bool]) -> list[str]:
        """Return running tasks whose lease owner is no longer alive to pending (crash recovery)."""
        requeued: list[str] = []
        now = datetime.now(UTC)
        with self.db.transaction() as conn:
            rows = conn.execute(
                "SELECT body, lease_owner FROM task_queue WHERE status = 'running'"
            ).fetchall()
            for row in rows:
                if row["lease_owner"] and is_alive(str(row["lease_owner"])):
                    continue
                task = Task.model_validate_json(row["body"]).model_copy(
                    update={"status": TaskStatus.PENDING, "updated_at": now}
                )
                conn.execute(
                    "UPDATE task_queue SET status = 'pending', body = ?, lease_owner = NULL,"
                    " lease_expires_at = NULL, updated_at = ? WHERE id = ?",
                    (canonical_json(task), _iso(now), task.id),
                )
                requeued.append(task.id)
        return requeued

    def list(
        self,
        status: TaskStatus | None = None,
        correlation_id: str | None = None,
        limit: int = 200,
    ) -> list[Task]:
        sql = "SELECT body FROM task_queue WHERE 1=1"
        params: list[Any] = []
        if status is not None:
            sql += " AND status = ?"
            params.append(status.value)
        if correlation_id is not None:
            sql += " AND correlation_id = ?"
            params.append(correlation_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [Task.model_validate_json(r["body"]) for r in self.db.query(sql, params)]

    def counts(self) -> dict[str, int]:
        rows = self.db.query("SELECT status, COUNT(*) AS n FROM task_queue GROUP BY status")
        return {r["status"]: int(r["n"]) for r in rows}
