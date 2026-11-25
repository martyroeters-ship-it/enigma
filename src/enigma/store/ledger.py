"""Cost ledger, workflow checkpoints, and control flags (kill switch)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from enigma.core.hashing import canonical_json
from enigma.schemas.governance import CostRecord
from enigma.store.db import Database

KILL_SWITCH = "kill_switch"


class CostLedger:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record(self, cost: CostRecord) -> None:
        self.db.execute(
            "INSERT INTO costs(id, actor, task_id, category, provider, model, input_tokens,"
            " output_tokens, usd, http_requests, cpu_seconds, cached, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                cost.id,
                str(cost.actor),
                cost.task_id,
                cost.category,
                cost.provider,
                cost.model,
                cost.input_tokens,
                cost.output_tokens,
                cost.usd,
                cost.http_requests,
                cost.cpu_seconds,
                int(cost.cached),
                cost.created_at.astimezone(UTC).isoformat(),
            ),
        )

    def totals(
        self, actor: str | None = None, task_id: str | None = None, since: datetime | None = None
    ) -> dict[str, float]:
        sql = (
            "SELECT COALESCE(SUM(usd),0) usd, COALESCE(SUM(input_tokens+output_tokens),0) tokens,"
            " COALESCE(SUM(http_requests),0) http, COALESCE(SUM(cpu_seconds),0) cpu,"
            " COUNT(*) n FROM costs WHERE 1=1"
        )
        params: list[Any] = []
        if actor is not None:
            sql += " AND actor = ?"
            params.append(actor)
        if task_id is not None:
            sql += " AND task_id = ?"
            params.append(task_id)
        if since is not None:
            sql += " AND created_at >= ?"
            params.append(since.astimezone(UTC).isoformat())
        row = self.db.query_one(sql, params)
        assert row is not None
        return {
            "usd": float(row["usd"]),
            "tokens": float(row["tokens"]),
            "http_requests": float(row["http"]),
            "cpu_seconds": float(row["cpu"]),
            "records": float(row["n"]),
        }

    def by_actor(self) -> dict[str, dict[str, float]]:
        rows = self.db.query(
            "SELECT actor, category, SUM(usd) usd, SUM(input_tokens+output_tokens) tokens,"
            " SUM(http_requests) http, SUM(cpu_seconds) cpu FROM costs GROUP BY actor, category"
        )
        out: dict[str, dict[str, float]] = {}
        for r in rows:
            bucket = out.setdefault(r["actor"], {})
            bucket[f"{r['category']}_usd"] = float(r["usd"])
            bucket[f"{r['category']}_tokens"] = float(r["tokens"])
            bucket[f"{r['category']}_http"] = float(r["http"])
            bucket[f"{r['category']}_cpu"] = float(r["cpu"])
        return out


class Checkpoints:
    """Persistent workflow state so interrupted cycles resume at the last completed step."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def save(
        self, workflow_id: str, workflow: str, step: str, state: dict[str, Any], status: str
    ) -> None:
        self.db.execute(
            "INSERT INTO checkpoints(workflow_id, workflow, step, state, status, updated_at)"
            " VALUES (?,?,?,?,?,?) ON CONFLICT(workflow_id) DO UPDATE SET step=excluded.step,"
            " state=excluded.state, status=excluded.status, updated_at=excluded.updated_at",
            (
                workflow_id,
                workflow,
                step,
                canonical_json(state),
                status,
                datetime.now(UTC).isoformat(),
            ),
        )

    def load(self, workflow_id: str) -> tuple[str, dict[str, Any], str] | None:
        row = self.db.query_one(
            "SELECT step, state, status FROM checkpoints WHERE workflow_id = ?", (workflow_id,)
        )
        if row is None:
            return None
        return row["step"], dict(json.loads(row["state"])), row["status"]

    def list(self, workflow: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM checkpoints WHERE 1=1"
        params: list[Any] = []
        if workflow is not None:
            sql += " AND workflow = ?"
            params.append(workflow)
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY updated_at DESC"
        return [
            {
                "workflow_id": r["workflow_id"],
                "workflow": r["workflow"],
                "step": r["step"],
                "state": json.loads(r["state"]),
                "status": r["status"],
                "updated_at": r["updated_at"],
            }
            for r in self.db.query(sql, params)
        ]


class Flags:
    def __init__(self, db: Database) -> None:
        self.db = db

    def set(self, name: str, value: str, set_by: str) -> None:
        self.db.execute(
            "INSERT INTO flags(name, value, set_by, updated_at) VALUES (?,?,?,?)"
            " ON CONFLICT(name) DO UPDATE SET value=excluded.value, set_by=excluded.set_by,"
            " updated_at=excluded.updated_at",
            (name, value, set_by, datetime.now(UTC).isoformat()),
        )

    def get(self, name: str) -> str | None:
        row = self.db.query_one("SELECT value FROM flags WHERE name = ?", (name,))
        return None if row is None else str(row["value"])

    def kill_switch_active(self) -> bool:
        return self.get(KILL_SWITCH) == "on"
