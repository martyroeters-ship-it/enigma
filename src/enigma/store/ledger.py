"""Cost ledger, workflow checkpoints, and control flags (kill switch)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from enigma.core.errors import BudgetExceededError
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

    def expire_stale_reservations(self, now: datetime | None = None) -> int:
        """Mark outstanding reservations past their TTL as released. Returns how many expired."""
        now_iso = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
        with self.db.transaction() as conn:
            cur = conn.execute(
                "UPDATE cost_reservations SET status = 'expired', updated_at = ?"
                " WHERE status = 'open' AND expires_at < ?",
                (now_iso, now_iso),
            )
            return int(cur.rowcount)

    def reserved_totals(
        self, actor: str | None = None, task_id: str | None = None, since: datetime | None = None
    ) -> dict[str, float]:
        """Sum of open reservations (caller should expire stale rows first if needed)."""
        sql = (
            "SELECT COALESCE(SUM(usd),0) usd, COALESCE(SUM(tokens),0) tokens,"
            " COALESCE(SUM(http_requests),0) http, COALESCE(SUM(cpu_seconds),0) cpu,"
            " COUNT(*) n FROM cost_reservations WHERE status = 'open'"
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

    def exposure(
        self, actor: str | None = None, task_id: str | None = None, since: datetime | None = None
    ) -> dict[str, float]:
        """Committed spend plus open reservations (what ceilings must cover)."""
        spent = self.totals(actor=actor, task_id=task_id, since=since)
        reserved = self.reserved_totals(actor=actor, task_id=task_id, since=since)
        return {
            k: spent[k] + reserved[k]
            for k in ("usd", "tokens", "http_requests", "cpu_seconds", "records")
        }

    def reserve(
        self,
        *,
        reservation_id: str,
        actor: str,
        task_id: str | None,
        category: str,
        provider: str,
        model: str | None,
        usd: float,
        tokens: int,
        http_requests: int = 1,
        cpu_seconds: float = 0.0,
        ttl_seconds: float = 120.0,
        max_usd: float | None = None,
        max_tokens: float | None = None,
        max_http: float | None = None,
        daily_cap_usd: float | None = None,
        day_start: datetime | None = None,
        now: datetime | None = None,
    ) -> str:
        """Atomically reserve worst-case spend or raise BudgetExceededError."""
        now_dt = (now or datetime.now(UTC)).astimezone(UTC)
        expires = datetime.fromtimestamp(now_dt.timestamp() + ttl_seconds, tz=UTC)
        with self.db.transaction() as conn:
            # expire inside the same lock
            conn.execute(
                "UPDATE cost_reservations SET status = 'expired', updated_at = ?"
                " WHERE status = 'open' AND expires_at < ?",
                (now_dt.isoformat(), now_dt.isoformat()),
            )

            def _sum(table: str, status_clause: str, params: list[Any]) -> dict[str, float]:
                if table == "costs":
                    sql = (
                        "SELECT COALESCE(SUM(usd),0) usd,"
                        " COALESCE(SUM(input_tokens+output_tokens),0) tokens,"
                        " COALESCE(SUM(http_requests),0) http FROM costs WHERE 1=1"
                    )
                else:
                    sql = (
                        "SELECT COALESCE(SUM(usd),0) usd, COALESCE(SUM(tokens),0) tokens,"
                        " COALESCE(SUM(http_requests),0) http FROM cost_reservations"
                        " WHERE status = 'open'"
                    )
                p = list(params)
                row = conn.execute(sql + status_clause, p).fetchone()
                return {
                    "usd": float(row["usd"]),
                    "tokens": float(row["tokens"]),
                    "http": float(row["http"]),
                }

            actor_clause = ""
            params: list[Any] = []
            if task_id is not None:
                actor_clause = " AND actor = ? AND task_id = ?"
                params = [actor, task_id]
            else:
                actor_clause = " AND actor = ?"
                params = [actor]
                if day_start is not None:
                    actor_clause += " AND created_at >= ?"
                    params.append(day_start.astimezone(UTC).isoformat())

            spent = _sum("costs", actor_clause, params)
            reserved = _sum("reservations", actor_clause, params)
            if max_usd is not None and spent["usd"] + reserved["usd"] + usd > max_usd:
                raise BudgetExceededError(
                    f"reserve ${usd:.6f} would exceed usd ceiling "
                    f"(spent ${spent['usd']:.6f} + reserved ${reserved['usd']:.6f} > {max_usd})"
                )
            if (
                max_tokens is not None
                and spent["tokens"] + reserved["tokens"] + tokens > max_tokens
            ):
                raise BudgetExceededError(
                    f"reserve {tokens} tokens would exceed token ceiling "
                    f"(spent {spent['tokens']:.0f} + reserved {reserved['tokens']:.0f})"
                )
            if max_http is not None and spent["http"] + reserved["http"] + http_requests > max_http:
                raise BudgetExceededError("reserve would exceed HTTP request ceiling")
            if daily_cap_usd is not None and usd > 0:
                day_clause = " AND created_at >= ?"
                day_params = [
                    (
                        day_start or now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                    ).isoformat()
                ]
                day_spent = _sum("costs", day_clause, day_params)
                day_reserved = _sum("reservations", day_clause, day_params)
                if day_spent["usd"] + day_reserved["usd"] + usd > daily_cap_usd:
                    raise BudgetExceededError(
                        f"reserve ${usd:.6f} would exceed daily API ceiling ${daily_cap_usd:.2f}"
                    )
            conn.execute(
                "INSERT INTO cost_reservations(id, actor, task_id, category, provider, model, usd,"
                " tokens, http_requests, cpu_seconds, status, created_at, expires_at,"
                " committed_cost_id, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    reservation_id,
                    actor,
                    task_id,
                    category,
                    provider,
                    model,
                    usd,
                    tokens,
                    http_requests,
                    cpu_seconds,
                    "open",
                    now_dt.isoformat(),
                    expires.isoformat(),
                    None,
                    now_dt.isoformat(),
                ),
            )
        return reservation_id

    def commit_reservation(self, reservation_id: str, cost: CostRecord) -> None:
        """Close a reservation and write the actual CostRecord in one transaction."""
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT status FROM cost_reservations WHERE id = ?", (reservation_id,)
            ).fetchone()
            if row is None or row["status"] != "open":
                raise BudgetExceededError(f"reservation {reservation_id} is not open")
            conn.execute(
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
            conn.execute(
                "UPDATE cost_reservations SET status = 'committed', committed_cost_id = ?,"
                " updated_at = ? WHERE id = ?",
                (cost.id, cost.created_at.astimezone(UTC).isoformat(), reservation_id),
            )

    def release_reservation(self, reservation_id: str, now: datetime | None = None) -> None:
        now_iso = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE cost_reservations SET status = 'released', updated_at = ?"
                " WHERE id = ? AND status = 'open'",
                (now_iso, reservation_id),
            )


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
