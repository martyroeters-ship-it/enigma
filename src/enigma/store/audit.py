"""Append-only, hash-chained audit log. Tampering with any entry breaks `verify()`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from enigma.core.errors import IntegrityError
from enigma.core.hashing import canonical_json, sha256_hex
from enigma.store.db import Database

GENESIS = "0" * 64


@dataclass(frozen=True, slots=True)
class AuditEntry:
    seq: int
    ts: str
    actor: str
    action: str
    subject_id: str | None
    details: dict[str, Any]
    prev_hash: str
    entry_hash: str


def _entry_hash(
    prev_hash: str, ts: str, actor: str, action: str, subject_id: str | None, details_json: str
) -> str:
    return sha256_hex(canonical_json([prev_hash, ts, actor, action, subject_id, details_json]))


class AuditLog:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record(
        self,
        actor: str,
        action: str,
        subject_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        details_json = canonical_json(details or {})
        ts = datetime.now(UTC).isoformat()
        with self.db.transaction() as conn:
            row = conn.execute("SELECT entry_hash FROM audit_log ORDER BY seq DESC LIMIT 1")
            last = row.fetchone()
            prev_hash = GENESIS if last is None else str(last[0])
            entry_hash = _entry_hash(prev_hash, ts, actor, action, subject_id, details_json)
            cur = conn.execute(
                "INSERT INTO audit_log(ts, actor, action, subject_id, details, prev_hash,"
                " entry_hash) VALUES (?,?,?,?,?,?,?)",
                (ts, actor, action, subject_id, details_json, prev_hash, entry_hash),
            )
            seq = int(cur.lastrowid or 0)
        return AuditEntry(seq, ts, actor, action, subject_id, details or {}, prev_hash, entry_hash)

    def entries(
        self, subject_id: str | None = None, limit: int = 200, after_seq: int = 0
    ) -> list[AuditEntry]:
        sql = "SELECT * FROM audit_log WHERE seq > ?"
        params: list[Any] = [after_seq]
        if subject_id is not None:
            sql += " AND subject_id = ?"
            params.append(subject_id)
        sql += " ORDER BY seq DESC LIMIT ?"
        params.append(limit)
        return [
            AuditEntry(
                int(r["seq"]),
                r["ts"],
                r["actor"],
                r["action"],
                r["subject_id"],
                json.loads(r["details"]),
                r["prev_hash"],
                r["entry_hash"],
            )
            for r in self.db.query(sql, params)
        ]

    def verify(self) -> int:
        """Recompute the full chain. Returns entry count; raises IntegrityError on mismatch."""
        prev = GENESIS
        count = 0
        for r in self.db.query("SELECT * FROM audit_log ORDER BY seq"):
            if r["prev_hash"] != prev:
                raise IntegrityError(f"audit chain broken at seq {r['seq']} (prev_hash)")
            expected = _entry_hash(
                prev, r["ts"], r["actor"], r["action"], r["subject_id"], r["details"]
            )
            if expected != r["entry_hash"]:
                raise IntegrityError(f"audit chain broken at seq {r['seq']} (entry_hash)")
            prev = expected
            count += 1
        return count
