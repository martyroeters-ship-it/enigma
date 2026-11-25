"""Typed message bus over the SQLite outbox (ADR-002). At-least-once, idempotent by message id."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from enigma.core.hashing import canonical_json, sha256_hex, sha256_of
from enigma.core.ids import new_id
from enigma.schemas.common import Actor
from enigma.schemas.messages import MessageEnvelope, MessageType
from enigma.store.audit import AuditLog
from enigma.store.db import Database


def idempotent_message_id(key: str) -> str:
    """Deterministic id so a replayed side effect (e.g. after resume) publishes nothing new."""
    return "msg_" + sha256_hex(key)[:26].upper()


class MessageBus:
    def __init__(self, db: Database, audit: AuditLog) -> None:
        self.db = db
        self.audit = audit

    def publish(
        self,
        from_agent: Actor,
        to_agent: Actor,
        message_type: MessageType,
        payload: dict[str, Any],
        correlation_id: str,
        causation_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> MessageEnvelope:
        message_id = idempotent_message_id(idempotency_key) if idempotency_key else new_id("msg")
        envelope = MessageEnvelope(
            message_id=message_id,
            correlation_id=correlation_id,
            causation_id=causation_id,
            timestamp_utc=datetime.now(UTC),
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            payload=payload,
            payload_hash=sha256_of(payload),
        )
        with self.db.transaction() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO outbox(message_id, correlation_id, to_agent, from_agent,"
                " message_type, envelope, created_at) VALUES (?,?,?,?,?,?,?)",
                (
                    envelope.message_id,
                    correlation_id,
                    str(to_agent),
                    str(from_agent),
                    message_type.value,
                    canonical_json(envelope),
                    envelope.timestamp_utc.isoformat(),
                ),
            )
            if cur.rowcount:
                self.audit.record(
                    str(from_agent),
                    "message.published",
                    envelope.message_id,
                    {
                        "to": str(to_agent),
                        "type": message_type.value,
                        "correlation_id": correlation_id,
                        "payload_hash": envelope.payload_hash,
                    },
                )
            else:
                row = conn.execute(
                    "SELECT envelope FROM outbox WHERE message_id = ?", (message_id,)
                ).fetchone()
                envelope = MessageEnvelope.model_validate_json(row["envelope"])
        return envelope

    def poll(self, to_agent: Actor, limit: int = 50) -> list[MessageEnvelope]:
        rows = self.db.query(
            "SELECT envelope FROM outbox WHERE to_agent = ? AND acked_at IS NULL"
            " ORDER BY created_at LIMIT ?",
            (str(to_agent), limit),
        )
        return [MessageEnvelope.model_validate_json(r["envelope"]) for r in rows]

    def ack(self, message_id: str) -> None:
        self.db.execute(
            "UPDATE outbox SET acked_at = ?, attempts = attempts + 1 WHERE message_id = ?",
            (datetime.now(UTC).isoformat(), message_id),
        )

    def conversation(self, correlation_id: str) -> list[MessageEnvelope]:
        rows = self.db.query(
            "SELECT envelope FROM outbox WHERE correlation_id = ? ORDER BY created_at",
            (correlation_id,),
        )
        return [MessageEnvelope.model_validate_json(r["envelope"]) for r in rows]

    def recent(self, limit: int = 100) -> list[MessageEnvelope]:
        rows = self.db.query(
            "SELECT envelope FROM outbox ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [MessageEnvelope.model_validate_json(r["envelope"]) for r in rows]
