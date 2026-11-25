"""Persistence: control database, versioned documents, audit chain, queue, ledgers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from enigma.store.audit import AuditEntry, AuditLog
from enigma.store.db import Database
from enigma.store.documents import DocumentStore, PlaneView, kind_of, register_kind
from enigma.store.ledger import Checkpoints, CostLedger, Flags
from enigma.store.queue import TaskQueue


@dataclass(frozen=True)
class Stores:
    """Composition root bundle. Agents receive scoped views derived from this, not this itself."""

    db: Database
    documents: DocumentStore
    audit: AuditLog
    queue: TaskQueue
    costs: CostLedger
    checkpoints: Checkpoints
    flags: Flags

    @classmethod
    def open(cls, path: Path | str) -> Stores:
        db = Database(path)
        return cls(
            db=db,
            documents=DocumentStore(db),
            audit=AuditLog(db),
            queue=TaskQueue(db),
            costs=CostLedger(db),
            checkpoints=Checkpoints(db),
            flags=Flags(db),
        )


__all__ = [
    "AuditEntry",
    "AuditLog",
    "Checkpoints",
    "CostLedger",
    "Database",
    "DocumentStore",
    "Flags",
    "PlaneView",
    "Stores",
    "TaskQueue",
    "kind_of",
    "register_kind",
]
