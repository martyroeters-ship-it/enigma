"""Append-only, versioned document store for typed registry entities.

Every `put` writes a new version; history is never rewritten. Each kind belongs to a plane, and
`PlaneView` enforces which planes a caller may read or write.
"""

from __future__ import annotations

import builtins
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel

from enigma.core.errors import PermissionDeniedError
from enigma.core.hashing import canonical_json, sha256_of
from enigma.schemas.common import Plane
from enigma.schemas.governance import (
    AgentDecision,
    ImprovementProposal,
    Reflection,
    Task,
)
from enigma.schemas.knowledge import (
    Author,
    EvidenceItem,
    FailureCondition,
    KnowledgeEdge,
    Lesson,
    MarketMechanism,
    Source,
)
from enigma.schemas.research import (
    DatasetVersion,
    Experiment,
    Hypothesis,
    ResearchObjective,
    StrategyPackage,
)
from enigma.schemas.validation import (
    HoldoutManifest,
    RedactedValidationSummary,
    ValidationReport,
    ValidationRequest,
)
from enigma.store.db import Database

M = TypeVar("M", bound=BaseModel)

_KINDS: dict[type[BaseModel], tuple[str, Plane]] = {}


def register_kind(model: type[BaseModel], kind: str, plane: Plane) -> None:
    existing = _KINDS.get(model)
    if existing is not None and existing != (kind, plane):
        raise ValueError(f"{model.__name__} already registered as {existing}")
    _KINDS[model] = (kind, plane)


def kind_of(model: type[BaseModel]) -> tuple[str, Plane]:
    try:
        return _KINDS[model]
    except KeyError as exc:
        raise KeyError(f"{model.__name__} is not a registered document kind") from exc


for _model, _kind, _plane in (
    (ResearchObjective, "objective", Plane.CONTROL),
    (Task, "task", Plane.CONTROL),
    (AgentDecision, "decision", Plane.CONTROL),
    (Reflection, "reflection", Plane.CONTROL),
    (ImprovementProposal, "improvement_proposal", Plane.CONTROL),
    (Lesson, "lesson", Plane.CONTROL),
    (Source, "source", Plane.RESEARCH),
    (Author, "author", Plane.RESEARCH),
    (EvidenceItem, "evidence", Plane.RESEARCH),
    (KnowledgeEdge, "edge", Plane.RESEARCH),
    (MarketMechanism, "mechanism", Plane.RESEARCH),
    (DatasetVersion, "dataset_version", Plane.RESEARCH),
    (Hypothesis, "hypothesis", Plane.RESEARCH),
    (Experiment, "experiment", Plane.RESEARCH),
    (StrategyPackage, "strategy_package", Plane.RESEARCH),
    (FailureCondition, "failure_condition", Plane.CONTROL),
    (ValidationRequest, "validation_request", Plane.CONTROL),
    (RedactedValidationSummary, "validation_summary", Plane.CONTROL),
    (ValidationReport, "validation_report", Plane.VALIDATION),
    (HoldoutManifest, "holdout_manifest", Plane.VALIDATION),
):
    register_kind(_model, _kind, _plane)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DocumentStore:
    """Unscoped store. Only the composition root should hold one; agents get a `PlaneView`."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def put(self, doc: BaseModel, recorded_by: str) -> int:
        kind, plane = kind_of(type(doc))
        doc_id = str(getattr(doc, "id"))  # noqa: B009 - every registered kind has an id
        body = canonical_json(doc)
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT MAX(version) FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
            prior = row[0] if row is not None else None
            if prior is not None:
                prior_kind = conn.execute(
                    "SELECT kind FROM documents WHERE id = ? LIMIT 1", (doc_id,)
                ).fetchone()[0]
                if prior_kind != kind:
                    raise ValueError(f"id {doc_id} already used by kind {prior_kind}")
            version = 1 if prior is None else int(prior) + 1
            conn.execute(
                "INSERT INTO documents(kind, id, version, plane, body, body_hash, recorded_at,"
                " recorded_by) VALUES (?,?,?,?,?,?,?,?)",
                (kind, doc_id, version, plane.value, body, sha256_of(doc), _now(), recorded_by),
            )
        return version

    def get(self, model: type[M], doc_id: str) -> M | None:
        kind, _ = kind_of(model)
        row = self.db.query_one(
            "SELECT body FROM documents_latest WHERE id = ? AND kind = ?", (doc_id, kind)
        )
        return None if row is None else model.model_validate_json(row["body"])

    def require(self, model: type[M], doc_id: str) -> M:
        doc = self.get(model, doc_id)
        if doc is None:
            raise KeyError(f"{model.__name__} {doc_id} not found")
        return doc

    def history(self, model: type[M], doc_id: str) -> builtins.list[M]:
        kind, _ = kind_of(model)
        rows = self.db.query(
            "SELECT body FROM documents WHERE id = ? AND kind = ? ORDER BY version", (doc_id, kind)
        )
        return [model.model_validate_json(r["body"]) for r in rows]

    def list(
        self,
        model: type[M],
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        newest_first: bool = True,
    ) -> builtins.list[M]:
        """List latest versions; `where` filters on top-level JSON fields by equality."""
        kind, _ = kind_of(model)
        sql = "SELECT body FROM documents_latest WHERE kind = ?"
        params: list[Any] = [kind]
        for field, value in (where or {}).items():
            if not field.isidentifier():
                raise ValueError(f"invalid field name {field!r}")
            sql += f" AND json_extract(body, '$.{field}') = ?"
            params.append(value.value if hasattr(value, "value") else value)
        sql += " ORDER BY recorded_at DESC, id DESC" if newest_first else " ORDER BY recorded_at"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [model.model_validate_json(r["body"]) for r in self.db.query(sql, params)]

    def count(self, model: type[M]) -> int:
        kind, _ = kind_of(model)
        row = self.db.query_one("SELECT COUNT(*) FROM documents_latest WHERE kind = ?", (kind,))
        return 0 if row is None else int(row[0])

    def index_text(
        self, doc_id: str, kind: str, plane: Plane, title: str, body: str, topics: Iterable[str]
    ) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM search_index WHERE id = ?", (doc_id,))
            conn.execute(
                "INSERT INTO search_index(id, kind, plane, title, body, topics)"
                " VALUES (?,?,?,?,?,?)",
                (doc_id, kind, plane.value, title, body, " ".join(topics)),
            )

    def search_text(
        self,
        query: str,
        planes: Iterable[Plane],
        kinds: Iterable[str] | None = None,
        limit: int = 20,
    ) -> builtins.list[tuple[str, str, float]]:
        """BM25 full-text search. Returns (id, kind, score) with lower score = better match."""
        terms = [t for t in "".join(c if c.isalnum() else " " for c in query).split() if t]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        plane_values = [p.value for p in planes]
        sql = (
            "SELECT id, kind, bm25(search_index) AS score FROM search_index "
            f"WHERE search_index MATCH ? AND plane IN ({','.join('?' * len(plane_values))})"
        )
        params: list[Any] = [match, *plane_values]
        kind_list = list(kinds or [])
        if kind_list:
            sql += f" AND kind IN ({','.join('?' * len(kind_list))})"
            params.extend(kind_list)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        return [(r["id"], r["kind"], float(r["score"])) for r in self.db.query(sql, params)]

    def raw_json(self, doc_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT body FROM documents_latest WHERE id = ?", (doc_id,))
        return None if row is None else dict(json.loads(row["body"]))


class PlaneView:
    """Least-privilege view: reads and writes limited to an explicit set of planes."""

    def __init__(
        self,
        store: DocumentStore,
        readable: frozenset[Plane],
        writable: frozenset[Plane],
        actor: str,
    ) -> None:
        self._store = store
        self.readable = readable
        self.writable = writable
        self.actor = actor

    def _check(self, model: type[BaseModel], allowed: frozenset[Plane], verb: str) -> None:
        _, plane = kind_of(model)
        if plane not in allowed:
            raise PermissionDeniedError(
                f"{self.actor} may not {verb} {model.__name__} in plane {plane.value}"
            )

    def put(self, doc: BaseModel) -> int:
        self._check(type(doc), self.writable, "write")
        return self._store.put(doc, recorded_by=self.actor)

    def get(self, model: type[M], doc_id: str) -> M | None:
        self._check(model, self.readable, "read")
        return self._store.get(model, doc_id)

    def require(self, model: type[M], doc_id: str) -> M:
        self._check(model, self.readable, "read")
        return self._store.require(model, doc_id)

    def list(
        self, model: type[M], where: dict[str, Any] | None = None, limit: int | None = None
    ) -> builtins.list[M]:
        self._check(model, self.readable, "read")
        return self._store.list(model, where=where, limit=limit)

    def history(self, model: type[M], doc_id: str) -> builtins.list[M]:
        self._check(model, self.readable, "read")
        return self._store.history(model, doc_id)

    def search_text(
        self, query: str, kinds: Iterable[str] | None = None, limit: int = 20
    ) -> builtins.list[tuple[str, str, float]]:
        return self._store.search_text(query, self.readable, kinds=kinds, limit=limit)

    def index_text(self, doc: BaseModel, title: str, body: str, topics: Iterable[str]) -> None:
        kind, plane = kind_of(type(doc))
        self._check(type(doc), self.writable, "index")
        self._store.index_text(str(getattr(doc, "id")), kind, plane, title, body, topics)  # noqa: B009
