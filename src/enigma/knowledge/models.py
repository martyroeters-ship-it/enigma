"""Read models returned by the knowledge API."""

from __future__ import annotations

from pydantic import Field

from enigma.schemas.common import ClaimLevel, StrictModel, UtcDatetime
from enigma.schemas.knowledge import (
    Author,
    EvidenceItem,
    FailureCondition,
    KnowledgeEdge,
    Lesson,
    Source,
)
from enigma.schemas.research import Hypothesis


class KnowledgeHit(StrictModel):
    id: str
    kind: str
    score: float = Field(description="BM25 relevance; higher is better")
    title: str
    snippet: str
    claim_level: ClaimLevel | None = None
    topics: tuple[str, ...] = ()


class PriorKnowledge(StrictModel):
    """What the institution already knows about a topic; consulted before new research."""

    topic_terms: tuple[str, ...]
    lessons: tuple[Lesson, ...] = ()
    failure_conditions: tuple[FailureCondition, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    contradictions: tuple[KnowledgeEdge, ...] = ()

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(
            x.id
            for group in (self.lessons, self.failure_conditions, self.hypotheses, self.evidence)
            for x in group
        )

    @property
    def is_empty(self) -> bool:
        return not self.ids


class Provenance(StrictModel):
    evidence: EvidenceItem
    source: Source
    canonical_source: Source | None = Field(
        default=None, description="Set when `source` is a recorded duplicate of another source"
    )
    authors: tuple[Author, ...] = ()
    edges: tuple[KnowledgeEdge, ...] = Field(default=(), description="Edges touching the evidence")
    effective_claim_level: ClaimLevel


class SourceSummary(StrictModel):
    id: str
    title: str
    provider: str
    url: str
    retrieved_at: UtcDatetime


class KnowledgeStats(StrictModel):
    counts: dict[str, int]
    evidence_by_claim_level: dict[str, int]
    sources_by_provider: dict[str, int]
    sources_by_kind: dict[str, int]
    edges_by_relation: dict[str, int]
    duplicate_sources: int
    newest_sources: tuple[SourceSummary, ...]
