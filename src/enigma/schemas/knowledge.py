"""Knowledge registry entities (KNOWLEDGE_ARCHITECTURE.md §4).

Retrieved evidence and AI-generated interpretation are kept distinct: `EvidenceItem.excerpt` is
always verbatim retrieved text; any model-produced interpretation is flagged with
`is_ai_interpretation=True` and the generating model id.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from enigma.schemas.common import (
    Actor,
    ClaimLevel,
    NonEmptyStr,
    Probability,
    Sha256,
    StrictModel,
    UtcDatetime,
)


class SourceKind(StrEnum):
    ACADEMIC_PAPER = "academic_paper"
    PREPRINT = "preprint"
    CODE_REPOSITORY = "code_repository"
    WEB_PAGE = "web_page"
    ENCYCLOPEDIA = "encyclopedia"
    FORUM_POST = "forum_post"
    DOCUMENTATION = "documentation"
    DATASET_DOC = "dataset_doc"
    OTHER = "other"


class Source(StrictModel):
    id: NonEmptyStr
    kind: SourceKind
    provider: NonEmptyStr = Field(description="Retrieval provider, e.g. arxiv, openalex, github")
    url: NonEmptyStr
    title: NonEmptyStr
    authors: tuple[str, ...] = ()
    published_at: UtcDatetime | None = None
    language: str | None = None
    abstract: str | None = Field(default=None, description="Verbatim retrieved abstract/summary")
    external_ids: dict[str, str] = Field(default_factory=dict, description="doi, arxiv, openalex")
    content_hash: Sha256
    raw_uri: str | None = Field(default=None, description="Path of stored raw response")
    cited_source_ids: tuple[str, ...] = ()
    retrieved_by: Actor
    retrieved_at: UtcDatetime
    query: str | None = Field(default=None, description="Query that surfaced this source")
    raw_response_uri: str | None = Field(
        default=None, description="Stored full HTTP response the record was parsed from"
    )
    raw_response_hash: Sha256 | None = Field(
        default=None, description="sha256 of the full HTTP response bytes"
    )
    citation_count: int | None = Field(
        default=None, ge=0, description="Provider-reported citation count at retrieval time"
    )


class Author(StrictModel):
    id: NonEmptyStr
    name: NonEmptyStr
    external_ids: dict[str, str] = Field(default_factory=dict)
    source_ids: tuple[str, ...] = ()


class EvidenceItem(StrictModel):
    """A claim extracted from a source. Always L0 until tested by Enigma."""

    id: NonEmptyStr
    source_id: NonEmptyStr
    excerpt: NonEmptyStr = Field(description="Verbatim text from the source")
    locator: str | None = Field(default=None, description="Section, page, or char offsets")
    claim: NonEmptyStr = Field(description="Normalized claim statement")
    topics: tuple[str, ...] = ()
    extraction_method: NonEmptyStr = Field(description="e.g. deterministic:keyword-v1, llm:<model>")
    is_ai_interpretation: bool = False
    model_id: str | None = None
    claim_level: ClaimLevel = ClaimLevel.L0_UNVERIFIED
    confidence: Probability | None = None
    created_by: Actor
    created_at: UtcDatetime


class EdgeRelation(StrEnum):
    CITES = "cites"
    AUTHORED_BY = "authored_by"
    DISCUSSES = "discusses"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DUPLICATE_OF = "duplicate_of"
    BASED_ON = "based_on"
    TESTED_BY = "tested_by"
    USES = "uses"
    VALIDATED_BY = "validated_by"
    FAILURE_MODE = "failure_mode"
    DERIVED_FROM = "derived_from"
    SUPERSEDES = "supersedes"
    REFERENCES = "references"
    TARGETS = "targets"


class KnowledgeEdge(StrictModel):
    id: NonEmptyStr
    src_id: NonEmptyStr
    dst_id: NonEmptyStr
    relation: EdgeRelation
    weight: Probability = 1.0
    method: NonEmptyStr = Field(description="How the edge was established")
    created_by: Actor
    created_at: UtcDatetime
    details: dict[str, Any] = Field(default_factory=dict)


class MarketMechanism(StrictModel):
    id: NonEmptyStr
    name: NonEmptyStr
    description: NonEmptyStr
    evidence_ids: tuple[str, ...] = ()
    created_by: Actor
    created_at: UtcDatetime


class FailureKind(StrEnum):
    OVERFIT = "overfit"
    LEAKAGE = "leakage"
    COST_FRAGILITY = "cost_fragility"
    REGIME_DEPENDENCE = "regime_dependence"
    INSUFFICIENT_SIGNIFICANCE = "insufficient_significance"
    BENCHMARK_INFERIOR = "benchmark_inferior"
    IRREPRODUCIBLE = "irreproducible"
    CAPACITY = "capacity"
    DATA_INTEGRITY = "data_integrity"
    IMPLEMENTATION_ERROR = "implementation_error"
    OPERATIONAL = "operational"


class FailureCondition(StrictModel):
    id: NonEmptyStr
    kind: FailureKind
    description: NonEmptyStr
    subject_id: NonEmptyStr = Field(description="Package, experiment, or task that failed")
    gate_id: str | None = None
    created_by: Actor
    created_at: UtcDatetime


class ClaimPromotion(StrictModel):
    """Append-only record of a claim-level change; the latest record is authoritative."""

    id: NonEmptyStr
    entity_id: NonEmptyStr
    entity_kind: NonEmptyStr
    from_level: ClaimLevel
    to_level: ClaimLevel
    actor: Actor
    justification_ref: NonEmptyStr = Field(
        description="Hypothesis (L1), Experiment (L2), or PASS ValidationReport (L3) id"
    )
    created_at: UtcDatetime


class Lesson(StrictModel):
    """Institutional lesson distilled from an outcome; retrieved before new research."""

    id: NonEmptyStr
    statement: NonEmptyStr
    topics: tuple[str, ...]
    derived_from_ids: tuple[str, ...] = Field(description="Reflections, reports, experiments")
    outcome: NonEmptyStr = Field(description="success | failure | inconclusive")
    claim_level: ClaimLevel
    created_by: Actor
    created_at: UtcDatetime
