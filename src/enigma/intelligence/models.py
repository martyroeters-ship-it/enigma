"""Typed contracts of the Intelligence engine (WS4 `ResearchQuery` / result)."""

from __future__ import annotations

from pydantic import Field, model_validator

from enigma.schemas.common import NonEmptyStr, Sha256, StrictModel, UtcDatetime
from enigma.schemas.governance import Budget
from enigma.schemas.knowledge import Author, EvidenceItem, KnowledgeEdge, Source, SourceKind


class AuthorRef(StrictModel):
    name: NonEmptyStr
    external_ids: dict[str, str] = Field(default_factory=dict)


class SearchHit(StrictModel):
    """One record returned by a provider, with provenance of the raw bytes it came from."""

    provider: NonEmptyStr
    kind: SourceKind
    title: NonEmptyStr
    url: NonEmptyStr
    abstract: str | None = None
    authors: tuple[AuthorRef, ...] = ()
    published_at: UtcDatetime | None = None
    language: str | None = None
    external_ids: dict[str, str] = Field(default_factory=dict)
    citation_count: int | None = Field(default=None, ge=0)
    popularity: int | None = Field(default=None, ge=0, description="e.g. GitHub stars")
    referenced_ids: tuple[str, ...] = Field(default=(), description="OpenAlex work ids cited")
    pdf_url: str | None = None
    query: NonEmptyStr
    rank: int = Field(ge=0, description="Position in the provider's result list")
    raw_uri: str = Field(description="Stored raw bytes of this record")
    content_hash: Sha256 = Field(description="sha256 of the record's raw bytes")
    response_uri: str
    response_hash: Sha256
    retrieved_at: UtcDatetime


class ResearchQuery(StrictModel):
    topic: NonEmptyStr
    keywords: tuple[str, ...] = ()
    languages: tuple[str, ...] = ("en",)
    translations: dict[str, str] = Field(
        default_factory=dict, description="Optional per-language query text, e.g. {'de': '...'}"
    )
    max_sources: int = Field(default=25, ge=1, le=500)
    max_depth: int = Field(default=1, ge=0, le=3, description="Citation traversal depth")
    citation_fanout: int = Field(default=3, ge=0, le=25, description="Works expanded per level")
    per_provider_limit: int = Field(default=10, ge=1, le=50)
    max_expansion_queries: int = Field(default=2, ge=0, le=10)
    author_fanout: int = Field(default=1, ge=0, le=10)
    full_text_fetches: int = Field(
        default=0, ge=0, le=10, description="Top-ranked sources whose full text is fetched"
    )
    evidence_per_source: int = Field(default=3, ge=1, le=20)
    budget: Budget = Field(default_factory=Budget)

    @model_validator(mode="after")
    def _languages_present(self) -> ResearchQuery:
        if not self.languages:
            raise ValueError("at least one language is required")
        return self


class ProviderStats(StrictModel):
    queries: int = 0
    hits: int = 0
    errors: tuple[str, ...] = ()
    sources_kept: int = 0


class ResearchResult(StrictModel):
    query: ResearchQuery
    sources: tuple[Source, ...] = Field(description="Canonical (deduplicated) sources, ranked")
    duplicate_sources: tuple[Source, ...] = Field(
        default=(), description="Non-canonical records kept for provenance (DUPLICATE_OF edges)"
    )
    authors: tuple[Author, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    edges: tuple[KnowledgeEdge, ...] = ()
    queries_issued: tuple[str, ...] = ()
    provider_stats: dict[str, ProviderStats] = Field(default_factory=dict)
    http_requests_used: int = 0
    budget_exhausted: bool = False
    scores: dict[str, float] = Field(default_factory=dict, description="Ranking score per source")
    prior_knowledge_ids: tuple[str, ...] = Field(
        default=(), description="Lessons/failures/hypotheses/evidence consulted before searching"
    )
    started_at: UtcDatetime
    finished_at: UtcDatetime
