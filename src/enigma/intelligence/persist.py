"""Persist a `ResearchResult` into the knowledge base, remapping ids to canonical ones."""

from __future__ import annotations

from pydantic import Field

from enigma.intelligence.models import ResearchResult
from enigma.knowledge import KnowledgeBase
from enigma.schemas.common import StrictModel


class PersistReport(StrictModel):
    source_ids: dict[str, str] = Field(description="run-local source id -> stored canonical id")
    sources_new: int = 0
    sources_merged: int = Field(default=0, description="Matched an already-stored source")
    duplicates: int = 0
    authors: int = 0
    evidence_new: int = 0
    evidence_existing: int = 0
    edges: int = 0
    edges_skipped: int = Field(default=0, description="Self-loops or repeats after remapping")


def persist_result(result: ResearchResult, kb: KnowledgeBase) -> PersistReport:
    """Canonical sources first (deduped against existing knowledge), then duplicates, authors,
    evidence (FK-checked by the knowledge base) and edges with remapped endpoints. Re-persisting
    the same result is idempotent."""
    id_map: dict[str, str] = {}
    new = merged = 0
    for src in result.sources:
        stored = kb.add_source(src)
        id_map[src.id] = stored
        if stored == src.id:
            new += 1
        else:
            merged += 1
    for dup in result.duplicate_sources:
        id_map[dup.id] = kb.add_source(dup, dedupe=False)
    for author in result.authors:
        sources = tuple(dict.fromkeys(id_map.get(s, s) for s in author.source_ids))
        id_map[author.id] = kb.add_author(author.model_copy(update={"source_ids": sources}))
    evidence_new = evidence_existing = 0
    for item in result.evidence:
        source_id = id_map.get(item.source_id, item.source_id)
        stored = kb.add_evidence(item.model_copy(update={"source_id": source_id}))
        id_map[item.id] = stored
        if stored == item.id:
            evidence_new += 1
        else:
            evidence_existing += 1
    edges = skipped = 0
    seen: set[tuple[str, str, str]] = set()
    for edge in result.edges:
        src_id = id_map.get(edge.src_id, edge.src_id)
        dst_id = id_map.get(edge.dst_id, edge.dst_id)
        key = (src_id, dst_id, edge.relation.value)
        if src_id == dst_id or key in seen:
            skipped += 1
            continue
        seen.add(key)
        kb.add_edge(edge.model_copy(update={"src_id": src_id, "dst_id": dst_id}))
        edges += 1
    return PersistReport(
        source_ids={s.id: id_map[s.id] for s in (*result.sources, *result.duplicate_sources)},
        sources_new=new,
        sources_merged=merged,
        duplicates=len(result.duplicate_sources),
        authors=len(result.authors),
        evidence_new=evidence_new,
        evidence_existing=evidence_existing,
        edges=edges,
        edges_skipped=skipped,
    )
