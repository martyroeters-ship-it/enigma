"""Institutional knowledge registry API (WS3) on top of the plane-scoped document store.

Guarantees:
- Evidence must reference an existing Source (AT4); orphan evidence is rejected.
- Edges must connect existing entities.
- New evidence enters at L0. Claim levels change only through `promote`, which enforces the
  constitutional ladder (L3 only by the validation service with a PASS ValidationReport) and
  records an append-only `ClaimPromotion`.
- History is never rewritten: merges and promotions write new document versions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel

from enigma.core.clock import Clock
from enigma.core.errors import IntegrityError, PermissionDeniedError
from enigma.core.ids import new_id
from enigma.knowledge.identifiers import id_keys, normalize_external_ids
from enigma.knowledge.models import (
    KnowledgeHit,
    KnowledgeStats,
    PriorKnowledge,
    Provenance,
    SourceSummary,
)
from enigma.schemas.common import Actor, AgentRole, ClaimLevel, Plane, SystemActor
from enigma.schemas.knowledge import (
    Author,
    ClaimPromotion,
    EdgeRelation,
    EvidenceItem,
    FailureCondition,
    KnowledgeEdge,
    Lesson,
    MarketMechanism,
    Source,
)
from enigma.schemas.research import Experiment, ExperimentStatus, Hypothesis
from enigma.schemas.validation import GateOutcome, RedactedValidationSummary, ValidationReport
from enigma.store import PlaneView, kind_of, register_kind

register_kind(ClaimPromotion, "claim_promotion", Plane.CONTROL)

_AUTHOR_ID_KEYS = ("openalex", "orcid", "s2", "github")
_LADDER: dict[ClaimLevel, ClaimLevel] = {
    ClaimLevel.L0_UNVERIFIED: ClaimLevel.L1_HYPOTHESIS,
    ClaimLevel.L1_HYPOTHESIS: ClaimLevel.L2_HISTORICAL,
    ClaimLevel.L2_HISTORICAL: ClaimLevel.L3_INDEPENDENTLY_VALIDATED,
}
_PROMOTERS: dict[ClaimLevel, frozenset[Actor]] = {
    ClaimLevel.L1_HYPOTHESIS: frozenset({AgentRole.QUANT_RESEARCH}),
    ClaimLevel.L2_HISTORICAL: frozenset({AgentRole.QUANT_RESEARCH, AgentRole.STRATEGY_ENGINEERING}),
    ClaimLevel.L3_INDEPENDENTLY_VALIDATED: frozenset({SystemActor.VALIDATION_SERVICE}),
}
# Lookup order when resolving an id of unknown kind; prefixes are only a hint.
_ENTITY_MODELS: tuple[type[BaseModel], ...] = (
    Source,
    EvidenceItem,
    Author,
    Hypothesis,
    Lesson,
    FailureCondition,
    MarketMechanism,
    Experiment,
    KnowledgeEdge,
    ClaimPromotion,
)
_PREFIX_HINTS: dict[str, type[BaseModel]] = {
    "src": Source,
    "evid": EvidenceItem,
    "auth": Author,
    "hyp": Hypothesis,
    "lesson": Lesson,
    "les": Lesson,
    "fail": FailureCondition,
    "exp": Experiment,
    "edge": KnowledgeEdge,
}
INDEXED_KINDS = ("source", "evidence", "hypothesis", "lesson", "failure_condition", "mechanism")


class OrphanEvidenceError(IntegrityError):
    """Evidence references a source id that does not exist (AT4)."""


class DanglingEdgeError(IntegrityError):
    """An edge endpoint does not exist."""


class UnknownEntityError(IntegrityError):
    """No readable entity with this id exists."""


def _actor_from(name: str) -> Actor:
    for enum in (AgentRole, SystemActor):
        try:
            return enum(name)
        except ValueError:
            continue
    return SystemActor.SYSTEM


class KnowledgeBase:
    def __init__(self, view: PlaneView, clock: Clock, *, actor: Actor | None = None) -> None:
        self.view = view
        self.clock = clock
        self.actor: Actor = actor if actor is not None else _actor_from(view.actor)
        self._source_keys: dict[str, str] | None = None
        self._known: dict[str, type[BaseModel]] = {}
        self._kind_models = {kind_of(m)[0]: m for m in _ENTITY_MODELS}

    # ------------------------------------------------------------------ lookup
    def _readable(self, model: type[BaseModel]) -> bool:
        return kind_of(model)[1] in self.view.readable

    def _writable(self, model: type[BaseModel]) -> bool:
        return kind_of(model)[1] in self.view.writable

    def get_entity(self, entity_id: str) -> BaseModel | None:
        hinted = self._known.get(entity_id) or _PREFIX_HINTS.get(entity_id.partition("_")[0])
        order = ([hinted] if hinted else []) + [m for m in _ENTITY_MODELS if m is not hinted]
        for model in order:
            if not self._readable(model):
                continue
            doc = self.view.get(model, entity_id)
            if doc is not None:
                self._known[entity_id] = model
                return doc
        return None

    def exists(self, entity_id: str) -> bool:
        return entity_id in self._known or self.get_entity(entity_id) is not None

    def _put(self, doc: BaseModel) -> None:
        self.view.put(doc)
        self._known[str(getattr(doc, "id"))] = type(doc)  # noqa: B009

    def _duplicate_source_ids(self) -> set[str]:
        return {
            e.src_id
            for e in self.view.list(KnowledgeEdge, where={"relation": EdgeRelation.DUPLICATE_OF})
        }

    def _source_index(self) -> dict[str, str]:
        if self._source_keys is None:
            dups = self._duplicate_source_ids()
            index: dict[str, str] = {}
            for src in reversed(self.view.list(Source)):  # oldest first: first writer wins
                if src.id in dups:
                    continue
                for key in self._keys_of(src):
                    index.setdefault(key, src.id)
                self._known[src.id] = Source
            self._source_keys = index
        return self._source_keys

    def _source_with_hash(self, content_hash: str) -> str | None:
        """Any stored source (canonical or duplicate) with byte-identical raw record."""
        matches = self.view.list(Source, where={"content_hash": content_hash}, limit=1)
        return matches[0].id if matches else None

    @staticmethod
    def _keys_of(src: Source) -> set[str]:
        return {*id_keys(src.external_ids), f"hash:{src.content_hash}"}

    def find_source(
        self, external_ids: dict[str, str], content_hash: str | None = None
    ) -> str | None:
        index = self._source_index()
        keys = sorted(id_keys(external_ids))
        if content_hash:
            keys.append(f"hash:{content_hash}")
        return next((index[k] for k in keys if k in index), None)

    # ------------------------------------------------------------------ writes
    def add_source(self, source: Source, *, dedupe: bool = True) -> str:
        """Store a source; returns the canonical id (an existing one when it is a duplicate).

        A duplicate's new identifiers are merged into the canonical source as a new version.
        With `dedupe=False` the record is stored as-is (used for provenance-preserving
        duplicates that are linked by a DUPLICATE_OF edge) and never becomes canonical.
        """
        source = source.model_copy(
            update={
                "external_ids": {
                    **source.external_ids,
                    **normalize_external_ids(source.external_ids),
                }
            }
        )
        index = self._source_index()
        if dedupe:
            existing_id = self.find_source(source.external_ids, source.content_hash)
            if existing_id is not None:
                self._merge_ids(existing_id, source)
                return existing_id
        else:
            same_record = self._source_with_hash(source.content_hash)
            if same_record is not None:
                return same_record
        self._put(source)
        self._index_source(source)
        if dedupe:
            for key in self._keys_of(source):
                index.setdefault(key, source.id)
        return source.id

    def _merge_ids(self, existing_id: str, incoming: Source) -> None:
        existing = self.view.require(Source, existing_id)
        merged = dict(existing.external_ids)
        changed = False
        for k, v in incoming.external_ids.items():
            if k == "providers":
                continue
            if k not in merged:
                merged[k] = v
                changed = True
        providers = sorted(
            {
                *merged.get("providers", existing.provider).split(","),
                incoming.provider,
                *incoming.external_ids.get("providers", "").split(","),
            }
            - {""}
        )
        if len(providers) > 1 and merged.get("providers") != ",".join(providers):
            merged["providers"] = ",".join(providers)
            changed = True
        if changed:
            updated = existing.model_copy(update={"external_ids": merged})
            self._put(updated)
            index = self._source_index()
            for key in self._keys_of(updated):
                index.setdefault(key, existing_id)

    def _index_source(self, source: Source) -> None:
        body = "\n".join(x for x in (source.abstract, " ".join(source.authors)) if x)
        topics = [source.kind.value, source.provider, *(x for x in [source.language] if x)]
        self.view.index_text(source, source.title, body, topics)

    def add_author(self, author: Author) -> str:
        """Dedupe on strong author ids (OpenAlex/ORCID/S2/GitHub), else exact normalized name
        when neither record carries strong ids. Source links are merged as a new version."""
        incoming_keys = {
            f"{k}:{author.external_ids[k]}" for k in _AUTHOR_ID_KEYS if author.external_ids.get(k)
        }
        name_key = " ".join(author.name.lower().split())
        for existing in self.view.list(Author):
            keys = {
                f"{k}:{existing.external_ids[k]}"
                for k in _AUTHOR_ID_KEYS
                if existing.external_ids.get(k)
            }
            same = bool(keys & incoming_keys) or (
                not keys
                and not incoming_keys
                and " ".join(existing.name.lower().split()) == name_key
            )
            if same:
                sources = tuple(dict.fromkeys((*existing.source_ids, *author.source_ids)))
                ids = {**author.external_ids, **existing.external_ids}
                if sources != existing.source_ids or ids != existing.external_ids:
                    self._put(
                        existing.model_copy(update={"source_ids": sources, "external_ids": ids})
                    )
                return existing.id
        self._put(author)
        return author.id

    def add_evidence(self, evidence: EvidenceItem) -> str:
        """Store L0 evidence; idempotent for an identical excerpt/locator/method on a source."""
        if self.view.get(Source, evidence.source_id) is None:
            raise OrphanEvidenceError(
                f"evidence {evidence.id} references unknown source {evidence.source_id}"
            )
        if evidence.claim_level is not ClaimLevel.L0_UNVERIFIED:
            raise PermissionDeniedError("new evidence enters at L0; use promote() to elevate")
        if evidence.is_ai_interpretation and not evidence.model_id:
            raise IntegrityError("AI interpretations must record the generating model_id")
        for existing in self.view.list(EvidenceItem, where={"source_id": evidence.source_id}):
            if (
                existing.excerpt == evidence.excerpt
                and existing.locator == evidence.locator
                and existing.extraction_method == evidence.extraction_method
            ):
                return existing.id
        self._put(evidence)
        self.view.index_text(evidence, evidence.claim[:300], evidence.excerpt, evidence.topics)
        return evidence.id

    def add_hypothesis(self, hypothesis: Hypothesis) -> str:
        for ref in (*hypothesis.evidence_ids, *hypothesis.contradictory_evidence_ids):
            if self.view.get(EvidenceItem, ref) is None:
                raise IntegrityError(f"hypothesis {hypothesis.id} cites unknown evidence {ref}")
        self._put(hypothesis)
        body = "\n".join((hypothesis.mechanism, hypothesis.prediction, hypothesis.formal_statement))
        topics = [*hypothesis.features, *(i.symbol for i in hypothesis.instruments)]
        self.view.index_text(hypothesis, hypothesis.title, body, topics)
        return hypothesis.id

    def add_lesson(self, lesson: Lesson) -> str:
        self._put(lesson)
        self.view.index_text(lesson, lesson.statement[:300], lesson.statement, lesson.topics)
        return lesson.id

    def add_failure(self, failure: FailureCondition) -> str:
        self._put(failure)
        self.view.index_text(
            failure,
            f"{failure.kind.value}: {failure.description[:200]}",
            failure.description,
            [failure.kind.value],
        )
        return failure.id

    def add_mechanism(self, mechanism: MarketMechanism) -> str:
        self._put(mechanism)
        self.view.index_text(mechanism, mechanism.name, mechanism.description, ())
        return mechanism.id

    def add_edge(self, edge: KnowledgeEdge) -> str:
        for endpoint in (edge.src_id, edge.dst_id):
            if not self.exists(endpoint):
                raise DanglingEdgeError(f"edge {edge.id}: endpoint {endpoint} does not exist")
        if edge.src_id == edge.dst_id:
            raise IntegrityError(f"edge {edge.id} is a self-loop")
        for existing in self.edges(edge.src_id, edge.relation, direction="out"):
            if existing.dst_id == edge.dst_id and existing.method == edge.method:
                return existing.id
        self._put(edge)
        if edge.relation is EdgeRelation.DUPLICATE_OF and self._source_keys is not None:
            self._source_keys = None  # rebuild so the duplicate never wins lookups
        return edge.id

    def link(
        self,
        src: str,
        dst: str,
        relation: EdgeRelation,
        method: str,
        weight: float = 1.0,
        details: dict[str, Any] | None = None,
    ) -> str:
        return self.add_edge(
            KnowledgeEdge(
                id=new_id("edge"),
                src_id=src,
                dst_id=dst,
                relation=relation,
                weight=weight,
                method=method,
                created_by=self.actor,
                created_at=self.clock.now(),
                details=details or {},
            )
        )

    # ------------------------------------------------------------------ reads
    def edges(
        self, entity_id: str, relation: EdgeRelation | None = None, direction: str = "both"
    ) -> list[KnowledgeEdge]:
        out: list[KnowledgeEdge] = []
        fields = {"out": ("src_id",), "in": ("dst_id",), "both": ("src_id", "dst_id")}[direction]
        for f in fields:
            where: dict[str, Any] = {f: entity_id}
            if relation is not None:
                where["relation"] = relation
            out.extend(self.view.list(KnowledgeEdge, where=where))
        return out

    def _topics_of(self, doc: BaseModel) -> tuple[str, ...]:
        if isinstance(doc, EvidenceItem | Lesson):
            return doc.topics
        if isinstance(doc, Source):
            return (doc.kind.value, doc.provider)
        if isinstance(doc, Hypothesis):
            return doc.features
        if isinstance(doc, FailureCondition):
            return (doc.kind.value,)
        return ()

    def retrieve(
        self,
        query: str,
        topics: Iterable[str] = (),
        kinds: Sequence[str] = INDEXED_KINDS,
        limit: int = 20,
    ) -> list[KnowledgeHit]:
        """BM25 full-text retrieval over readable planes, optionally filtered by topic."""
        wanted = {t.lower() for t in topics}
        raw = self.view.search_text(query, kinds=list(kinds), limit=limit * 5 if wanted else limit)
        hits: list[KnowledgeHit] = []
        for doc_id, kind, score in raw:
            model = self._kind_models.get(kind)
            if model is None:
                continue
            doc = self.view.get(model, doc_id)
            if doc is None:
                continue
            doc_topics = self._topics_of(doc)
            if wanted and not wanted & {t.lower() for t in doc_topics}:
                continue
            hits.append(self._hit(doc, kind, -score, doc_topics))
            if len(hits) >= limit:
                break
        return hits

    def _hit(
        self, doc: BaseModel, kind: str, score: float, topics: tuple[str, ...]
    ) -> KnowledgeHit:
        title, snippet = "", ""
        if isinstance(doc, Source):
            title, snippet = doc.title, doc.abstract or ""
        elif isinstance(doc, EvidenceItem):
            title, snippet = doc.claim, doc.excerpt
        elif isinstance(doc, Hypothesis):
            title, snippet = doc.title, doc.prediction
        elif isinstance(doc, Lesson):
            title, snippet = doc.statement, doc.outcome
        elif isinstance(doc, FailureCondition):
            title, snippet = doc.kind.value, doc.description
        elif isinstance(doc, MarketMechanism):
            title, snippet = doc.name, doc.description
        level = getattr(doc, "claim_level", None)
        return KnowledgeHit(
            id=str(getattr(doc, "id")),  # noqa: B009
            kind=kind,
            score=round(score, 6),
            title=title[:300],
            snippet=snippet[:500],
            claim_level=self.claim_level(str(getattr(doc, "id"))) if level is not None else None,  # noqa: B009
            topics=topics,
        )

    def prior_knowledge(self, topic_terms: Sequence[str], limit: int = 10) -> PriorKnowledge:
        """Lessons, failure conditions, hypotheses and evidence relevant to `topic_terms`."""
        query = " ".join(topic_terms)

        def fetch(model: type[Any], kind: str) -> list[Any]:
            if not self._readable(model):
                return []
            return [
                doc
                for h in self.retrieve(query, kinds=(kind,), limit=limit)
                if (doc := self.view.get(model, h.id)) is not None
            ]

        evidence: list[EvidenceItem] = fetch(EvidenceItem, "evidence")
        contradictions: dict[str, KnowledgeEdge] = {}
        for e in evidence:
            for edge in self.edges(e.id, EdgeRelation.CONTRADICTS):
                contradictions[edge.id] = edge
        return PriorKnowledge(
            topic_terms=tuple(topic_terms),
            lessons=tuple(fetch(Lesson, "lesson")),
            failure_conditions=tuple(fetch(FailureCondition, "failure_condition")),
            hypotheses=tuple(fetch(Hypothesis, "hypothesis")),
            evidence=tuple(evidence),
            contradictions=tuple(contradictions.values()),
        )

    def canonical_source_id(self, source_id: str) -> str:
        seen = {source_id}
        current = source_id
        while True:
            dups = self.edges(current, EdgeRelation.DUPLICATE_OF, direction="out")
            if not dups or dups[0].dst_id in seen:
                return current
            current = dups[0].dst_id
            seen.add(current)

    def provenance(self, evidence_id: str) -> Provenance:
        evidence = self.view.get(EvidenceItem, evidence_id)
        if evidence is None:
            raise UnknownEntityError(f"evidence {evidence_id} not found")
        source = self.view.get(Source, evidence.source_id)
        if source is None:
            raise OrphanEvidenceError(f"evidence {evidence_id} has no source {evidence.source_id}")
        canonical_id = self.canonical_source_id(source.id)
        canonical = self.view.get(Source, canonical_id) if canonical_id != source.id else None
        author_ids = [
            e.dst_id
            for sid in dict.fromkeys((source.id, canonical_id))
            for e in self.edges(sid, EdgeRelation.AUTHORED_BY, direction="out")
        ]
        authors = [a for aid in dict.fromkeys(author_ids) if (a := self.view.get(Author, aid))]
        return Provenance(
            evidence=evidence,
            source=source,
            canonical_source=canonical,
            authors=tuple(authors),
            edges=tuple(self.edges(evidence_id)),
            effective_claim_level=self.claim_level(evidence_id),
        )

    # ------------------------------------------------------------------ claim levels
    def claim_level(self, entity_id: str) -> ClaimLevel:
        """Effective level: latest ClaimPromotion if any, else the entity's recorded level."""
        if self._readable(ClaimPromotion):
            promotions = self.view.list(ClaimPromotion, where={"entity_id": entity_id})
            if promotions:
                return max(promotions, key=lambda p: (p.created_at, p.id)).to_level
        entity = self.get_entity(entity_id)
        if entity is None:
            raise UnknownEntityError(f"entity {entity_id} not found")
        level = getattr(entity, "claim_level", None)
        if not isinstance(level, ClaimLevel):
            raise UnknownEntityError(f"entity {entity_id} does not carry a claim level")
        return level

    def promote(
        self, entity_id: str, to_level: ClaimLevel, actor: Actor, justification_ref: str
    ) -> ClaimPromotion:
        """Move a claim exactly one rung up the ladder; every other request is denied.

        L0->L1: Quant Research, justified by a registered Hypothesis.
        L1->L2: Quant Research / Strategy Engineering, justified by a SUCCEEDED Experiment.
        L2->L3: only the Validation Service, justified by a PASS ValidationReport.
        """
        current = self.claim_level(entity_id)
        if _LADDER.get(current) is not to_level:
            raise PermissionDeniedError(
                f"{entity_id}: {current.value}->{to_level.value} is not a permitted promotion"
            )
        if actor not in _PROMOTERS[to_level]:
            raise PermissionDeniedError(f"{actor} may not promote claims to {to_level.value}")
        self._check_justification(to_level, justification_ref)
        entity = self.get_entity(entity_id)
        if entity is None:
            raise UnknownEntityError(f"entity {entity_id} not found")
        promotion = ClaimPromotion(
            id=new_id("promo"),
            entity_id=entity_id,
            entity_kind=kind_of(type(entity))[0],
            from_level=current,
            to_level=to_level,
            actor=actor,
            justification_ref=justification_ref,
            created_at=self.clock.now(),
        )
        self._put(promotion)
        if self._writable(type(entity)):
            try:
                updated = type(entity).model_validate(
                    {**entity.model_dump(), "claim_level": to_level}
                )
            except ValueError:
                updated = None  # e.g. Hypothesis cannot carry L3; the promotion record governs
            if updated is not None:
                self._put(updated)
        return promotion

    def _check_justification(self, to_level: ClaimLevel, ref: str) -> None:
        if to_level is ClaimLevel.L1_HYPOTHESIS:
            if not self._readable(Hypothesis) or self.view.get(Hypothesis, ref) is None:
                raise PermissionDeniedError(f"L1 requires a registered Hypothesis; {ref} not found")
            return
        if to_level is ClaimLevel.L2_HISTORICAL:
            experiment = self.view.get(Experiment, ref) if self._readable(Experiment) else None
            if experiment is None or experiment.status is not ExperimentStatus.SUCCEEDED:
                raise PermissionDeniedError(f"L2 requires a SUCCEEDED Experiment; {ref} is not one")
            return
        outcome: GateOutcome | None = None
        if self._readable(ValidationReport):
            report = self.view.get(ValidationReport, ref)
            outcome = report.overall if report is not None else None
        if outcome is None and self._readable(RedactedValidationSummary):
            for summary in self.view.list(RedactedValidationSummary, where={"report_id": ref}):
                outcome = summary.overall
        if outcome is not GateOutcome.PASS:
            raise PermissionDeniedError(f"L3 requires a PASS ValidationReport; {ref} is not one")

    # ------------------------------------------------------------------ stats
    def stats(self, newest: int = 5) -> KnowledgeStats:
        counts: dict[str, int] = {}
        for model in _ENTITY_MODELS:
            if self._readable(model):
                counts[kind_of(model)[0]] = len(self.view.list(model))
        sources = self.view.list(Source) if self._readable(Source) else []
        evidence = self.view.list(EvidenceItem) if self._readable(EvidenceItem) else []
        edges = self.view.list(KnowledgeEdge) if self._readable(KnowledgeEdge) else []
        levels: Counter[str] = Counter()
        promoted = {}
        if self._readable(ClaimPromotion):
            for p in sorted(self.view.list(ClaimPromotion), key=lambda p: (p.created_at, p.id)):
                promoted[p.entity_id] = p.to_level
        for e in evidence:
            levels[promoted.get(e.id, e.claim_level).value] += 1
        dup_ids = {e.src_id for e in edges if e.relation is EdgeRelation.DUPLICATE_OF}
        newest_sources = sorted(sources, key=lambda s: (s.retrieved_at, s.id), reverse=True)[
            :newest
        ]
        return KnowledgeStats(
            counts=counts,
            evidence_by_claim_level=dict(sorted(levels.items())),
            sources_by_provider=dict(sorted(Counter(s.provider for s in sources).items())),
            sources_by_kind=dict(sorted(Counter(s.kind.value for s in sources).items())),
            edges_by_relation=dict(sorted(Counter(e.relation.value for e in edges).items())),
            duplicate_sources=len(dup_ids),
            newest_sources=tuple(
                SourceSummary(
                    id=s.id,
                    title=s.title,
                    provider=s.provider,
                    url=s.url,
                    retrieved_at=s.retrieved_at,
                )
                for s in newest_sources
            ),
        )
