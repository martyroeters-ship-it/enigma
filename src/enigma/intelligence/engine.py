"""Recursive research engine (WS4).

Phases: consult prior knowledge -> seed search across providers (and languages) -> query
expansion (TF-IDF key phrases + optional injected expander) -> deduplication -> citation
traversal via OpenAlex (referenced works / cited-by) -> author discovery -> ranking and selection
-> optional full-text fetch -> evidence extraction -> contradiction candidates.

Every request goes through the shared `HttpClient` budget; when it or the wall-clock budget runs
out the engine stops issuing requests and returns a partial result flagged `budget_exhausted`.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial

from enigma.core.clock import Clock
from enigma.core.errors import BudgetExceededError, EnigmaError, PermissionDeniedError
from enigma.core.ids import new_id
from enigma.core.logging import get_logger
from enigma.intelligence.evidence import detect_contradictions, extract_evidence
from enigma.intelligence.expansion import QueryExpander, TfidfQueryExpander, sanitize_queries
from enigma.intelligence.extract import DocumentFetcher, ExtractionError
from enigma.intelligence.http import HttpClient
from enigma.intelligence.models import (
    AuthorRef,
    ProviderStats,
    ResearchQuery,
    ResearchResult,
    SearchHit,
)
from enigma.intelligence.normalize import (
    id_keys,
    near_duplicate_score,
    normalize_external_ids,
    normalize_title,
)
from enigma.intelligence.providers.base import SearchProvider
from enigma.intelligence.providers.openalex import OpenAlexProvider
from enigma.intelligence.ranking import relevance, score_source
from enigma.intelligence.text import content_terms, stems
from enigma.knowledge import KnowledgeBase
from enigma.schemas.common import Actor, AgentRole
from enigma.schemas.knowledge import Author, EdgeRelation, EvidenceItem, KnowledgeEdge, Source

log = get_logger(__name__)

NEAR_DUPLICATE_THRESHOLD = 0.85
TRAVERSAL_MIN_RELEVANCE = 0.34
PROVIDER_PRIORITY = ("openalex", "arxiv", "crossref", "semantic_scholar", "wikipedia", "github")
ACADEMIC_PROVIDERS = frozenset({"openalex", "arxiv", "crossref", "semantic_scholar"})


class _StopResearch(Exception):  # noqa: N818 - internal control flow, not an error
    pass


@dataclass
class _Group:
    """One real-world work and all records that describe it."""

    hits: list[SearchHit]
    reasons: dict[str, str] = field(default_factory=dict)  # content_hash -> dedupe reason

    @property
    def canonical(self) -> SearchHit:
        def key(h: SearchHit) -> tuple[int, int, int, int]:
            prio = (
                PROVIDER_PRIORITY.index(h.provider)
                if h.provider in PROVIDER_PRIORITY
                else len(PROVIDER_PRIORITY)
            )
            return (0 if h.abstract else 1, prio, h.rank, 0)

        return min(self.hits, key=key)

    @property
    def providers(self) -> list[str]:
        return sorted({h.provider for h in self.hits})

    @property
    def external_ids(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for h in sorted(self.hits, key=lambda x: x.provider):
            for k, v in normalize_external_ids(h.external_ids).items():
                merged.setdefault(k, v)
        return merged

    @property
    def referenced_ids(self) -> set[str]:
        return {r for h in self.hits for r in h.referenced_ids}

    @property
    def citation_count(self) -> int | None:
        counts = [h.citation_count for h in self.hits if h.citation_count is not None]
        return max(counts) if counts else None

    @property
    def authors(self) -> list[AuthorRef]:
        best = max(
            self.hits, key=lambda h: (sum(len(a.external_ids) for a in h.authors), len(h.authors))
        )
        return list(best.authors)


class _Dedupe:
    def __init__(self) -> None:
        self.groups: list[_Group] = []
        self._by_key: dict[str, int] = {}
        self._hashes: set[str] = set()

    def add(self, hit: SearchHit) -> int | None:
        """Returns the group index, or None if this exact record was already seen."""
        if hit.content_hash in self._hashes:
            return None
        self._hashes.add(hit.content_hash)
        keys = id_keys(hit.external_ids)
        idx = next((self._by_key[k] for k in sorted(keys) if k in self._by_key), None)
        reason = "id-match"
        if idx is None:
            idx, reason = self._near_duplicate(hit), "shingle-jaccard"
        if idx is None:
            self.groups.append(_Group([hit]))
            idx = len(self.groups) - 1
        else:
            self.groups[idx].hits.append(hit)
            self.groups[idx].reasons[hit.content_hash] = reason
        for k in keys:
            self._by_key.setdefault(k, idx)
        return idx

    def _near_duplicate(self, hit: SearchHit) -> int | None:
        title = normalize_title(hit.title)
        if len(title.split()) < 3:
            return None
        best, best_score = None, NEAR_DUPLICATE_THRESHOLD
        for i, group in enumerate(self.groups):
            c = group.canonical
            if hit.language and c.language and hit.language != c.language:
                continue
            if hit.kind != c.kind and {hit.kind, c.kind} & {"code_repository", "encyclopedia"}:
                continue
            score = near_duplicate_score(hit.title, hit.abstract, c.title, c.abstract)
            if score >= best_score:
                best, best_score = i, score
        return best

    def group_of_openalex(self, work_id: str) -> int | None:
        return self._by_key.get(f"openalex:{work_id}")


class ResearchEngine:
    def __init__(
        self,
        providers: Sequence[SearchProvider],
        client: HttpClient,
        clock: Clock,
        *,
        knowledge: KnowledgeBase | None = None,
        expander: QueryExpander | None = None,
        actor: Actor = AgentRole.INTELLIGENCE,
        id_factory: Callable[[str], str] = new_id,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not providers:
            raise ValueError("at least one search provider is required")
        self.providers = list(providers)
        self.client = client
        self.clock = clock
        self.knowledge = knowledge
        self.expander = expander
        self.deterministic_expander = TfidfQueryExpander()
        self.actor = actor
        self._new_id = id_factory
        self._monotonic = monotonic
        self.fetcher = DocumentFetcher(client)

    # ---------------------------------------------------------------- helpers
    def _openalex(self) -> OpenAlexProvider | None:
        return next((p for p in self.providers if isinstance(p, OpenAlexProvider)), None)

    def _guard(self) -> None:
        if self.client.budget.remaining <= 0:
            self._exhausted = True
            raise _StopResearch
        if self._monotonic() - self._t0 > self._query.budget.max_wall_seconds:
            self._exhausted = True
            raise _StopResearch

    def _call(
        self, provider: str, label: str, fn: Callable[[], list[SearchHit]]
    ) -> list[SearchHit]:
        self._guard()
        stats = self._stats[provider]
        stats["queries"] += 1
        self._issued.append(label)
        try:
            hits = fn()
        except BudgetExceededError as exc:
            self._exhausted = True
            self._errors[provider].append(str(exc))
            raise _StopResearch from exc
        except (EnigmaError, ValueError, KeyError, TypeError) as exc:
            # Provider outages, HTTP errors and malformed payloads are recorded, not fatal.
            if isinstance(exc, PermissionDeniedError):
                log.warning("intelligence.egress_denied", provider=provider, error=str(exc))
            self._errors[provider].append(f"{type(exc).__name__}: {exc}"[:300])
            return []
        stats["hits"] += len(hits)
        for h in hits:
            self._dedupe.add(h)
        return hits

    # ---------------------------------------------------------------- phases
    def run(self, query: ResearchQuery) -> ResearchResult:
        self._query = query
        self._t0 = self._monotonic()
        self._exhausted = False
        self._issued: list[str] = []
        self._stats: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self._errors: defaultdict[str, list[str]] = defaultdict(list)
        self._dedupe = _Dedupe()
        self._dup_meta: dict[str, tuple[str, str]] = {}
        self._pdf_urls: dict[str, str] = {}
        started = self.clock.now()
        requests_before = self.client.budget.used
        topic_text = " ".join((query.topic, *query.keywords))
        self._topic_stems = stems(topic_text)

        prior_ids = self._consult_prior_knowledge(topic_text)
        try:
            seed_hits = self._seed_search(query)
            self._expand(query, seed_hits)
            self._traverse_citations(query)
            self._discover_authors(query)
        except _StopResearch:
            log.info("intelligence.budget_stop", issued=len(self._issued))

        selected = self._select(query)
        sources, duplicates, scores, group_source = self._materialize(query, selected)
        authors, author_edges = self._authors(selected, group_source)
        edges = [*author_edges, *self._group_edges(selected, group_source, duplicates)]
        evidence = self._evidence(query, sources)
        edges.extend(
            detect_contradictions(evidence, self.clock, actor=self.actor, id_factory=self._new_id)
        )
        stats = {}
        for name in sorted({*self._stats, *self._errors}):
            kept = sum(1 for s in sources if s.provider == name)
            stats[name] = ProviderStats(
                queries=self._stats[name]["queries"],
                hits=self._stats[name]["hits"],
                errors=tuple(self._errors[name]),
                sources_kept=kept,
            )
        return ResearchResult(
            query=query,
            sources=tuple(sources),
            duplicate_sources=tuple(duplicates),
            authors=tuple(authors),
            evidence=tuple(evidence),
            edges=tuple(edges),
            queries_issued=tuple(self._issued),
            provider_stats=stats,
            http_requests_used=self.client.budget.used - requests_before,
            budget_exhausted=self._exhausted,
            scores=scores,
            prior_knowledge_ids=prior_ids,
            started_at=started,
            finished_at=self.clock.now(),
        )

    def _consult_prior_knowledge(self, topic_text: str) -> tuple[str, ...]:
        if self.knowledge is None:
            return ()
        prior = self.knowledge.prior_knowledge(content_terms(topic_text))
        return prior.ids

    def _seed_search(self, query: ResearchQuery) -> list[SearchHit]:
        seeds = [query.topic]
        if query.keywords:
            seeds.append(" ".join(content_terms(query.topic)[:3] + list(query.keywords)))
        hits: list[SearchHit] = []
        for q in seeds:
            for p in self.providers:
                label = f"{p.name}:en:{q}"
                hits.extend(
                    self._call(p.name, label, partial(p.search, q, query.per_provider_limit, None))
                )
        for lang in query.languages:
            if lang == "en":
                continue
            q = query.translations.get(lang, query.topic)
            for p in self.providers:
                if not p.multilingual:
                    continue
                limit = max(2, query.per_provider_limit // 2)
                hits.extend(
                    self._call(
                        p.name,
                        f"{p.name}:{lang}:{q}",
                        partial(p.search, q, limit, lang),
                    )
                )
        return hits

    def _expand(self, query: ResearchQuery, seed_hits: list[SearchHit]) -> None:
        if query.max_expansion_queries <= 0:
            return
        relevant = [
            h
            for h in seed_hits
            if relevance(h.title, h.abstract, self._topic_stems) >= TRAVERSAL_MIN_RELEVANCE
        ]
        candidates = self.deterministic_expander.expand(
            query.topic, relevant, query.max_expansion_queries
        )
        if self.expander is not None:
            try:
                candidates = (
                    list(self.expander.expand(query.topic, relevant, query.max_expansion_queries))
                    + candidates
                )
            except Exception as exc:  # injected expanders must never break a research run
                self._errors["expander"].append(f"{type(exc).__name__}: {exc}"[:300])
        issued = [label.split(":", 2)[2] for label in self._issued]
        expansions = sanitize_queries(candidates, issued, query.max_expansion_queries)
        for q in expansions:
            for p in self.providers:
                if p.name not in ACADEMIC_PROVIDERS:
                    continue
                self._call(
                    p.name,
                    f"{p.name}:en:{q}",
                    partial(p.search, q, query.per_provider_limit, None),
                )

    def _ranked_groups(self) -> list[tuple[float, int]]:
        ranked = []
        for i, g in enumerate(self._dedupe.groups):
            c = g.canonical
            s = score_source(
                title=c.title,
                abstract=c.abstract,
                kind=c.kind,
                external_ids=g.external_ids,
                citation_count=g.citation_count,
                popularity=max((h.popularity or 0) for h in g.hits) or None,
                providers=len(g.providers),
                topic_stems=self._topic_stems,
            )
            ranked.append((s, i))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        return ranked

    def _traverse_citations(self, query: ResearchQuery) -> None:
        openalex = self._openalex()
        if openalex is None or query.max_depth == 0 or query.citation_fanout == 0:
            return
        expanded: set[str] = set()
        anchor = " ".join(content_terms(query.topic)[:4])
        for depth in range(1, query.max_depth + 1):
            frontier: list[str] = []
            for _, i in self._ranked_groups():
                wid = self._dedupe.groups[i].external_ids.get("openalex")
                c = self._dedupe.groups[i].canonical
                if (
                    wid
                    and wid not in expanded
                    and relevance(c.title, c.abstract, self._topic_stems) >= TRAVERSAL_MIN_RELEVANCE
                ):
                    frontier.append(wid)
                if len(frontier) >= query.citation_fanout:
                    break
            if not frontier:
                return
            expanded.update(frontier)
            refs: list[str] = []
            for wid in frontier:
                gi = self._dedupe.group_of_openalex(wid)
                if gi is None:
                    continue
                g = self._dedupe.groups[gi]
                refs.extend(
                    r for r in sorted(g.referenced_ids) if self._dedupe.group_of_openalex(r) is None
                )
            refs = list(dict.fromkeys(refs))[: query.citation_fanout * 15]
            if refs:
                self._call(
                    "openalex",
                    f"openalex:refs:d{depth}:{len(refs)}",
                    partial(openalex.works_by_ids, refs),
                )
            for wid in frontier:
                self._call(
                    "openalex",
                    f"openalex:cites:d{depth}:{wid}",
                    partial(openalex.cited_by, wid, query.per_provider_limit, anchor),
                )

    def _discover_authors(self, query: ResearchQuery) -> None:
        openalex = self._openalex()
        if openalex is None or query.author_fanout == 0:
            return
        counts: Counter[str] = Counter()
        for _, i in self._ranked_groups()[: query.max_sources]:
            g = self._dedupe.groups[i]
            c = g.canonical
            if relevance(c.title, c.abstract, self._topic_stems) < TRAVERSAL_MIN_RELEVANCE:
                continue
            for a in g.authors:
                if aid := a.external_ids.get("openalex"):
                    counts[aid] += 1
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: query.author_fanout]
        anchor = " ".join(content_terms(query.topic)[:3])
        for aid, _ in top:
            self._call(
                "openalex",
                f"openalex:author:{aid}",
                partial(openalex.works_by_author, aid, query.per_provider_limit, anchor),
            )

    def _select(self, query: ResearchQuery) -> list[int]:
        """Top-ranked groups, with a floor of one relevant source per provider and language."""
        ranked = self._ranked_groups()
        chosen: list[int] = []
        seen_providers: set[str] = set()
        seen_langs: set[str] = set()
        for _, i in ranked:
            c = self._dedupe.groups[i].canonical
            rel = relevance(c.title, c.abstract, self._topic_stems)
            lang = c.language or "en"
            # Non-English Wikipedia hits are reached via interlanguage links of relevant English
            # articles, so English-term relevance does not apply to them.
            novel_lang = (
                lang != "en"
                and lang in query.languages
                and lang not in seen_langs
                and (c.provider == "wikipedia" or rel >= TRAVERSAL_MIN_RELEVANCE)
            )
            novel_provider = c.provider not in seen_providers and rel >= TRAVERSAL_MIN_RELEVANCE
            if novel_lang or novel_provider:
                chosen.append(i)
                seen_providers.add(c.provider)
                seen_langs.add(lang)
        for _, i in ranked:
            if len(chosen) >= query.max_sources:
                break
            if i not in chosen:
                chosen.append(i)
        order = {i: rank for rank, (_, i) in enumerate(ranked)}
        return sorted(chosen[: query.max_sources], key=lambda i: order[i])

    # ---------------------------------------------------------------- materialize
    def _source(
        self, hit: SearchHit, external_ids: dict[str, str], citation_count: int | None
    ) -> Source:
        return Source(
            id=self._new_id("src"),
            kind=hit.kind,
            provider=hit.provider,
            url=hit.url,
            title=hit.title,
            authors=tuple(a.name for a in hit.authors),
            published_at=hit.published_at,
            language=hit.language,
            abstract=hit.abstract,
            external_ids=external_ids,
            content_hash=hit.content_hash,
            raw_uri=hit.raw_uri,
            retrieved_by=self.actor,
            retrieved_at=hit.retrieved_at,
            query=hit.query,
            raw_response_uri=hit.response_uri,
            raw_response_hash=hit.response_hash,
            citation_count=citation_count,
        )

    def _materialize(
        self, query: ResearchQuery, selected: list[int]
    ) -> tuple[list[Source], list[Source], dict[str, float], dict[int, Source]]:
        scores_by_group = dict((i, s) for s, i in self._ranked_groups())
        sources: list[Source] = []
        duplicates: list[Source] = []
        scores: dict[str, float] = {}
        group_source: dict[int, Source] = {}
        for i in selected:
            g = self._dedupe.groups[i]
            canonical = g.canonical
            ids = g.external_ids
            if len(g.providers) > 1:
                ids["providers"] = ",".join(g.providers)
            src = self._source(canonical, ids, g.citation_count)
            sources.append(src)
            pdf = canonical.pdf_url or next((h.pdf_url for h in g.hits if h.pdf_url), None)
            if pdf:
                self._pdf_urls[src.id] = pdf
            scores[src.id] = scores_by_group[i]
            group_source[i] = src
            for h in g.hits:
                if h is canonical:
                    continue
                dup = self._source(h, normalize_external_ids(h.external_ids), h.citation_count)
                duplicates.append(dup)
                self._dup_meta[dup.id] = (src.id, g.reasons.get(h.content_hash, "id-match"))
        return sources, duplicates, scores, group_source

    def _group_edges(
        self, selected: list[int], group_source: dict[int, Source], duplicates: list[Source]
    ) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        now = self.clock.now()
        for dup in duplicates:
            canonical_id, reason = self._dup_meta[dup.id]
            canonical = next(s for s in group_source.values() if s.id == canonical_id)
            edges.append(
                KnowledgeEdge(
                    id=self._new_id("edge"),
                    src_id=dup.id,
                    dst_id=canonical_id,
                    relation=EdgeRelation.DUPLICATE_OF,
                    weight=1.0 if reason == "id-match" else 0.9,
                    method=f"dedupe:{reason}",
                    created_by=self.actor,
                    created_at=now,
                    details={"providers": [dup.provider, canonical.provider]},
                )
            )
            if dup.provider != canonical.provider:
                edges.append(
                    KnowledgeEdge(
                        id=self._new_id("edge"),
                        src_id=dup.id,
                        dst_id=canonical_id,
                        relation=EdgeRelation.SUPPORTS,
                        weight=1.0 if reason == "id-match" else 0.8,
                        method="corroboration:cross-provider",
                        created_by=self.actor,
                        created_at=now,
                        details={
                            "corroborates": "existence-and-metadata",
                            "providers": [dup.provider, canonical.provider],
                        },
                    )
                )
        selected_set = set(selected)
        for i in selected:
            for ref in sorted(self._dedupe.groups[i].referenced_ids):
                j = self._dedupe.group_of_openalex(ref)
                if j is None or j == i or j not in selected_set:
                    continue
                edges.append(
                    KnowledgeEdge(
                        id=self._new_id("edge"),
                        src_id=group_source[i].id,
                        dst_id=group_source[j].id,
                        relation=EdgeRelation.CITES,
                        weight=1.0,
                        method="openalex:referenced_works",
                        created_by=self.actor,
                        created_at=now,
                    )
                )
        return edges

    def _authors(
        self, selected: list[int], group_source: dict[int, Source]
    ) -> tuple[list[Author], list[KnowledgeEdge]]:
        by_key: dict[str, Author] = {}
        edges: list[KnowledgeEdge] = []
        now = self.clock.now()
        for i in selected:
            src = group_source[i]
            for ref in self._dedupe.groups[i].authors:
                oa = ref.external_ids.get("openalex")
                key = f"openalex:{oa}" if oa else "name:" + normalize_title(ref.name)
                author = by_key.get(key)
                if author is None:
                    author = Author(
                        id=self._new_id("auth"),
                        name=ref.name,
                        external_ids=dict(ref.external_ids),
                        source_ids=(src.id,),
                    )
                elif src.id not in author.source_ids:
                    author = author.model_copy(update={"source_ids": (*author.source_ids, src.id)})
                by_key[key] = author
                edges.append(
                    KnowledgeEdge(
                        id=self._new_id("edge"),
                        src_id=src.id,
                        dst_id=author.id,
                        relation=EdgeRelation.AUTHORED_BY,
                        weight=1.0,
                        method=f"{src.provider}:authorship",
                        created_by=self.actor,
                        created_at=now,
                    )
                )
        return list(by_key.values()), edges

    def _evidence(self, query: ResearchQuery, sources: list[Source]) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        for src in sources:
            if src.abstract:
                evidence.extend(
                    extract_evidence(
                        src,
                        src.abstract,
                        "abstract",
                        self._topic_stems,
                        self.clock,
                        query.evidence_per_source,
                        self.actor,
                        self._new_id,
                    )
                )
        fetched = 0
        for src in sources:
            if fetched >= query.full_text_fetches:
                break
            url = self._pdf_urls.get(src.id)
            if url is None:
                continue
            fetched += 1
            try:
                self._guard()
                doc = self.fetcher.fetch(url)
            except _StopResearch:
                break
            except BudgetExceededError:
                self._exhausted = True
                break
            except (EnigmaError, ExtractionError) as exc:
                self._errors["fulltext"].append(f"{url}: {type(exc).__name__}: {exc}"[:300])
                continue
            self._issued.append(f"fulltext:{url}")
            self._stats["fulltext"]["queries"] += 1
            evidence.extend(
                extract_evidence(
                    src,
                    doc.text,
                    f"fulltext[{doc.content_hash[:19]}]",
                    self._topic_stems,
                    self.clock,
                    query.evidence_per_source,
                    self.actor,
                    self._new_id,
                )
            )
        return evidence
