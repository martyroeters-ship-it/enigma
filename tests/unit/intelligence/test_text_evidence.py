"""Deterministic text processing, evidence extraction, contradiction heuristic, normalization."""

from __future__ import annotations

from datetime import UTC, datetime

from enigma.core.clock import FrozenClock
from enigma.intelligence.evidence import (
    CONTRADICTION_METHOD,
    EXTRACTION_METHOD,
    detect_contradictions,
    extract_evidence,
    polarity,
)
from enigma.intelligence.expansion import TfidfQueryExpander, sanitize_queries
from enigma.intelligence.models import SearchHit
from enigma.intelligence.normalize import (
    id_keys,
    near_duplicate_score,
    normalize_arxiv,
    normalize_doi,
    normalize_external_ids,
    normalize_openalex,
)
from enigma.intelligence.ranking import score_source
from enigma.intelligence.text import split_sentences, stems, tfidf_key_phrases
from enigma.schemas.common import AgentRole, ClaimLevel
from enigma.schemas.knowledge import EdgeRelation, EvidenceItem, Source, SourceKind

NOW = datetime(2026, 11, 25, tzinfo=UTC)
TOPIC_STEMS = stems("perpetual futures funding rate basis cryptocurrency")
ABSTRACT = (
    "Perpetual futures are the most traded crypto derivative. We study funding payments on "
    "Binance from 2019 to 2024, i.e. 6,570 eight-hour intervals. We find that the funding rate "
    "significantly predicts next-day basis changes, with a coefficient of 0.42 (t = 5.1). "
    "The paper is organized as follows. Annualized carry returns averaged 11.3% net of fees."
)


def _source(sid: str = "src_A", abstract: str = ABSTRACT) -> Source:
    return Source(
        id=sid,
        kind=SourceKind.PREPRINT,
        provider="arxiv",
        url="https://arxiv.org/abs/1",
        title="Funding",
        abstract=abstract,
        content_hash="sha256:" + "0" * 64,
        retrieved_by=AgentRole.INTELLIGENCE,
        retrieved_at=NOW,
    )


def _evidence(
    eid: str, sid: str, text: str, topics: tuple[str, ...] = ("funding_rate",)
) -> EvidenceItem:
    return EvidenceItem(
        id=eid,
        source_id=sid,
        excerpt=text,
        claim=text,
        topics=topics,
        extraction_method=EXTRACTION_METHOD,
        created_by=AgentRole.INTELLIGENCE,
        created_at=NOW,
    )


def test_split_sentences_offsets_are_verbatim() -> None:
    sentences = split_sentences(ABSTRACT)
    assert len(sentences) == 5
    for s in sentences:
        assert ABSTRACT[s.start : s.end] == s.text
    assert sentences[1].text.endswith("eight-hour intervals.")  # "i.e." is not a boundary


def test_extract_evidence_is_verbatim_quantitative_and_l0() -> None:
    clock = FrozenClock(NOW)
    items = extract_evidence(_source(), ABSTRACT, "abstract", TOPIC_STEMS, clock, limit=2)
    assert len(items) == 2
    texts = [i.excerpt for i in items]
    assert any("significantly predicts" in t for t in texts)
    assert all("organized as follows" not in t for t in texts)
    for item in items:
        start, end = map(int, item.locator.split("chars=")[1].split("-"))  # type: ignore[union-attr]
        assert ABSTRACT[start:end] == item.excerpt
        assert item.extraction_method == "deterministic:quant-sentence-v1"
        assert item.is_ai_interpretation is False and item.model_id is None
        assert item.claim_level is ClaimLevel.L0_UNVERIFIED
        assert item.source_id == "src_A"
        assert item.claim == " ".join(item.excerpt.split()).rstrip(" .;")
    assert "funding_rate" in items[0].topics or "funding_rate" in items[1].topics


def test_extract_evidence_ignores_non_empirical_text() -> None:
    text = "This chapter introduces the reader to the history of markets. It is a long story."
    assert (
        extract_evidence(_source(abstract=text), text, "abstract", TOPIC_STEMS, FrozenClock(NOW))
        == []
    )


def test_polarity() -> None:
    assert polarity("Funding rates significantly predict returns").existence == 1
    assert polarity("Funding rates do not predict returns").existence == -1
    assert polarity("a positive relationship").direction == 1
    assert polarity("a negative relationship").direction == -1


def test_contradiction_heuristic_flags_opposing_claims_across_sources() -> None:
    a = _evidence(
        "e1", "s1", "We find that funding rates significantly predict future basis returns."
    )
    b = _evidence("e2", "s2", "Funding rates do not predict future basis returns in our sample.")
    c = _evidence("e3", "s3", "Funding rates significantly predict future basis returns too.")
    same_source = _evidence("e4", "s1", "Funding rates do not predict future basis returns at all.")
    other_topic = _evidence(
        "e5", "s5", "Funding rates do not predict future basis returns.", ("volatility",)
    )
    edges = detect_contradictions([a, b, c, same_source, other_topic], FrozenClock(NOW))
    pairs = {(e.src_id, e.dst_id) for e in edges}
    assert ("e1", "e2") in pairs and ("e2", "e3") in pairs
    assert ("e1", "e3") not in pairs  # agreeing
    assert ("e1", "e4") not in pairs  # same source
    assert not any("e5" in p for p in pairs)  # different topic
    for e in edges:
        assert e.relation is EdgeRelation.CONTRADICTS
        assert e.method == CONTRADICTION_METHOD == "heuristic:polarity-v1"
        assert 0 < e.weight < 1 and e.details["heuristic"] is True


def test_contradiction_requires_subject_overlap() -> None:
    a = _evidence("e1", "s1", "Funding rates significantly predict basis returns.")
    b = _evidence("e2", "s2", "Liquidation cascades did not increase exchange outages during 2022.")
    assert detect_contradictions([a, b], FrozenClock(NOW)) == []


def test_identifier_normalization() -> None:
    assert normalize_doi("https://doi.org/10.48550/arXiv.2506.08573") == "10.48550/arxiv.2506.08573"
    assert normalize_arxiv("http://arxiv.org/abs/2506.08573v1") == "2506.08573"
    assert normalize_arxiv("arXiv:hep-th/9901001v2") == "hep-th/9901001"
    assert normalize_openalex("https://openalex.org/W4417257235") == "W4417257235"
    ids = normalize_external_ids({"doi": "https://doi.org/10.48550/arxiv.2506.08573"})
    assert ids["arxiv"] == "2506.08573"
    assert id_keys({"arxiv": "2506.08573v3"}) & id_keys(ids)


def test_near_duplicate_score() -> None:
    assert (
        near_duplicate_score(
            "Fundamentals of Perpetual Futures", None, "Fundamentals of perpetual futures.", None
        )
        == 1.0
    )
    assert (
        near_duplicate_score(
            "Fundamentals of Perpetual Futures",
            None,
            "Designing funding rates for perpetual futures",
            None,
        )
        < 0.5
    )


def test_tfidf_key_phrases_deterministic() -> None:
    docs = [
        "Funding rate arbitrage in perpetual swaps with open interest data.",
        "Open interest and funding rate arbitrage strategies on crypto exchanges.",
        "Market making with open interest signals.",
    ]
    first = tfidf_key_phrases(docs, exclude_terms=["perpetual", "funding", "rate"], top_k=3)
    assert first == tfidf_key_phrases(docs, exclude_terms=["perpetual", "funding", "rate"], top_k=3)
    phrases = [p for p, _ in first]
    assert "open interest" in phrases
    assert "funding rate" not in phrases  # contains no term beyond the query


def test_query_expansion_and_sanitizing() -> None:
    expander = TfidfQueryExpander()
    assert expander.name.startswith("deterministic:")
    assert sanitize_queries(["a b\x00  c", "A B C", "", "x" * 500], ["prior"], 5) == [
        "a b c",
        "x" * 200,
    ]


def test_ranking_prefers_empirical_specific_sources_over_popularity() -> None:
    paper = score_source(
        title="Funding rate predictability in perpetual futures",
        abstract=ABSTRACT,
        kind=SourceKind.ACADEMIC_PAPER,
        external_ids={"doi": "10.1/x"},
        citation_count=15,
        popularity=None,
        providers=1,
        topic_stems=TOPIC_STEMS,
    )
    repo = score_source(
        title="perpetual-futures-funding-rate-bot",
        abstract="A trading bot for funding rates.",
        kind=SourceKind.CODE_REPOSITORY,
        external_ids={"github": "x/y"},
        citation_count=None,
        popularity=50_000,
        providers=1,
        topic_stems=TOPIC_STEMS,
    )
    assert paper > repo


def test_identical_long_titles_merge_across_versions() -> None:
    title = "Perpetual Futures for Stocks: The SpaceX Pre-IPO Market"
    preprint = (
        "Robert Shiller proposed perpetual futures in 1993 to create derivative markets. " * 2
    )
    published = (
        "Shiller proposed perpetual futures in 1993 for illiquid assets; we study designs. " * 2
    )
    assert near_duplicate_score(title, preprint, title, published) >= 0.85
    unrelated = "Completely different content about protein folding kinetics in yeast cells. " * 2
    assert near_duplicate_score(title, preprint, title, unrelated) < 0.85


def test_tabular_pdf_residue_is_not_evidence() -> None:
    table = (
        "Basis Spread 3D\nGap between futures\nand spot prices 0.4\nRVol 24h 1.2%\nReturn 3D 0.8%"
    )
    clock = FrozenClock(NOW)
    assert extract_evidence(_source(abstract=table), table, "fulltext", TOPIC_STEMS, clock) == []


def test_expansion_skips_generic_terms() -> None:
    def hit(title: str, abstract: str) -> SearchHit:
        h = "sha256:" + "0" * 64
        return SearchHit(
            provider="openalex",
            kind=SourceKind.ACADEMIC_PAPER,
            title=title,
            url="https://x.org",
            abstract=abstract,
            query="q",
            rank=0,
            raw_uri="r",
            content_hash=h,
            response_uri="r",
            response_hash=h,
            retrieved_at=NOW,
        )

    hits = [
        hit(
            "Funding rate markets and open interest",
            "Perpetual futures markets with open interest.",
        ),
        hit(
            "Open interest in perpetual futures markets", "Funding rate markets and open interest."
        ),
    ]
    queries = TfidfQueryExpander().expand("perpetual futures funding rate", hits, 3)
    assert queries == ["perpetual futures funding rate open interest"]
