"""Deterministic evidence extraction and a labelled contradiction heuristic.

`EvidenceItem.excerpt` is always an exact slice of the retrieved text (`locator` gives the char
offsets). `claim` is the same sentence with whitespace normalized; no paraphrasing happens here.
Contradiction edges are a cheap lexical heuristic (`heuristic:polarity-v1`) with weight < 1,
meant to surface candidates for human/agent review, not to adjudicate truth.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

from enigma.core.clock import Clock
from enigma.core.ids import new_id
from enigma.intelligence.text import split_sentences, stems
from enigma.schemas.common import Actor, AgentRole, ClaimLevel
from enigma.schemas.knowledge import EdgeRelation, EvidenceItem, KnowledgeEdge, Source

EXTRACTION_METHOD = "deterministic:quant-sentence-v1"
CONTRADICTION_METHOD = "heuristic:polarity-v1"
MIN_SENTENCE_CHARS = 40
MAX_SENTENCE_CHARS = 700
MIN_SCORE = 3.0

_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?\s*(?:%|bps|bp\b|basis points|percent|x\b)?", re.I
)
_EMPIRICAL_CUES = (
    "significant",
    "significantly",
    "predict",
    "predicts",
    "predictive",
    "forecast",
    "funding",
    "basis",
    "return",
    "returns",
    "premium",
    "arbitrage",
    "spread",
    "volatility",
    "correlation",
    "correlated",
    "regression",
    "coefficient",
    "sharpe",
    "annualized",
    "annualised",
    "profit",
    "profitable",
    "outperform",
    "empirical",
    "empirically",
    "estimate",
    "estimated",
    "liquidity",
    "open interest",
    "carry",
    "deviation",
    "mispricing",
    "price discovery",
    "leverage",
    "liquidation",
    "hedge",
    "hedging",
    "drawdown",
    "alpha",
    "statistically",
    "sample",
)
_FINDING_CUES = (
    "we find",
    "we show",
    "we document",
    "results show",
    "results indicate",
    "results suggest",
    "evidence",
    "we observe",
    "our findings",
    "is associated with",
    "leads to",
    "explains",
)
DOMAIN_TAGS: dict[str, tuple[str, ...]] = {
    "funding_rate": ("funding rate", "funding rates", "funding payment", "funding fee"),
    "basis": ("basis", "premium", "contango", "backwardation", "spot-futures", "price gap"),
    "perpetual_futures": ("perpetual", "perpetuals", "perps", "inverse swap", "perpetual swap"),
    "returns": ("return", "returns", "profit", "pnl", "sharpe", "alpha"),
    "volatility": ("volatility", "variance", "drawdown"),
    "arbitrage": ("arbitrage", "carry trade", "cash-and-carry", "hedge", "hedging"),
    "liquidity": ("liquidity", "liquidation", "open interest", "leverage", "order book"),
    "cryptocurrency": ("bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency"),
}

_NEGATION = (
    "no significant",
    "not significant",
    "insignificant",
    "not statistically",
    "does not",
    "do not",
    "did not",
    "fail to",
    "fails to",
    "failed to",
    "no evidence",
    "little evidence",
    "cannot",
    "unable to",
    "no relationship",
    "no impact",
    "no effect",
    "not profitable",
    "unprofitable",
    "not predict",
    "weak evidence",
    "does not hold",
    "no predictive",
)
_AFFIRM = (
    "significant",
    "significantly",
    "predicts",
    "predictive power",
    "strong evidence",
    "evidence of",
    "robust",
    "profitable",
    "outperform",
    "outperforms",
    "explains",
    "leads to",
)
_POSITIVE_DIR = ("positive", "positively", "increase", "increases", "higher", "rise", "rises")
_NEGATIVE_DIR = ("negative", "negatively", "decrease", "decreases", "lower", "decline", "declines")
_GENERIC_STEMS = frozenset({"rate", "market", "price", "trading", "future", "perpetual", "crypto"})


def _count(text: str, cues: Iterable[str]) -> int:
    return sum(1 for cue in cues if re.search(rf"\b{re.escape(cue)}\b", text))


@dataclass(frozen=True)
class ScoredSentence:
    text: str
    start: int
    end: int
    score: float
    topics: tuple[str, ...]


def domain_topics(text: str) -> tuple[str, ...]:
    lower = text.lower()
    return tuple(tag for tag, cues in DOMAIN_TAGS.items() if _count(lower, cues))


def score_sentence(sentence: str, topic_stems: set[str]) -> float:
    """Higher = more quantitative/empirical and on-topic. Zero if it has no empirical signal."""
    lower = sentence.lower()
    numbers = len(_NUMBER_RE.findall(sentence))
    empirical = _count(lower, _EMPIRICAL_CUES)
    finding = _count(lower, _FINDING_CUES)
    if numbers == 0 and empirical == 0:
        return 0.0
    overlap = len(stems(sentence) & topic_stems)
    return 2.0 * min(numbers, 2) + min(empirical, 4) + finding + 1.5 * min(overlap, 3)


def select_sentences(
    text: str, topic_stems: set[str], limit: int, min_score: float = MIN_SCORE
) -> list[ScoredSentence]:
    scored: list[ScoredSentence] = []
    for s in split_sentences(text):
        if not MIN_SENTENCE_CHARS <= len(s.text) <= MAX_SENTENCE_CHARS or _looks_tabular(s.text):
            continue
        score = score_sentence(s.text, topic_stems)
        if score >= min_score:
            scored.append(ScoredSentence(s.text, s.start, s.end, score, domain_topics(s.text)))
    scored.sort(key=lambda x: (-x.score, x.start))
    return sorted(scored[:limit], key=lambda x: x.start)


def _looks_tabular(sentence: str) -> bool:
    """Table/figure residue from PDF extraction: many short lines."""
    lines = [ln for ln in sentence.split("\n") if ln.strip()]
    return len(lines) >= 4 and sum(len(ln) for ln in lines) / len(lines) < 45


def normalize_claim(sentence: str) -> str:
    return " ".join(sentence.split()).rstrip(" .;")


def extract_evidence(
    source: Source,
    text: str,
    section: str,
    topic_stems: set[str],
    clock: Clock,
    limit: int = 3,
    actor: Actor = AgentRole.INTELLIGENCE,
    id_factory: Callable[[str], str] = new_id,
) -> list[EvidenceItem]:
    """Evidence items whose excerpts are verbatim `text[start:end]` slices."""
    items: list[EvidenceItem] = []
    for s in select_sentences(text, topic_stems, limit):
        if text[s.start : s.end] != s.text:
            raise ValueError("excerpt must be a verbatim slice of the source text")
        matched = sorted(stems(s.text) & topic_stems)
        items.append(
            EvidenceItem(
                id=id_factory("evid"),
                source_id=source.id,
                excerpt=s.text,
                locator=f"{section}:chars={s.start}-{s.end}",
                claim=normalize_claim(s.text),
                topics=tuple(dict.fromkeys((*s.topics, *matched))),
                extraction_method=EXTRACTION_METHOD,
                is_ai_interpretation=False,
                claim_level=ClaimLevel.L0_UNVERIFIED,
                confidence=None,
                created_by=actor,
                created_at=clock.now(),
            )
        )
    return items


@dataclass(frozen=True)
class Polarity:
    existence: int  # +1 asserts an effect, -1 denies one, 0 neither
    direction: int  # +1 positive/increase, -1 negative/decrease, 0 neither


def polarity(text: str) -> Polarity:
    lower = text.lower()
    neg = _count(lower, _NEGATION)
    existence = -1 if neg else (1 if _count(lower, _AFFIRM) else 0)
    pos_dir, neg_dir = _count(lower, _POSITIVE_DIR), _count(lower, _NEGATIVE_DIR)
    direction = (pos_dir > neg_dir) - (neg_dir > pos_dir)
    return Polarity(existence, direction)


def detect_contradictions(
    evidence: Sequence[EvidenceItem],
    clock: Clock,
    min_overlap: float = 0.15,
    actor: Actor = AgentRole.INTELLIGENCE,
    id_factory: Callable[[str], str] = new_id,
) -> list[KnowledgeEdge]:
    """CONTRADICTS candidates: same topic, different sources, overlapping subject terms, and
    opposite effect-existence or opposite direction polarity. Weight is capped at 0.6."""
    edges: list[KnowledgeEdge] = []
    prepared = [(e, polarity(e.excerpt), stems(e.excerpt) - _GENERIC_STEMS) for e in evidence]
    for (a, pa, sa), (b, pb, sb) in combinations(prepared, 2):
        if a.source_id == b.source_id or not set(a.topics) & set(b.topics):
            continue
        if not sa or not sb:
            continue
        overlap = len(sa & sb) / len(sa | sb)
        if overlap < min_overlap:
            continue
        opposite_existence = pa.existence * pb.existence == -1
        opposite_direction = pa.direction * pb.direction == -1
        if not (opposite_existence or opposite_direction):
            continue
        edges.append(
            KnowledgeEdge(
                id=id_factory("edge"),
                src_id=a.id,
                dst_id=b.id,
                relation=EdgeRelation.CONTRADICTS,
                weight=round(min(0.6, 0.2 + overlap), 3),
                method=CONTRADICTION_METHOD,
                created_by=actor,
                created_at=clock.now(),
                details={
                    "heuristic": True,
                    "basis": "existence" if opposite_existence else "direction",
                    "shared_terms": sorted(sa & sb)[:12],
                    "overlap": round(overlap, 3),
                    "polarity_src": [pa.existence, pa.direction],
                    "polarity_dst": [pb.existence, pb.direction],
                },
            )
        )
    return edges
