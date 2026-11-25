"""Source ranking that prefers technically specific, empirical sources over popularity alone."""

from __future__ import annotations

import math
import re

from enigma.intelligence.evidence import score_sentence
from enigma.intelligence.text import split_sentences, stems
from enigma.schemas.knowledge import SourceKind

RANKING_METHOD = "deterministic:empirical-rank-v1"
_QUANT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|bps|basis points|percent)|\bt-stat|\bp\s*[<=]", re.I)
_KIND_BONUS = {
    SourceKind.ACADEMIC_PAPER: 0.6,
    SourceKind.PREPRINT: 0.5,
    SourceKind.DATASET_DOC: 0.3,
    SourceKind.ENCYCLOPEDIA: 0.2,
    SourceKind.DOCUMENTATION: 0.2,
}


def relevance(title: str, abstract: str | None, topic_stems: set[str]) -> float:
    """Fraction of topic stems present in title+abstract, with a bonus for title matches."""
    if not topic_stems:
        return 0.0
    body = stems(f"{title} {abstract or ''}") & topic_stems
    in_title = stems(title) & topic_stems
    return len(body) / len(topic_stems) + 0.25 * len(in_title) / len(topic_stems)


def empirical_signal(abstract: str | None, topic_stems: set[str]) -> float:
    if not abstract:
        return 0.0
    sentence_scores = sorted(
        (score_sentence(s.text, topic_stems) for s in split_sentences(abstract)), reverse=True
    )
    top = sum(sentence_scores[:3]) / 3.0
    quant = len(_QUANT_RE.findall(abstract))
    return min(2.0, top / 6.0) + min(1.0, 0.25 * quant)


def score_source(
    *,
    title: str,
    abstract: str | None,
    kind: SourceKind,
    external_ids: dict[str, str],
    citation_count: int | None,
    popularity: int | None,
    providers: int,
    topic_stems: set[str],
) -> float:
    score = 4.0 * relevance(title, abstract, topic_stems)
    score += empirical_signal(abstract, topic_stems)
    if "doi" in external_ids or "arxiv" in external_ids:
        score += 1.0
    if citation_count:
        score += min(1.5, 0.5 * math.log10(1 + citation_count))
    if popularity:
        score += min(0.4, 0.1 * math.log10(1 + popularity))
    score += _KIND_BONUS.get(kind, 0.0)
    score += min(1.0, 0.5 * (providers - 1))
    if not abstract:
        score -= 1.0
    return round(score, 6)
