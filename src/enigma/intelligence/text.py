"""Deterministic text utilities: tokenization, sentence splitting, TF-IDF key-phrase mining."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*[a-z0-9]|[a-z0-9]", re.IGNORECASE)

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each few for
    from further had has have having he her here hers herself him himself his how i if in into is
    it its itself just me more most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself yourselves
    via using use used based new study paper results result show shows shown propose proposed
    approach method methods analysis article within without however thus therefore among across
    whether may might must one two three first second also well many much et al us non
    provide provides find found investigate investigates examine examines present presents
    consider considers effect effects data model models evidence""".split()  # noqa: SIM905
)


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def content_terms(text: str) -> list[str]:
    """Lower-cased non-stopword tokens of length >= 3, de-duplicated, order preserved."""
    seen: dict[str, None] = {}
    for tok in tokens(text):
        if len(tok) >= 3 and tok not in STOPWORDS and not tok.isdigit():
            seen.setdefault(tok, None)
    return list(seen)


def stem(term: str) -> str:
    """Tiny plural stripper so 'rates'/'rate' and 'futures'/'future' match in overlap scoring."""
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if term.endswith("sses"):
        return term[:-2]
    if len(term) > 3 and term.endswith("s") and not term.endswith(("ss", "is", "us")):
        return term[:-1]
    return term


def stems(text: str) -> set[str]:
    return {stem(t) for t in content_terms(text)}


_ABBREV = ("e.g.", "i.e.", "et al.", "vs.", "fig.", "eq.", "no.", "approx.", "cf.", "resp.")
_SENT_END_RE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[A-Z0-9(\"'])|\n{2,}")


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int


def split_sentences(text: str) -> list[Sentence]:
    """Split into sentences with exact char offsets so excerpts stay verbatim slices."""
    out: list[Sentence] = []
    pos = 0
    for match in _SENT_END_RE.finditer(text):
        if text[pos : match.end()].rstrip().lower().endswith(_ABBREV):
            continue
        _append(out, text, pos, match.start())
        pos = match.end()
    _append(out, text, pos, len(text))
    return out


def _append(out: list[Sentence], text: str, start: int, end: int) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end > start:
        out.append(Sentence(text[start:end], start, end))


def ngrams(words: Sequence[str], n: int) -> list[str]:
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def _phrase_words(text: str) -> list[list[str]]:
    """Runs of content words split at stopwords/punctuation (RAKE-style candidate phrases)."""
    runs: list[list[str]] = []
    for chunk in re.split(r"[.,;:!?()\[\]\"]+", text.lower()):
        run: list[str] = []
        for tok in tokens(chunk):
            if tok in STOPWORDS or len(tok) < 3 or tok.isdigit():
                if run:
                    runs.append(run)
                run = []
            else:
                run.append(tok)
        if run:
            runs.append(run)
    return runs


def tfidf_key_phrases(
    documents: Iterable[str],
    exclude_terms: Iterable[str] = (),
    top_k: int = 10,
    ngram_range: tuple[int, int] = (2, 3),
    min_doc_freq: int = 2,
) -> list[tuple[str, float]]:
    """Mine key phrases by summed TF-IDF over documents; deterministic tie-break by phrase.

    Phrases must appear in >= `min_doc_freq` documents and contain at least one term not in
    `exclude_terms` (so expansions add information beyond the original query).
    """
    docs = [d for d in documents if d.strip()]
    if not docs:
        return []
    excluded = {stem(t) for t in exclude_terms}
    per_doc: list[Counter[str]] = []
    df: Counter[str] = Counter()
    for doc in docs:
        counts: Counter[str] = Counter()
        for run in _phrase_words(doc):
            for n in range(ngram_range[0], ngram_range[1] + 1):
                counts.update(ngrams(run, n))
        per_doc.append(counts)
        df.update(counts.keys())
    n_docs = len(docs)
    scores: dict[str, float] = {}
    for counts in per_doc:
        total = sum(counts.values()) or 1
        for phrase, c in counts.items():
            if df[phrase] < min_doc_freq:
                continue
            idf = math.log((1 + n_docs) / (1 + df[phrase])) + 1.0
            scores[phrase] = scores.get(phrase, 0.0) + (c / total) * idf
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[tuple[str, float]] = []
    for phrase, score in ranked:
        words = phrase.split()
        if all(stem(w) in excluded for w in words):
            continue
        if any(phrase in kept or kept in phrase for kept, _ in out):
            continue
        out.append((phrase, round(score, 6)))
        if len(out) >= top_k:
            break
    return out
