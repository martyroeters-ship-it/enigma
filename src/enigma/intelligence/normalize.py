"""Title normalization and near-duplicate detection (word-shingle Jaccard)."""

from __future__ import annotations

import re
import unicodedata

from enigma.knowledge.identifiers import (
    STRONG_ID_KEYS,
    id_keys,
    normalize_arxiv,
    normalize_doi,
    normalize_external_ids,
    normalize_openalex,
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)

__all__ = [
    "STRONG_ID_KEYS",
    "id_keys",
    "jaccard",
    "near_duplicate_score",
    "normalize_arxiv",
    "normalize_doi",
    "normalize_external_ids",
    "normalize_openalex",
    "normalize_title",
    "shingles",
]


def normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode() or title
    return " ".join(_WORD_RE.findall(text.lower()))


def shingles(text: str, k: int = 3) -> frozenset[str]:
    """Word k-shingles of normalized text (character shingles for very short text)."""
    words = normalize_title(text).split()
    if len(words) < k:
        joined = " ".join(words)
        return frozenset(joined[i : i + 4] for i in range(max(1, len(joined) - 3)))
    return frozenset(" ".join(words[i : i + k]) for i in range(len(words) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def near_duplicate_score(
    title_a: str, abstract_a: str | None, title_b: str, abstract_b: str | None
) -> float:
    """Similarity in [0, 1]: exact normalized-title match scores 1; else shingle Jaccard.

    Titles dominate; abstracts (when both present) must also agree so that generic titles
    like "Introduction" do not merge distinct works.
    """
    ta, tb = normalize_title(title_a), normalize_title(title_b)
    if not ta or not tb:
        return 0.0
    title_sim = 1.0 if ta == tb else jaccard(shingles(ta, 2), shingles(tb, 2))
    if abstract_a and abstract_b and len(abstract_a) > 80 and len(abstract_b) > 80:
        abs_sim = jaccard(shingles(abstract_a), shingles(abstract_b))
        return min(1.0, 0.6 * title_sim + 0.4 * max(abs_sim, title_sim * 0.5))
    return title_sim if len(ta.split()) >= 4 else title_sim * 0.8
