"""Query expansion: deterministic TF-IDF key-phrase mining, plus an injectable protocol.

A future LLM gateway can implement `QueryExpander`; its suggestions are treated as untrusted
strings (cleaned, length-capped) and never as instructions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from enigma.intelligence.models import SearchHit
from enigma.intelligence.text import content_terms, stem, stems, tfidf_key_phrases
from enigma.intelligence.untrusted import clean_text

MAX_QUERY_CHARS = 200


@runtime_checkable
class QueryExpander(Protocol):
    name: str

    def expand(self, topic: str, hits: Sequence[SearchHit], limit: int) -> list[str]: ...


class TfidfQueryExpander:
    """Expands `topic` with key phrases mined from retrieved titles and abstracts."""

    name = "deterministic:tfidf-ngram-v1"

    def __init__(self, min_doc_freq: int = 2) -> None:
        self.min_doc_freq = min_doc_freq

    def key_phrases(self, topic: str, hits: Sequence[SearchHit], limit: int) -> list[str]:
        docs = [f"{h.title}. {h.abstract or ''}" for h in hits]
        mined = tfidf_key_phrases(
            docs, exclude_terms=content_terms(topic), top_k=limit, min_doc_freq=self.min_doc_freq
        )
        return [phrase for phrase, _ in mined]

    def expand(self, topic: str, hits: Sequence[SearchHit], limit: int) -> list[str]:
        anchor = " ".join(content_terms(topic)[:4])
        topic_stems = stems(topic)
        queries: list[str] = []
        for phrase in self.key_phrases(topic, hits, limit * 3):
            new_terms = [
                t
                for t in phrase.split()
                if any(stem(part) not in topic_stems for part in t.split("-") if part)
            ]
            if not new_terms:
                continue
            queries.append(f"{anchor} {' '.join(new_terms)}".strip())
            if len(queries) >= limit:
                break
        return queries


def sanitize_queries(queries: Sequence[str], issued: Sequence[str], limit: int) -> list[str]:
    seen = {q.lower() for q in issued}
    out: list[str] = []
    for raw in queries:
        q = clean_text(raw).replace("\n", " ")[:MAX_QUERY_CHARS].strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
        if len(out) >= limit:
            break
    return out
