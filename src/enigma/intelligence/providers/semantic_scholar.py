"""Semantic Scholar Graph API adapter.

Works without a key but the shared pool usually answers 429; the HTTP client backs off and the
provider then raises `ProviderUnavailableError`. `SEMANTIC_SCHOLAR_API_KEY` enables a dedicated
rate limit.
"""

from __future__ import annotations

import os

from enigma.intelligence.models import AuthorRef, SearchHit
from enigma.intelligence.providers.base import ProviderBase, parse_date, strip_markup
from enigma.schemas.knowledge import SourceKind

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,externalIds,url,year,publicationDate,authors,citationCount,openAccessPdf"
KEY_ENV = "SEMANTIC_SCHOLAR_API_KEY"


class SemanticScholarProvider(ProviderBase):
    name = "semantic_scholar"

    @staticmethod
    def configured() -> bool:
        return bool(os.environ.get(KEY_ENV, "").strip())

    def search(self, query: str, limit: int, language: str | None) -> list[SearchHit]:
        if language not in (None, "en"):
            return []
        headers = {}
        key = os.environ.get(KEY_ENV, "").strip()
        if key:
            headers["x-api-key"] = key
        response = self.client.get(
            API_URL, params={"query": query, "limit": limit, "fields": FIELDS}, headers=headers
        )
        hits: list[SearchHit] = []
        for rank, paper in enumerate(response.json().get("data") or []):
            title = strip_markup(paper.get("title"))
            pid = paper.get("paperId")
            if not title or not pid:
                continue
            ext = paper.get("externalIds") or {}
            ids = {"s2": str(pid)}
            if ext.get("DOI"):
                ids["doi"] = str(ext["DOI"])
            if ext.get("ArXiv"):
                ids["arxiv"] = str(ext["ArXiv"])
            date = paper.get("publicationDate") or (
                str(paper["year"]) if paper.get("year") else None
            )
            rec_uri, rec_hash = self._record_json(paper)
            hits.append(
                SearchHit(
                    kind=SourceKind.ACADEMIC_PAPER,
                    title=title,
                    url=str(paper.get("url") or f"https://www.semanticscholar.org/paper/{pid}"),
                    abstract=strip_markup(paper.get("abstract")),
                    authors=tuple(
                        AuthorRef(
                            name=name,
                            external_ids={"s2": str(a["authorId"])} if a.get("authorId") else {},
                        )
                        for a in paper.get("authors") or []
                        if (name := strip_markup(a.get("name")))
                    ),
                    published_at=parse_date(date),
                    external_ids=ids,
                    citation_count=int(paper.get("citationCount") or 0),
                    pdf_url=(paper.get("openAccessPdf") or {}).get("url"),
                    **self._provenance(response, rec_uri, rec_hash, query, rank),
                )
            )
        return hits
