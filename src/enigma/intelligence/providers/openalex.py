"""OpenAlex adapter: works search, batch lookup, cited-by and author works (polite pool).

Keyless requests share a small per-IP daily budget; set `OPENALEX_API_KEY` for a dedicated one.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from enigma.intelligence.http import HttpResponse
from enigma.intelligence.models import AuthorRef, SearchHit
from enigma.intelligence.normalize import normalize_openalex
from enigma.intelligence.providers.base import ProviderBase, mailto, parse_date, strip_markup
from enigma.schemas.knowledge import SourceKind

API_URL = "https://api.openalex.org/works"
SELECT = (
    "id,doi,display_name,publication_date,language,authorships,abstract_inverted_index,"
    "cited_by_count,referenced_works,ids,primary_location,type"
)
MAX_IDS_PER_REQUEST = 50
_KIND = {
    "article": SourceKind.ACADEMIC_PAPER,
    "review": SourceKind.ACADEMIC_PAPER,
    "book-chapter": SourceKind.ACADEMIC_PAPER,
    "book": SourceKind.ACADEMIC_PAPER,
    "dissertation": SourceKind.ACADEMIC_PAPER,
    "preprint": SourceKind.PREPRINT,
    "dataset": SourceKind.DATASET_DOC,
}


def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    """Rebuild abstract text from OpenAlex's inverted index (word order by position)."""
    if not inverted:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inverted.items():
        for i in idxs:
            positions[i] = word
    return strip_markup(" ".join(positions[i] for i in sorted(positions)))


class OpenAlexProvider(ProviderBase):
    name = "openalex"
    multilingual = True

    def _params(self, **extra: Any) -> dict[str, Any]:
        params: dict[str, Any] = {"mailto": mailto(), "select": SELECT, **extra}
        key = os.environ.get("OPENALEX_API_KEY", "").strip()
        if key:
            params["api_key"] = key
        return params

    def search(self, query: str, limit: int, language: str | None) -> list[SearchHit]:
        extra: dict[str, Any] = {"search": query, "per_page": limit}
        if language:
            extra["filter"] = f"language:{language}"
        return self._hits(self.client.get(API_URL, params=self._params(**extra)), query)

    def works_by_ids(self, work_ids: Sequence[str]) -> list[SearchHit]:
        hits: list[SearchHit] = []
        ids = [w for w in (normalize_openalex(x) for x in work_ids) if w]
        for i in range(0, len(ids), MAX_IDS_PER_REQUEST):
            batch = ids[i : i + MAX_IDS_PER_REQUEST]
            label = f"openalex_ids:{len(batch)}"
            params = self._params(filter="openalex_id:" + "|".join(batch), per_page=len(batch))
            hits.extend(self._hits(self.client.get(API_URL, params=params), label))
        return hits

    def cited_by(self, work_id: str, limit: int, search: str | None = None) -> list[SearchHit]:
        wid = normalize_openalex(work_id)
        if wid is None:
            return []
        extra: dict[str, Any] = {"filter": f"cites:{wid}", "per_page": limit}
        if search:
            extra["search"] = search
        else:
            extra["sort"] = "cited_by_count:desc"
        return self._hits(self.client.get(API_URL, params=self._params(**extra)), f"cites:{wid}")

    def works_by_author(self, author_id: str, limit: int, search: str | None) -> list[SearchHit]:
        aid = normalize_openalex(author_id)
        if aid is None:
            return []
        extra: dict[str, Any] = {"filter": f"author.id:{aid}", "per_page": limit}
        if search:
            extra["search"] = search
        label = f"author:{aid}"
        return self._hits(self.client.get(API_URL, params=self._params(**extra)), label)

    def _hits(self, response: HttpResponse, query: str) -> list[SearchHit]:
        payload = response.json()
        hits: list[SearchHit] = []
        for rank, work in enumerate(payload.get("results") or []):
            hit = self._hit(work, response, query, rank)
            if hit is not None:
                hits.append(hit)
        return hits

    def _hit(
        self, work: dict[str, Any], response: HttpResponse, query: str, rank: int
    ) -> SearchHit | None:
        title = strip_markup(work.get("display_name"))
        wid = normalize_openalex(str(work.get("id") or ""))
        if not title or wid is None:
            return None
        ids: dict[str, str] = {"openalex": wid}
        if work.get("doi"):
            ids["doi"] = str(work["doi"])
        for key, value in (work.get("ids") or {}).items():
            if key in ("pmid", "pmcid", "mag") and value:
                ids[key] = str(value)
        location = work.get("primary_location") or {}
        url = str(work.get("doi") or location.get("landing_page_url") or work["id"])
        authors = []
        for authorship in work.get("authorships") or []:
            author = authorship.get("author") or {}
            name = strip_markup(author.get("display_name"))
            if not name:
                continue
            aids: dict[str, str] = {}
            if author.get("id") and (aid := normalize_openalex(str(author["id"]))):
                aids["openalex"] = aid
            if author.get("orcid"):
                aids["orcid"] = str(author["orcid"])
            authors.append(AuthorRef(name=name, external_ids=aids))
        rec_uri, rec_hash = self._record_json(work)
        return SearchHit(
            kind=_KIND.get(str(work.get("type")), SourceKind.OTHER),
            title=title,
            url=url,
            abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
            authors=tuple(authors),
            published_at=parse_date(work.get("publication_date")),
            language=work.get("language"),
            external_ids=ids,
            citation_count=int(work.get("cited_by_count") or 0),
            referenced_ids=tuple(
                r
                for r in (normalize_openalex(str(x)) for x in work.get("referenced_works") or [])
                if r
            ),
            pdf_url=location.get("pdf_url"),
            **self._provenance(response, rec_uri, rec_hash, query, rank),
        )
