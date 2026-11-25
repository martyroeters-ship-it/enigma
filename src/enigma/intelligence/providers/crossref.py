"""Crossref REST adapter (https://api.crossref.org, polite pool via mailto)."""

from __future__ import annotations

import re
from typing import Any

from enigma.intelligence.models import AuthorRef, SearchHit
from enigma.intelligence.providers.base import ProviderBase, mailto, parse_date, strip_markup
from enigma.schemas.knowledge import SourceKind

API_URL = "https://api.crossref.org/works"
SELECT = "DOI,title,abstract,author,issued,URL,type,is-referenced-by-count"
_KIND = {
    "journal-article": SourceKind.ACADEMIC_PAPER,
    "proceedings-article": SourceKind.ACADEMIC_PAPER,
    "book-chapter": SourceKind.ACADEMIC_PAPER,
    "posted-content": SourceKind.PREPRINT,
    "dataset": SourceKind.DATASET_DOC,
}


_ABSTRACT_LABEL_RE = re.compile(r"^\s*abstract[\s:.]+", re.IGNORECASE)


def _abstract(raw: str | None) -> str | None:
    text = strip_markup(raw)
    return _ABSTRACT_LABEL_RE.sub("", text) or None if text else None


def _issued(item: dict[str, Any]) -> str | None:
    parts = ((item.get("issued") or {}).get("date-parts") or [[None]])[0]
    if not parts or parts[0] is None:
        return None
    return "-".join(f"{int(p):02d}" if i else str(int(p)) for i, p in enumerate(parts[:3]))


class CrossrefProvider(ProviderBase):
    name = "crossref"

    def search(self, query: str, limit: int, language: str | None) -> list[SearchHit]:
        if language not in (None, "en"):
            return []
        response = self.client.get(
            API_URL,
            params={"query": query, "rows": limit, "select": SELECT, "mailto": mailto()},
        )
        items = (response.json().get("message") or {}).get("items") or []
        hits: list[SearchHit] = []
        for rank, item in enumerate(items):
            title = strip_markup(" ".join(item.get("title") or []))
            doi = item.get("DOI")
            if not title or not doi:
                continue
            authors = tuple(
                AuthorRef(
                    name=name,
                    external_ids={"orcid": str(a["ORCID"])} if a.get("ORCID") else {},
                )
                for a in item.get("author") or []
                if (
                    name := strip_markup(
                        " ".join(p for p in (a.get("given"), a.get("family")) if p) or a.get("name")
                    )
                )
            )
            rec_uri, rec_hash = self._record_json(item)
            hits.append(
                SearchHit(
                    kind=_KIND.get(str(item.get("type")), SourceKind.OTHER),
                    title=title,
                    url=str(item.get("URL") or f"https://doi.org/{doi}"),
                    abstract=_abstract(item.get("abstract")),
                    authors=authors,
                    published_at=parse_date(_issued(item)),
                    language=None,
                    external_ids={"doi": str(doi)},
                    citation_count=int(item.get("is-referenced-by-count") or 0),
                    **self._provenance(response, rec_uri, rec_hash, query, rank),
                )
            )
        return hits
