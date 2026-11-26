"""arXiv API adapter (Atom feed, https://info.arxiv.org/help/api/)."""

from __future__ import annotations

import re
from xml.etree.ElementTree import Element

from defusedxml.ElementTree import fromstring

from enigma.intelligence.models import AuthorRef, SearchHit
from enigma.intelligence.normalize import normalize_arxiv
from enigma.intelligence.providers.base import ProviderBase, parse_date, strip_markup
from enigma.intelligence.text import content_terms
from enigma.schemas.knowledge import SourceKind

API_URL = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_ENTRY_RE = re.compile(rb"<entry\b.*?</entry>", re.DOTALL)
MAX_QUERY_TERMS = 4


def build_search_query(query: str) -> str:
    """AND of up to four content terms over all fields, e.g. `all:funding AND all:perpetual`."""
    terms = content_terms(query)[:MAX_QUERY_TERMS] or query.split()[:MAX_QUERY_TERMS]
    return " AND ".join(f"all:{t}" for t in terms)


def _text(node: Element | None) -> str | None:
    return None if node is None or node.text is None else node.text


class ArxivProvider(ProviderBase):
    name = "arxiv"

    def search(self, query: str, limit: int, language: str | None) -> list[SearchHit]:
        if language not in (None, "en"):
            return []
        response = self.client.get(
            API_URL,
            params={
                "search_query": build_search_query(query),
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
            },
        )
        root = fromstring(response.content)
        entries = root.findall("a:entry", _NS)
        fragments = _ENTRY_RE.findall(response.content)
        hits: list[SearchHit] = []
        for rank, entry in enumerate(entries):
            title = strip_markup(_text(entry.find("a:title", _NS)))
            abs_url = _text(entry.find("a:id", _NS))
            if not title or not abs_url:
                continue
            arxiv_id = normalize_arxiv(abs_url)
            ids = {"arxiv": arxiv_id} if arxiv_id else {}
            doi = _text(entry.find("arxiv:doi", _NS))
            if doi:
                ids["doi"] = doi.strip()
            pdf_url = None
            for link in entry.findall("a:link", _NS):
                if link.get("title") == "pdf" and link.get("href"):
                    pdf_url = link.get("href")
            authors = tuple(
                AuthorRef(name=name)
                for a in entry.findall("a:author", _NS)
                if (name := strip_markup(_text(a.find("a:name", _NS))))
            )
            fragment = fragments[rank] if rank < len(fragments) else None
            if fragment is not None:
                rec_uri, rec_hash = self._record_bytes(fragment, ".xml")
            else:
                rec_uri, rec_hash = response.raw_uri, response.content_hash
            hits.append(
                SearchHit(
                    kind=SourceKind.PREPRINT,
                    title=title,
                    url=abs_url.replace("http://", "https://", 1),
                    abstract=strip_markup(_text(entry.find("a:summary", _NS))),
                    authors=authors,
                    published_at=parse_date(_text(entry.find("a:published", _NS))),
                    language="en",
                    external_ids=ids,
                    pdf_url=pdf_url,
                    **self._provenance(response, rec_uri, rec_hash, query, rank),
                )
            )
        return hits
