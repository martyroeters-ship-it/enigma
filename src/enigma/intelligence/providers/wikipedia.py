"""Multilingual Wikipedia adapter (MediaWiki Action API).

For a non-English language the English article hits are mapped through interlanguage links
(no machine translation needed), then topped up with a direct search on that language's wiki.
"""

from __future__ import annotations

import re
from typing import Any

from enigma.intelligence.http import HttpResponse
from enigma.intelligence.models import SearchHit
from enigma.intelligence.providers.base import ProviderBase, strip_markup
from enigma.schemas.knowledge import SourceKind

SUPPORTED_LANGUAGES = frozenset({"en", "de", "fr", "zh", "ja", "ru", "es", "it", "ko", "pt", "nl"})
_LANG_RE = re.compile(r"^[a-z]{2,3}$")


def api_url(language: str) -> str:
    if not _LANG_RE.match(language) or language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported Wikipedia language {language!r}")
    return f"https://{language}.wikipedia.org/w/api.php"


class WikipediaProvider(ProviderBase):
    name = "wikipedia"
    multilingual = True

    def _query(self, language: str, **params: Any) -> HttpResponse:
        base = {"action": "query", "format": "json", "formatversion": 2}
        return self.client.get(api_url(language), params={**base, **params})

    def _search_titles(self, language: str, query: str, limit: int) -> list[str]:
        payload = self._query(language, list="search", srsearch=query, srlimit=limit).json()
        return [str(r["title"]) for r in (payload.get("query") or {}).get("search") or []]

    def _langlinks(self, titles: list[str], language: str) -> list[str]:
        if not titles:
            return []
        payload = self._query(
            "en", prop="langlinks", lllang=language, titles="|".join(titles[:20]), lllimit="max"
        ).json()
        by_title = {
            p.get("title"): (p.get("langlinks") or [{}])[0].get("title")
            for p in (payload.get("query") or {}).get("pages") or []
        }
        return [t for t in (by_title.get(x) for x in titles) if t]

    def search(self, query: str, limit: int, language: str | None) -> list[SearchHit]:
        lang = language or "en"
        if lang == "en":
            titles = self._search_titles("en", query, limit)
        else:
            titles = self._langlinks(self._search_titles("en", query, limit), lang)
            if len(titles) < limit:
                for t in self._search_titles(lang, query, limit):
                    if t not in titles:
                        titles.append(t)
            titles = titles[:limit]
        if not titles:
            return []
        response = self._query(
            lang,
            prop="extracts|info|pageprops",
            exintro=1,
            explaintext=1,
            exlimit="max",
            inprop="url",
            ppprop="wikibase_item",
            redirects=1,
            titles="|".join(titles),
        )
        result = response.json().get("query") or {}
        pages = {p.get("title"): p for p in result.get("pages") or []}
        renamed = {
            r.get("from"): r.get("to")
            for r in (result.get("normalized") or []) + (result.get("redirects") or [])
        }
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for rank, title in enumerate(titles):
            resolved = title
            for _ in range(3):
                resolved = renamed.get(resolved, resolved)
            page = pages.get(resolved)
            if page is not None and page.get("title") in seen:
                continue
            if page is None or page.get("missing") or not page.get("fullurl"):
                continue
            seen.add(str(page["title"]))
            ids = {"wikipedia": f"{lang}:{page['title']}"}
            wikidata = (page.get("pageprops") or {}).get("wikibase_item")
            if wikidata:
                ids["wikidata"] = str(wikidata)
            rec_uri, rec_hash = self._record_json(page)
            hits.append(
                SearchHit(
                    kind=SourceKind.ENCYCLOPEDIA,
                    title=str(page["title"]),
                    url=str(page["fullurl"]),
                    abstract=strip_markup(page.get("extract")),
                    language=lang,
                    external_ids=ids,
                    **self._provenance(response, rec_uri, rec_hash, query, rank),
                )
            )
        return hits
