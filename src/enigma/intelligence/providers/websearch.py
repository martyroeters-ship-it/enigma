"""Keyed web-search adapters (Brave, Tavily, SerpAPI). They fail closed without credentials.

Keys are read from the environment at call time and never stored on the provider object.
"""

from __future__ import annotations

from typing import Any

from enigma.intelligence.http import HttpResponse
from enigma.intelligence.models import SearchHit
from enigma.intelligence.providers.base import ProviderBase, require_env, strip_markup
from enigma.schemas.knowledge import SourceKind

BRAVE_KEY_ENV = "BRAVE_SEARCH_API_KEY"
TAVILY_KEY_ENV = "TAVILY_API_KEY"
SERPAPI_KEY_ENV = "SERPAPI_API_KEY"


class _WebSearchBase(ProviderBase):
    multilingual = True

    def _hits(
        self,
        response: HttpResponse,
        results: list[dict[str, Any]],
        query: str,
        language: str | None,
        title_key: str,
        url_key: str,
        snippet_key: str,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for rank, item in enumerate(results):
            title = strip_markup(item.get(title_key))
            url = item.get(url_key)
            if not title or not url or not str(url).startswith(("http://", "https://")):
                continue
            rec_uri, rec_hash = self._record_json(item)
            hits.append(
                SearchHit(
                    kind=SourceKind.WEB_PAGE,
                    title=title,
                    url=str(url),
                    abstract=strip_markup(item.get(snippet_key)),
                    language=language,
                    **self._provenance(response, rec_uri, rec_hash, query, rank),
                )
            )
        return hits


class BraveSearchProvider(_WebSearchBase):
    name = "brave"

    def search(self, query: str, limit: int, language: str | None) -> list[SearchHit]:
        key = require_env(BRAVE_KEY_ENV, self.name)
        params: dict[str, Any] = {"q": query, "count": min(limit, 20)}
        if language:
            params["search_lang"] = language
        response = self.client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params=params,
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
        )
        results = (response.json().get("web") or {}).get("results") or []
        return self._hits(response, results, query, language, "title", "url", "description")


class TavilySearchProvider(_WebSearchBase):
    name = "tavily"

    def search(self, query: str, limit: int, language: str | None) -> list[SearchHit]:
        key = require_env(TAVILY_KEY_ENV, self.name)
        response = self.client.post_json(
            "https://api.tavily.com/search",
            {"query": query, "max_results": min(limit, 20), "search_depth": "basic"},
            headers={"Authorization": f"Bearer {key}"},
        )
        results = response.json().get("results") or []
        return self._hits(response, results, query, language, "title", "url", "content")


class SerpApiSearchProvider(_WebSearchBase):
    name = "serpapi"

    def search(self, query: str, limit: int, language: str | None) -> list[SearchHit]:
        key = require_env(SERPAPI_KEY_ENV, self.name)
        params: dict[str, Any] = {
            "engine": "google",
            "q": query,
            "num": min(limit, 20),
            "api_key": key,
        }
        if language:
            params["hl"] = language
        response = self.client.get("https://serpapi.com/search.json", params=params)
        results = response.json().get("organic_results") or []
        return self._hits(response, results, query, language, "title", "link", "snippet")
