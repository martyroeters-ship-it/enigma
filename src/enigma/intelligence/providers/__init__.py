"""Search provider adapters."""

from __future__ import annotations

import os

from enigma.core.clock import Clock
from enigma.intelligence.http import HttpClient
from enigma.intelligence.providers.arxiv import ArxivProvider
from enigma.intelligence.providers.base import SearchProvider
from enigma.intelligence.providers.crossref import CrossrefProvider
from enigma.intelligence.providers.github import GitHubProvider
from enigma.intelligence.providers.openalex import OpenAlexProvider
from enigma.intelligence.providers.semantic_scholar import SemanticScholarProvider
from enigma.intelligence.providers.websearch import (
    BRAVE_KEY_ENV,
    SERPAPI_KEY_ENV,
    TAVILY_KEY_ENV,
    BraveSearchProvider,
    SerpApiSearchProvider,
    TavilySearchProvider,
)
from enigma.intelligence.providers.wikipedia import WikipediaProvider


def default_providers(client: HttpClient, clock: Clock) -> list[SearchProvider]:
    """Keyless public providers, plus keyed providers only when their credentials are set."""
    providers: list[SearchProvider] = [
        ArxivProvider(client, clock),
        OpenAlexProvider(client, clock),
        CrossrefProvider(client, clock),
        GitHubProvider(client, clock),
        WikipediaProvider(client, clock),
    ]
    if SemanticScholarProvider.configured():
        providers.append(SemanticScholarProvider(client, clock))
    for env, cls in (
        (BRAVE_KEY_ENV, BraveSearchProvider),
        (TAVILY_KEY_ENV, TavilySearchProvider),
        (SERPAPI_KEY_ENV, SerpApiSearchProvider),
    ):
        if os.environ.get(env, "").strip():
            providers.append(cls(client, clock))
    return providers


__all__ = [
    "ArxivProvider",
    "BraveSearchProvider",
    "CrossrefProvider",
    "GitHubProvider",
    "OpenAlexProvider",
    "SearchProvider",
    "SemanticScholarProvider",
    "SerpApiSearchProvider",
    "TavilySearchProvider",
    "WikipediaProvider",
    "default_providers",
]
