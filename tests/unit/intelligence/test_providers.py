"""Provider adapters parsed against RECORDED real responses (respx replay, not live)."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import httpx
import pytest
import respx

from enigma.core.clock import FrozenClock
from enigma.core.errors import ProviderUnavailableError
from enigma.intelligence.http import HttpClient, RawStore
from enigma.intelligence.providers import (
    ArxivProvider,
    BraveSearchProvider,
    CrossrefProvider,
    GitHubProvider,
    OpenAlexProvider,
    SearchProvider,
    SemanticScholarProvider,
    SerpApiSearchProvider,
    TavilySearchProvider,
    WikipediaProvider,
    default_providers,
)
from enigma.intelligence.providers.arxiv import build_search_query
from enigma.intelligence.providers.openalex import reconstruct_abstract
from enigma.schemas.knowledge import SourceKind

ClientFactory = Callable[..., HttpClient]
TOPIC = "perpetual futures funding rate"


def _assert_provenance(store: RawStore, hit_raw_uri: str, content_hash: str) -> None:
    data = store.read(hit_raw_uri)
    assert content_hash == "sha256:" + hashlib.sha256(data).hexdigest()


def test_arxiv_query_builder() -> None:
    assert build_search_query(TOPIC) == (
        "all:perpetual AND all:futures AND all:funding AND all:rate"
    )


def test_arxiv(
    recorded_api: respx.MockRouter, make_client: ClientFactory, clock: FrozenClock
) -> None:
    client = make_client()
    hits = ArxivProvider(client, clock).search(TOPIC, 4, None)
    assert [h.external_ids["arxiv"] for h in hits] == [
        "2506.08573",
        "2212.06888",
        "2605.06405",
        "1912.03270",
    ]
    first = hits[0]
    assert first.kind is SourceKind.PREPRINT
    assert first.url == "https://arxiv.org/abs/2506.08573v1"
    assert first.pdf_url == "https://arxiv.org/pdf/2506.08573v1"
    assert first.authors[0].name == "Jaehyun Kim"
    assert first.abstract is not None and "funding rates" in first.abstract
    raw = client.raw_store.read(first.raw_uri)
    assert raw.startswith(b"<entry>") and b"2506.08573" in raw
    _assert_provenance(client.raw_store, first.raw_uri, first.content_hash)
    _assert_provenance(client.raw_store, first.response_uri, first.response_hash)
    assert first.retrieved_at.year == 2026
    assert ArxivProvider(make_client(), clock).search(TOPIC, 4, "de") == []


def test_openalex(
    recorded_api: respx.MockRouter, make_client: ClientFactory, clock: FrozenClock
) -> None:
    client = make_client()
    hits = OpenAlexProvider(client, clock).search(TOPIC, 4, None)
    assert len(hits) == 4
    by_id = {h.external_ids["openalex"]: h for h in hits}
    designing = by_id["W4417257235"]
    assert designing.external_ids["doi"] == "https://doi.org/10.48550/arxiv.2506.08573"
    assert designing.citation_count == 12
    assert designing.abstract and designing.abstract.startswith("In cryptocurrency markets")
    assert by_id["W7223101883"].referenced_ids[:3] == ("W2120832833", "W4312405632", "W4407361822")
    assert all(a.external_ids.get("openalex", "").startswith("A") for a in designing.authors)
    _assert_provenance(client.raw_store, designing.raw_uri, designing.content_hash)
    request = recorded_api.calls.last.request
    assert request.url.params["mailto"]
    assert request.url.params["search"] == TOPIC


def test_openalex_reconstruct_abstract() -> None:
    assert reconstruct_abstract({"funding": [1], "The": [0], "rate.": [2]}) == "The funding rate."
    assert reconstruct_abstract(None) is None


def test_openalex_traversal_calls(
    recorded_api: respx.MockRouter, make_client: ClientFactory, clock: FrozenClock
) -> None:
    provider = OpenAlexProvider(make_client(), clock)
    refs = provider.works_by_ids(["https://openalex.org/W2120832833", "W4312405632", "bad"])
    assert {h.external_ids["openalex"] for h in refs} >= {"W2120832833", "W4312405632"}
    assert recorded_api.calls.last.request.url.params["filter"] == (
        "openalex_id:W2120832833|W4312405632"
    )
    cites = provider.cited_by("W4417257235", 3, "perpetual futures")
    assert cites and all(h.query == "cites:W4417257235" for h in cites)


def test_crossref(
    recorded_api: respx.MockRouter, make_client: ClientFactory, clock: FrozenClock
) -> None:
    client = make_client()
    hits = CrossrefProvider(client, clock).search(TOPIC, 4, None)
    assert {h.external_ids["doi"] for h in hits} >= {"10.2139/ssrn.7690458"}
    for h in hits:
        if h.abstract:
            assert "<jats" not in h.abstract and not h.abstract.lower().startswith("abstract")
        _assert_provenance(client.raw_store, h.raw_uri, h.content_hash)


def test_github(
    recorded_api: respx.MockRouter, make_client: ClientFactory, clock: FrozenClock
) -> None:
    hits = GitHubProvider(make_client(), clock).search(TOPIC, 3, None)
    assert hits[0].title == "aoki-h-jp/funding-rate-arbitrage"
    assert hits[0].kind is SourceKind.CODE_REPOSITORY
    assert hits[0].popularity is not None and hits[0].popularity > 0
    assert "Authorization" not in recorded_api.calls.last.request.headers


def test_wikipedia_english(
    recorded_api: respx.MockRouter, make_client: ClientFactory, clock: FrozenClock
) -> None:
    hits = WikipediaProvider(make_client(), clock).search(TOPIC, 3, None)
    titles = [h.title for h in hits]
    assert "Perpetual futures" in titles
    pf = hits[titles.index("Perpetual futures")]
    assert pf.url.startswith("https://en.wikipedia.org/wiki/")
    assert pf.external_ids["wikipedia"] == "en:Perpetual futures"
    assert pf.kind is SourceKind.ENCYCLOPEDIA and pf.language == "en"


def test_wikipedia_german_via_langlinks(
    recorded_api: respx.MockRouter, make_client: ClientFactory, clock: FrozenClock
) -> None:
    hits = WikipediaProvider(make_client(), clock).search(TOPIC, 3, "de")
    assert {h.title for h in hits} == {"Terminkontrakt", "London Interbank Offered Rate"}
    assert hits[0].language == "de"
    assert hits[0].url.startswith("https://de.wikipedia.org/")
    hosts = [c.request.url.host for c in recorded_api.calls]
    assert "en.wikipedia.org" in hosts and "de.wikipedia.org" in hosts


def test_wikipedia_rejects_unknown_language(make_client: ClientFactory, clock: FrozenClock) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        WikipediaProvider(make_client(), clock).search(TOPIC, 3, "evil.example.com/")


@respx.mock
def test_semantic_scholar_429_backs_off_then_unavailable(
    make_client: ClientFactory,
    clock: FrozenClock,
    sleeps: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    route = respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
        return_value=httpx.Response(429)  # synthetic: the keyless pool answers 429 from here
    )
    client = make_client(max_retries=2, backoff_base_seconds=1.0, host_intervals={})
    with pytest.raises(ProviderUnavailableError):
        SemanticScholarProvider(client, clock).search(TOPIC, 5, None)
    assert route.call_count == 3
    assert [s for s in sleeps if s >= 1.0] == [1.0, 2.0]


@pytest.mark.parametrize(
    ("cls", "env"),
    [
        (BraveSearchProvider, "BRAVE_SEARCH_API_KEY"),
        (TavilySearchProvider, "TAVILY_API_KEY"),
        (SerpApiSearchProvider, "SERPAPI_API_KEY"),
    ],
)
@respx.mock
def test_keyed_web_search_fails_closed_without_key(
    cls: type[BraveSearchProvider],
    env: str,
    make_client: ClientFactory,
    clock: FrozenClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(ProviderUnavailableError, match=env):
        cls(make_client(), clock).search(TOPIC, 5, None)
    assert respx.calls.call_count == 0


@respx.mock
def test_brave_sends_key_header_when_configured(
    make_client: ClientFactory, clock: FrozenClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    route = respx.get("https://api.search.brave.com/res/v1/web/search").mock(
        return_value=httpx.Response(  # synthetic payload shaped like Brave's documented schema
            200,
            json={
                "web": {
                    "results": [
                        {"title": "T", "url": "https://example.org/p", "description": "d"},
                        {"title": "bad", "url": "javascript:alert(1)", "description": "x"},
                    ]
                }
            },
        )
    )
    hits = BraveSearchProvider(make_client(), clock).search(TOPIC, 5, "en")
    assert [h.url for h in hits] == ["https://example.org/p"]
    assert route.calls.last.request.headers["X-Subscription-Token"] == "test-key"


def test_default_providers_include_only_configured_keyed_providers(
    make_client: ClientFactory, clock: FrozenClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    for env in (
        "SEMANTIC_SCHOLAR_API_KEY",
        "BRAVE_SEARCH_API_KEY",
        "TAVILY_API_KEY",
        "SERPAPI_API_KEY",
    ):
        monkeypatch.delenv(env, raising=False)
    names = [p.name for p in default_providers(make_client(), clock)]
    assert names == ["arxiv", "openalex", "crossref", "github", "wikipedia"]
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    providers = default_providers(make_client(), clock)
    assert providers[-1].name == "tavily"
    assert all(isinstance(p, SearchProvider) for p in providers)
