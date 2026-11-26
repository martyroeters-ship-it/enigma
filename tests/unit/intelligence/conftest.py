"""Offline fixtures. Responses under tests/fixtures/intelligence are RECORDED real API responses
(see record_fixtures.py); they are replayed with respx and never presented as live data."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from enigma.core.clock import FrozenClock
from enigma.intelligence.http import HttpBudget, HttpClient, RawStore

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "intelligence"


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _json(name: str) -> httpx.Response:
    return httpx.Response(
        200, content=fixture_bytes(name), headers={"content-type": "application/json"}
    )


EMPTY_OPENALEX = httpx.Response(
    200, json={"meta": {"count": 0}, "results": []}, headers={"content-type": "application/json"}
)


def openalex_route(request: httpx.Request) -> httpx.Response:
    flt = request.url.params.get("filter", "")
    if flt.startswith("openalex_id:"):
        return _json("openalex_refs.json")
    if flt.startswith("cites:"):
        return _json("openalex_cites.json")
    if flt.startswith(("author.id:", "language:")):
        return EMPTY_OPENALEX
    return _json("openalex_search.json")


def wikipedia_route(request: httpx.Request) -> httpx.Response:
    lang = request.url.host.split(".")[0]
    params = request.url.params
    if params.get("list") == "search":
        return _json(f"wikipedia_{lang}_search.json")
    if params.get("prop") == "langlinks":
        return _json("wikipedia_en_langlinks_de.json")
    return _json(f"wikipedia_{lang}_extracts.json")


def install_recorded_routes(router: respx.MockRouter) -> None:
    router.get("https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(
            200,
            content=fixture_bytes("arxiv_search.xml"),
            headers={"content-type": "application/atom+xml; charset=UTF-8"},
        )
    )
    router.get("https://api.openalex.org/works").mock(side_effect=openalex_route)
    router.get("https://api.crossref.org/works").mock(return_value=_json("crossref_search.json"))
    router.get("https://api.github.com/search/repositories").mock(
        return_value=_json("github_search.json")
    )
    router.get(url__regex=r"https://(en|de)\.wikipedia\.org/w/api\.php.*").mock(
        side_effect=wikipedia_route
    )


@pytest.fixture
def recorded_api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        install_recorded_routes(router)
        yield router


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(datetime(2026, 11, 25, 12, 0, tzinfo=UTC), step=timedelta(milliseconds=1))


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def make_client(tmp_path: Path, sleeps: list[float]) -> Iterator[Callable[..., HttpClient]]:
    created: list[HttpClient] = []

    def factory(max_requests: int = 200, **kwargs: object) -> HttpClient:
        client = HttpClient(
            HttpBudget(max_requests),
            RawStore(tmp_path / "raw"),
            sleeper=sleeps.append,
            **kwargs,  # type: ignore[arg-type]
        )
        created.append(client)
        return client

    yield factory
    for c in created:
        c.close()
