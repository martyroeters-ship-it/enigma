"""HTTP layer: budget, retries/backoff, size caps, raw storage, robots.txt, egress guard.

Error responses here are synthetic HTTP statuses used to exercise failure handling."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
import respx

from enigma.core.errors import BudgetExceededError, PermissionDeniedError, ProviderUnavailableError
from enigma.intelligence.http import (
    HttpClient,
    HttpStatusError,
    ResponseTooLargeError,
    RobotsDisallowedError,
    check_egress,
)

ClientFactory = Callable[..., HttpClient]


@respx.mock
def test_budget_is_a_hard_ceiling(make_client: ClientFactory) -> None:
    respx.get("https://api.example.org/a").mock(return_value=httpx.Response(200, json={}))
    client = make_client(max_requests=2)
    client.get("https://api.example.org/a")
    client.get("https://api.example.org/a")
    with pytest.raises(BudgetExceededError):
        client.get("https://api.example.org/a")
    assert client.budget.used == 2
    assert respx.calls.call_count == 2


@respx.mock
def test_retries_count_against_budget(make_client: ClientFactory) -> None:
    respx.get("https://api.example.org/a").mock(return_value=httpx.Response(503))
    client = make_client(max_requests=2, max_retries=5)
    with pytest.raises(BudgetExceededError):
        client.get("https://api.example.org/a")
    assert respx.calls.call_count == 2


@respx.mock
def test_429_backoff_honours_retry_after_then_succeeds(
    make_client: ClientFactory, sleeps: list[float]
) -> None:
    respx.get("https://api.example.org/a").mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "7"}),
            httpx.Response(500),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    client = make_client(backoff_base_seconds=1.0, default_host_interval=0.0)
    response = client.get("https://api.example.org/a")
    assert response.json() == {"ok": True}
    backoffs = [s for s in sleeps if s >= 1.0]
    assert backoffs == [7.0, 2.0]
    assert client.stats.retries["api.example.org"] == 2


@respx.mock
def test_persistent_5xx_raises_provider_unavailable(make_client: ClientFactory) -> None:
    respx.get("https://api.example.org/a").mock(return_value=httpx.Response(502))
    client = make_client(max_retries=2)
    with pytest.raises(ProviderUnavailableError):
        client.get("https://api.example.org/a")
    assert respx.calls.call_count == 3


@respx.mock
def test_transport_errors_are_retried(make_client: ClientFactory) -> None:
    respx.get("https://api.example.org/a").mock(
        side_effect=[httpx.ConnectTimeout("t"), httpx.Response(200, text="ok")]
    )
    assert make_client().get("https://api.example.org/a").content == b"ok"


@respx.mock
def test_4xx_is_not_retried(make_client: ClientFactory) -> None:
    respx.get("https://api.example.org/a").mock(return_value=httpx.Response(404))
    with pytest.raises(HttpStatusError) as info:
        make_client().get("https://api.example.org/a")
    assert info.value.status == 404
    assert respx.calls.call_count == 1


@respx.mock
def test_response_size_cap(make_client: ClientFactory) -> None:
    respx.get("https://api.example.org/big").mock(
        return_value=httpx.Response(200, content=b"x" * 5000)
    )
    client = make_client(max_bytes=1000)
    with pytest.raises(ResponseTooLargeError):
        client.get("https://api.example.org/big")
    assert client.get("https://api.example.org/big", max_bytes=10_000).content == b"x" * 5000


@respx.mock
def test_raw_responses_are_content_addressed(make_client: ClientFactory, tmp_path: Path) -> None:
    body = b'{"hello": "world"}'
    respx.get("https://api.example.org/a").mock(
        return_value=httpx.Response(200, content=body, headers={"content-type": "application/json"})
    )
    client = make_client()
    response = client.get("https://api.example.org/a")
    digest = hashlib.sha256(body).hexdigest()
    assert response.content_hash == f"sha256:{digest}"
    stored = client.raw_store.resolve(response.raw_uri)
    assert stored == tmp_path / "raw" / digest[:2] / f"{digest}.json"
    assert stored.read_bytes() == body
    assert response.raw_uri.startswith("inst:catalog/")


@respx.mock
def test_per_host_rate_limit(make_client: ClientFactory, sleeps: list[float]) -> None:
    respx.get("https://slow.example.org/a").mock(return_value=httpx.Response(200))
    respx.get("https://other.example.org/a").mock(return_value=httpx.Response(200))
    clock = iter([0.0, 0.0, 0.5, 0.5])
    client = make_client(
        host_intervals={"slow.example.org": 3.0},
        default_host_interval=0.0,
        monotonic=lambda: next(clock),
    )
    client.get("https://slow.example.org/a")
    client.get("https://other.example.org/a")
    client.get("https://slow.example.org/a")
    assert sleeps == [pytest.approx(2.5)]


@respx.mock
def test_robots_txt_disallow_is_respected(make_client: ClientFactory) -> None:
    respx.get("https://site.example.org/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nDisallow: /private\nCrawl-delay: 5\n")
    )
    page = respx.get("https://site.example.org/public/page").mock(
        return_value=httpx.Response(200, text="<p>hi</p>", headers={"content-type": "text/html"})
    )
    blocked = respx.get("https://site.example.org/private/page").mock(
        return_value=httpx.Response(200, text="secret")
    )
    client = make_client()
    with pytest.raises(RobotsDisallowedError):
        client.get("https://site.example.org/private/page", respect_robots=True)
    assert not blocked.called
    assert client.get("https://site.example.org/public/page", respect_robots=True).status == 200
    assert page.called
    assert client.host_intervals["site.example.org"] == 5.0


@respx.mock
def test_unreachable_robots_fails_closed(make_client: ClientFactory) -> None:
    respx.get("https://down.example.org/robots.txt").mock(return_value=httpx.Response(503))
    client = make_client(max_retries=0)
    with pytest.raises(RobotsDisallowedError):
        client.get("https://down.example.org/page", respect_robots=True)


@respx.mock
def test_missing_robots_allows(make_client: ClientFactory) -> None:
    respx.get("https://open.example.org/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://open.example.org/page").mock(return_value=httpx.Response(200, text="ok"))
    assert make_client().get("https://open.example.org/page", respect_robots=True).status == 200


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.org/x",
        "http://localhost:8080/admin",
        "http://127.0.0.1/",
        "http://10.1.2.3/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "https://user:pw@example.org/",
    ],
)
def test_egress_guard_blocks_unsafe_urls(url: str) -> None:
    with pytest.raises(PermissionDeniedError):
        check_egress(url, None)


def test_egress_allowlist() -> None:
    allow = frozenset({"openalex.org"})
    assert check_egress("https://api.openalex.org/works", allow) == "api.openalex.org"
    with pytest.raises(PermissionDeniedError):
        check_egress("https://evil.example.com/", allow)


@respx.mock
def test_redirects_are_rechecked_per_hop(make_client: ClientFactory) -> None:
    respx.get("https://api.example.org/r").mock(
        return_value=httpx.Response(302, headers={"location": "http://169.254.169.254/x"})
    )
    with pytest.raises(PermissionDeniedError):
        make_client().get("https://api.example.org/r")
