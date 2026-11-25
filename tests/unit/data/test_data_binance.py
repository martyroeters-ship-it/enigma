from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from data_harness import RecordedServer, make_http

from enigma.core.clock import FrozenClock
from enigma.core.errors import IntegrityError, ProviderUnavailableError
from enigma.data.binance import (
    ArchiveKind,
    BinanceArchiveClient,
    archive_url,
    month_starts,
    parse_checksum,
)
from enigma.data.http import PoliteHttp

PERP_URL = archive_url(ArchiveKind.PERP_KLINES, "BTCUSDT", "2024-01")


def test_archive_urls() -> None:
    assert PERP_URL == (
        "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/8h/BTCUSDT-8h-2024-01.zip"
    )
    assert archive_url(ArchiveKind.SPOT_KLINES, "ETHUSDT", "2026-11-23").endswith(
        "/spot/daily/klines/ETHUSDT/8h/ETHUSDT-8h-2026-11-23.zip"
    )
    assert archive_url(ArchiveKind.FUNDING, "BTCUSDT", "2024-01").endswith(
        "/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip"
    )


def test_month_starts() -> None:
    months = month_starts(datetime(2023, 11, 15, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC))
    assert [m.isoformat() for m in months] == ["2023-11-01", "2023-12-01", "2024-01-01"]


def test_parse_checksum() -> None:
    digest = "a" * 64
    assert parse_checksum(f"{digest}  BTCUSDT-8h-2024-01.zip\n") == digest
    with pytest.raises(IntegrityError):
        parse_checksum("not-a-digest file.zip")


def test_verified_download_then_cache_hit(
    client: BinanceArchiveClient, server: RecordedServer
) -> None:
    file = client.fetch_archive(ArchiveKind.PERP_KLINES, "BTCUSDT", "2024-01")
    assert file is not None and not file.cached
    assert file.sha256 == "c45c93cd0a75f2fbf9d0bf32eb49f8604fae2f33094b2f9a5bd2e648414844d0"
    assert file.path.name == f"{file.sha256}.zip"
    assert client.http_requests == 2
    assert server.seen == [f"{PERP_URL}.CHECKSUM", PERP_URL]

    again = client.fetch_archive(ArchiveKind.PERP_KLINES, "BTCUSDT", "2024-01")
    assert again is not None and again.cached
    assert again.retrieved_at == file.retrieved_at
    assert client.http_requests == 2


def test_new_client_reuses_cache_without_network(
    tmp_path: Path, server: RecordedServer, clock: FrozenClock
) -> None:
    first = BinanceArchiveClient(tmp_path / "raw", http=make_http(server), clock=clock)
    assert first.fetch_archive(ArchiveKind.FUNDING, "BTCUSDT", "2024-01") is not None
    second = BinanceArchiveClient(tmp_path / "raw", http=make_http(server), clock=clock)
    file = second.fetch_archive(ArchiveKind.FUNDING, "BTCUSDT", "2024-01")
    assert file is not None and file.cached
    assert second.http_requests == 0


def test_corrupted_cache_blob_is_redownloaded(client: BinanceArchiveClient) -> None:
    file = client.fetch_archive(ArchiveKind.PERP_KLINES, "BTCUSDT", "2024-01")
    assert file is not None
    file.path.write_bytes(b"corrupted")
    again = client.fetch_archive(ArchiveKind.PERP_KLINES, "BTCUSDT", "2024-01")
    assert again is not None and not again.cached
    assert client.http_requests == 4


def test_checksum_mismatch_raises(client: BinanceArchiveClient, server: RecordedServer) -> None:
    server.overrides[PERP_URL] = lambda _r: httpx.Response(200, content=b"tampered payload")
    with pytest.raises(IntegrityError, match="does not match CHECKSUM"):
        client.fetch_archive(ArchiveKind.PERP_KLINES, "BTCUSDT", "2024-01")
    assert not (client.raw_dir / "blobs").exists()


def test_unpublished_archive_returns_none(client: BinanceArchiveClient) -> None:
    assert client.fetch_archive(ArchiveKind.PERP_KLINES, "BTCUSDT", "2019-01") is None


def test_daily_fallback_for_current_month(
    client: BinanceArchiveClient, server: RecordedServer
) -> None:
    result = client.fetch_range(
        ArchiveKind.PERP_KLINES,
        "BTCUSDT",
        datetime(2026, 11, 22, tzinfo=UTC),
        datetime(2026, 11, 25, tzinfo=UTC),
    )
    assert [f.period for f in result.files] == ["2026-11-23"]
    assert result.frame.height == 3
    assert result.frame["ts"][0] == datetime(2026, 11, 23, tzinfo=UTC)
    assert any("2026-11-22" in url for url in result.missing)
    assert any("2026-11-24" in url for url in result.missing)
    assert not any("2026-11-25" in url for url in server.seen), "never request today's file"


def test_funding_has_no_daily_fallback(
    client: BinanceArchiveClient, server: RecordedServer
) -> None:
    result = client.fetch_range(
        ArchiveKind.FUNDING,
        "BTCUSDT",
        datetime(2026, 11, 1, tzinfo=UTC),
        datetime(2026, 11, 25, tzinfo=UTC),
    )
    assert result.frame.height == 0 and result.files == []
    assert len(result.missing) == 1
    assert not any("/daily/" in url for url in server.seen)


def test_old_missing_month_does_not_try_daily(
    client: BinanceArchiveClient, server: RecordedServer
) -> None:
    result = client.fetch_range(
        ArchiveKind.SPOT_KLINES,
        "BTCUSDT",
        datetime(2023, 12, 1, tzinfo=UTC),
        datetime(2024, 2, 1, tzinfo=UTC),
    )
    assert [f.period for f in result.files] == ["2024-01"]
    assert result.missing == [archive_url(ArchiveKind.SPOT_KLINES, "BTCUSDT", "2023-12")]
    assert not any("/daily/" in url for url in server.seen)


def test_invalid_inputs_rejected(client: BinanceArchiveClient) -> None:
    with pytest.raises(ValueError, match="symbol"):
        client.fetch_archive(ArchiveKind.PERP_KLINES, "../etc", "2024-01")
    with pytest.raises(ValueError, match="interval"):
        client.fetch_archive(ArchiveKind.PERP_KLINES, "BTCUSDT", "2024-01", interval="7m")


def test_http_retries_with_backoff_then_succeeds() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503) if len(calls) < 3 else httpx.Response(200, content=b"ok")

    http = PoliteHttp(
        transport=httpx.MockTransport(handler), min_interval_s=0.0, sleep=sleeps.append
    )
    response = http.get("https://data.binance.vision/x")
    assert response is not None and response.content == b"ok"
    assert http.requests == 3
    assert sleeps == [1.0, 2.0]


def test_http_gives_up_after_max_retries() -> None:
    http = PoliteHttp(
        transport=httpx.MockTransport(lambda _r: httpx.Response(500)),
        max_retries=2,
        sleep=lambda _s: None,
    )
    with pytest.raises(ProviderUnavailableError, match="HTTP 500"):
        http.get("https://data.binance.vision/x")
    assert http.requests == 3


def test_http_geo_block_fails_closed_without_retry() -> None:
    http = PoliteHttp(
        transport=httpx.MockTransport(lambda _r: httpx.Response(451)), sleep=lambda _s: None
    )
    with pytest.raises(ProviderUnavailableError, match="451"):
        http.get("https://fapi.binance.com/x")
    assert http.requests == 1


def test_http_rate_limit_and_user_agent() -> None:
    agents: list[str] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        agents.append(request.headers["User-Agent"])
        return httpx.Response(200)

    http = PoliteHttp(
        user_agent="Enigma-Test/1.0",
        transport=httpx.MockTransport(handler),
        min_interval_s=0.5,
        sleep=sleeps.append,
        monotonic=lambda: 100.0,
    )
    http.get("https://data.binance.vision/a")
    http.get("https://data.binance.vision/b")
    assert agents == ["Enigma-Test/1.0", "Enigma-Test/1.0"]
    assert sleeps == [0.5]
