from __future__ import annotations

from datetime import UTC, datetime

import pytest
from data_harness import RecordedServer, make_http

from enigma.data.okx import OkxFundingClient, okx_swap_id


def test_okx_swap_id() -> None:
    assert okx_swap_id("BTCUSDT") == "BTC-USDT-SWAP"
    assert okx_swap_id("ETHUSDT") == "ETH-USDT-SWAP"
    with pytest.raises(ValueError):
        okx_swap_id("USDT")


def test_paginates_recorded_pages(server: RecordedServer) -> None:
    client = OkxFundingClient(http=make_http(server), page_limit=3)
    frame = client.fetch_funding_history(
        "BTC-USDT-SWAP", datetime(2026, 1, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC)
    )
    assert frame.height == 6
    assert frame["funding_time"].is_sorted()
    assert frame["funding_time"].n_unique() == 6
    assert client.http_requests == 3  # two recorded pages, then an empty page ends the walk
    assert "after=" in server.seen[1]


def test_range_filter(server: RecordedServer) -> None:
    client = OkxFundingClient(http=make_http(server), page_limit=3)
    full = client.fetch_funding_history(
        "BTC-USDT-SWAP", datetime(2026, 1, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC)
    )
    cutoff = full["funding_time"][3]
    part = client.fetch_funding_history("BTC-USDT-SWAP", cutoff, datetime(2027, 1, 1, tzinfo=UTC))
    assert part.height == 3
