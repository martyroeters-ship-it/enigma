"""Funding follows the sampling bar, and real BTC and ETH archives are distinct series."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from enigma.data.binance import ArchiveKind, BinanceArchiveClient
from enigma.data.panel import assemble_perp_funding_panel
from enigma.research.scan import scan_rows

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 2, 1, tzinfo=UTC)


def _bars(timestamps: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": timestamps,
            "open": [100.0] * len(timestamps),
            "high": [101.0] * len(timestamps),
            "low": [99.0] * len(timestamps),
            "close": [100.0] * len(timestamps),
            "volume": [1.0] * len(timestamps),
            "quote_volume": [1.0] * len(timestamps),
        }
    )


def _funding(stamps: list[datetime], rates: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"funding_time": stamps, "funding_rate": rates})


def test_daily_bar_sums_the_settlements_inside_it_and_an_hourly_bar_does_not() -> None:
    day = datetime(2024, 1, 1, tzinfo=UTC)
    settlements = [
        day,
        day + timedelta(hours=8),
        day + timedelta(hours=16),
        day + timedelta(days=1),
    ]
    rates = [0.010, 0.001, 0.002, 0.003]
    events = _funding(settlements, rates)
    spot = _bars([day]).select("ts", pl.col("close").alias("close"))
    daily = assemble_perp_funding_panel(
        {"BTCUSDT": _bars([day])},
        {"BTCUSDT": spot},
        {"BTCUSDT": events},
        day,
        day + timedelta(days=1),
        "1d",
    )
    assert daily.height == 1
    assert daily["funding_rate"][0] == pytest.approx(0.006)
    assert daily["funding_time"][0] == day + timedelta(days=1)

    hour_bars = _bars([day + timedelta(hours=7), day + timedelta(hours=8)])
    hour = assemble_perp_funding_panel(
        {"BTCUSDT": hour_bars},
        {"BTCUSDT": hour_bars.select("ts", "close")},
        {"BTCUSDT": events},
        day + timedelta(hours=7),
        day + timedelta(hours=9),
        "1h",
    )
    funded = hour.filter(pl.col("funding_rate").is_not_null())
    assert funded.height == 1
    assert funded["funding_rate"][0] == pytest.approx(0.001)
    assert funded["available_at"][0] == day + timedelta(hours=8)
    quiet = hour.filter(pl.col("funding_rate").is_null())
    assert quiet.height == 1


def test_real_btc_and_eth_funding_archives_are_not_one_series(client: BinanceArchiveClient) -> None:
    btc = client.fetch_range(ArchiveKind.FUNDING, "BTCUSDT", START, END).frame
    eth = client.fetch_range(ArchiveKind.FUNDING, "ETHUSDT", START, END).frame
    assert btc.height > 10
    assert eth.height > 10
    assert btc["funding_rate"].to_list() != eth["funding_rate"].to_list()
    btc_rows = [
        {"available_at": row["funding_time"], "funding_rate": row["funding_rate"]}
        for row in btc.iter_rows(named=True)
    ]
    cutoff = max(row["available_at"] for row in btc_rows)
    once = scan_rows(btc_rows, symbol="BTCUSDT", field="funding_rate", cutoff=cutoff, threshold=0.0)
    doubled = scan_rows(
        btc_rows + btc_rows,
        symbol="BTCUSDT",
        field="funding_rate",
        cutoff=cutoff,
        threshold=0.0,
    )
    assert [item.zscore for item in once] == [item.zscore for item in doubled]
    assert [item.sample_size for item in once] == [item.sample_size for item in doubled]
