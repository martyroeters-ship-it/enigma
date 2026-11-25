from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
from data_harness import fixture_bytes

from enigma.data.normalize import (
    BAR_SCHEMA,
    FUNDING_SCHEMA,
    parse_funding_csv,
    parse_klines_csv,
    read_zip_csv,
)

PERP_2020 = "futures/um/monthly/klines/BTCUSDT/8h/BTCUSDT-8h-2020-01.zip"
PERP_2024 = "futures/um/monthly/klines/BTCUSDT/8h/BTCUSDT-8h-2024-01.zip"
SPOT_2025_US = "spot/monthly/klines/BTCUSDT/8h/BTCUSDT-8h-2025-06.zip"
FUNDING_2024 = "futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip"


def _bars(relative: str) -> pl.DataFrame:
    return parse_klines_csv(read_zip_csv(fixture_bytes(relative)))


def test_headerless_2020_perp_klines() -> None:
    assert not read_zip_csv(fixture_bytes(PERP_2020)).startswith(b"open_time")
    bars = _bars(PERP_2020)
    assert dict(bars.schema) == BAR_SCHEMA
    assert bars.height == 93
    first = bars.row(0, named=True)
    assert first["ts"] == datetime(2020, 1, 1, tzinfo=UTC)
    assert first["open"] == 7189.43
    assert first["close"] == 7205.26
    assert first["trades"] == 29884
    assert first["close_time"] == datetime(2020, 1, 1, 7, 59, 59, 999000, tzinfo=UTC)


def test_header_2024_perp_klines() -> None:
    assert read_zip_csv(fixture_bytes(PERP_2024)).startswith(b"open_time")
    bars = _bars(PERP_2024)
    assert bars.height == 93
    assert bars["ts"][0] == datetime(2024, 1, 1, tzinfo=UTC)
    assert bars["ts"][-1] == datetime(2024, 1, 31, 16, tzinfo=UTC)
    assert bars["ts"].diff().drop_nulls().unique().to_list() == [timedelta(hours=8)]


def test_microsecond_spot_timestamps_2025() -> None:
    assert read_zip_csv(fixture_bytes(SPOT_2025_US)).startswith(b"1748736000000000")
    bars = _bars(SPOT_2025_US)
    assert bars["ts"][0] == datetime(2025, 6, 1, tzinfo=UTC)
    assert bars["close_time"][0] == datetime(2025, 6, 1, 7, 59, 59, 999999, tzinfo=UTC)
    assert bars.height == 90


def test_funding_snaps_jitter_to_hour() -> None:
    raw = read_zip_csv(fixture_bytes(FUNDING_2024)).decode()
    raw_times = [int(line.split(",")[0]) for line in raw.splitlines()[1:]]
    assert any(t % 3_600_000 != 0 for t in raw_times), "fixture should contain real ms jitter"
    funding = parse_funding_csv(raw.encode())
    assert dict(funding.schema) == FUNDING_SCHEMA
    assert funding.height == 93
    micros = funding["funding_time"].dt.epoch("us")
    assert (micros % (8 * 3_600_000_000) == 0).all()
    assert funding["mark_price"].null_count() == funding.height
    assert funding["funding_rate"][0] == 0.00037409


def test_funding_far_off_hour_is_not_snapped() -> None:
    csv = b"calc_time,funding_interval_hours,last_funding_rate\n1704067500000,8,0.0001\n"
    funding = parse_funding_csv(csv)
    assert funding["funding_time"][0] == datetime(2024, 1, 1, 0, 5, tzinfo=UTC)


def test_empty_csv_gives_typed_empty_frames() -> None:
    assert dict(parse_klines_csv(b"").schema) == BAR_SCHEMA
    assert parse_funding_csv(b"").height == 0
