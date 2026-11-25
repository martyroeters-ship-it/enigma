from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from enigma.data.binance import ArchiveKind, BinanceArchiveClient
from enigma.data.panel import PANEL_COLUMNS, PANEL_SCHEMA, build_perp_funding_panel

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 2, 1, tzinfo=UTC)
SYMBOLS = ("BTCUSDT", "ETHUSDT")


@pytest.fixture
def panel(client: BinanceArchiveClient) -> pl.DataFrame:
    return build_perp_funding_panel(SYMBOLS, START, END, client=client)


def test_exact_columns_dtypes_and_order(panel: pl.DataFrame) -> None:
    assert PANEL_COLUMNS == (
        "symbol",
        "ts",
        "available_at",
        "perp_open",
        "perp_high",
        "perp_low",
        "perp_close",
        "perp_volume",
        "perp_quote_volume",
        "spot_close",
        "funding_rate",
        "funding_time",
        "basis",
    )
    assert tuple(panel.columns) == PANEL_COLUMNS
    assert dict(panel.schema) == PANEL_SCHEMA
    assert panel.height == 2 * 93
    assert panel.equals(panel.sort(["symbol", "ts"]))
    assert (panel["available_at"] == panel["ts"] + timedelta(hours=8)).all()
    assert panel["available_at"].max() == END


def test_funding_aligned_to_bar_close(client: BinanceArchiveClient, panel: pl.DataFrame) -> None:
    funding = client.fetch_range(ArchiveKind.FUNDING, "BTCUSDT", START, END).frame
    btc = panel.filter(pl.col("symbol") == "BTCUSDT")
    first = btc.row(0, named=True)
    assert first["ts"] == datetime(2024, 1, 1, 0, tzinfo=UTC)
    assert first["funding_time"] == datetime(2024, 1, 1, 8, tzinfo=UTC)
    expected = funding.filter(pl.col("funding_time") == first["funding_time"])["funding_rate"][0]
    assert first["funding_rate"] == expected
    joined = btc.filter(pl.col("funding_time").is_not_null())
    assert (joined["funding_time"] == joined["available_at"]).all()
    # 2024-01-01 00:00 settles the bar that opened 2023-12-31 16:00, which is out of range.
    assert datetime(2024, 1, 1, tzinfo=UTC) not in joined["funding_time"].to_list()
    # The bar closing 2024-02-01 00:00 needs the February archive, absent from fixtures.
    last = btc.row(-1, named=True)
    assert last["available_at"] == END and last["funding_rate"] is None
    assert joined.height == 92


def test_basis_and_spot_join(panel: pl.DataFrame) -> None:
    assert panel["spot_close"].null_count() == 0
    row = panel.row(0, named=True)
    assert row["basis"] == pytest.approx(row["perp_close"] / row["spot_close"] - 1)
    assert panel["basis"].abs().max() < 0.01  # type: ignore[operator]


def test_end_is_exclusive_of_unfinished_information(client: BinanceArchiveClient) -> None:
    end = datetime(2024, 1, 10, 4, tzinfo=UTC)
    panel = build_perp_funding_panel(["BTCUSDT"], START, end, client=client)
    assert panel["available_at"].max() <= end  # type: ignore[operator]
    assert panel["ts"].max() == datetime(2024, 1, 9, 16, tzinfo=UTC)
