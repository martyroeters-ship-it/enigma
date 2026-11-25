"""Live tests against data.binance.vision and OKX public REST (no keys, no paid data)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from enigma.data.binance import ArchiveKind, BinanceArchiveClient
from enigma.data.catalog import DataCatalog
from enigma.data.okx import OkxFundingClient
from enigma.data.panel import PANEL_COLUMNS, build_perp_funding_panel
from enigma.schemas.common import AgentRole, Plane

pytestmark = pytest.mark.network

START = datetime(2024, 3, 1, tzinfo=UTC)
END = datetime(2024, 4, 1, tzinfo=UTC)


def test_live_month_panel(tmp_path: Path) -> None:
    client = BinanceArchiveClient(tmp_path / "raw")
    panel = build_perp_funding_panel(["BTCUSDT", "ETHUSDT"], START, END, client=client)
    assert tuple(panel.columns) == PANEL_COLUMNS
    assert panel.group_by("symbol").len().sort("symbol")["len"].to_list() == [93, 93]
    assert panel["spot_close"].null_count() == 0
    # The event settling the last bar (2024-04-01 00:00) comes from the April funding archive.
    assert panel["funding_rate"].null_count() == 0
    assert client.http_requests == 16


def test_live_catalog_acquire_with_okx_cross_check(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    first_of_month = datetime(now.year, now.month, 1, tzinfo=UTC)
    start = (first_of_month - timedelta(days=40)).replace(day=1)
    catalog = DataCatalog(
        tmp_path, None, Plane.RESEARCH, AgentRole.DATA_ENGINEERING, okx_cross_check=True
    )
    ds = catalog.acquire_perp_funding_panel(["BTCUSDT", "ETHUSDT"], start, first_of_month)
    assert ds.pit_certified
    checks = {c.name: c for c in ds.quality.checks}
    assert checks["okx_funding_cross_check"].passed, checks["okx_funding_cross_check"].details
    assert checks["no_gaps"].passed and checks["ohlc_consistency"].passed
    assert catalog.load(ds).height == ds.row_count


def test_live_daily_fallback_for_current_month(tmp_path: Path) -> None:
    client = BinanceArchiveClient(tmp_path / "raw")
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    result = client.fetch_range(
        ArchiveKind.SPOT_KLINES, "BTCUSDT", today - timedelta(days=3), today
    )
    assert result.frame.height > 0
    assert all(len(f.period) in (7, 10) for f in result.files)


def test_live_okx_funding_history() -> None:
    now = datetime.now(UTC)
    frame = OkxFundingClient().fetch_funding_history("ETH-USDT-SWAP", now - timedelta(days=30), now)
    assert frame.height >= 80
    assert isinstance(frame["funding_time"].dtype, pl.Datetime)
