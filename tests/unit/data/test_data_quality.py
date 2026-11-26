from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from enigma.data.binance import BinanceArchiveClient
from enigma.data.okx import OkxFundingClient
from enigma.data.panel import assemble_perp_funding_panel, fetch_perp_funding_sources
from enigma.data.quality import check_panel, is_pit_certified, okx_cross_check
from enigma.schemas.research import DataQualityReport

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 31, 16, tzinfo=UTC)
RETRIEVED = datetime(2026, 11, 25, tzinfo=UTC)


@pytest.fixture
def sources_and_panel(
    client: BinanceArchiveClient,
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    sources = fetch_perp_funding_sources(client, ["BTCUSDT", "ETHUSDT"], START, END)
    panel = assemble_perp_funding_panel(sources.perp, sources.spot, sources.funding, START, END)
    return sources.funding, panel


def _report(
    panel: pl.DataFrame,
    funding: dict[str, pl.DataFrame] | None = None,
    retrieved_at: datetime = RETRIEVED,
) -> DataQualityReport:
    return check_panel(panel, frequency="8h", retrieved_at=retrieved_at, funding_events=funding)


def _by_name(report: DataQualityReport) -> dict[str, bool]:
    return {c.name: c.passed for c in report.checks}


def _details(report: DataQualityReport, name: str) -> dict[str, object]:
    return next(c.details for c in report.checks if c.name == name)


def test_clean_real_panel_passes_everything(
    sources_and_panel: tuple[dict[str, pl.DataFrame], pl.DataFrame],
) -> None:
    funding, panel = sources_and_panel
    report = _report(panel, funding)
    assert report.passed, [c for c in report.checks if not c.passed]
    assert is_pit_certified(report)
    assert _details(report, "funding_coverage")["coverage"] == 1.0


def test_duplicate_row_breaks_pit(
    sources_and_panel: tuple[dict[str, pl.DataFrame], pl.DataFrame],
) -> None:
    _, panel = sources_and_panel
    defective = pl.concat([panel, panel.slice(10, 1)]).sort(["symbol", "ts"])
    report = _report(defective)
    assert not _by_name(report)["no_duplicate_timestamps"]
    assert not _by_name(report)["monotonic_order"]
    assert not is_pit_certified(report)


def test_gap_is_flagged_but_not_a_pit_failure(
    sources_and_panel: tuple[dict[str, pl.DataFrame], pl.DataFrame],
) -> None:
    funding, panel = sources_and_panel
    gapped = panel.filter(pl.col("ts") != datetime(2024, 1, 5, 8, tzinfo=UTC))
    report = _report(gapped, funding)
    checks = _by_name(report)
    assert not checks["no_gaps"]
    assert _details(report, "no_gaps")["missing_bars"] == 2
    assert not checks["funding_alignment"], "funding events inside the gap have no bar"
    assert is_pit_certified(report)


def test_future_timestamp_breaks_pit(
    sources_and_panel: tuple[dict[str, pl.DataFrame], pl.DataFrame],
) -> None:
    _, panel = sources_and_panel
    report = _report(panel, retrieved_at=datetime(2024, 1, 20, tzinfo=UTC))
    assert not _by_name(report)["no_future_timestamps"]
    assert "available_at" in _details(report, "no_future_timestamps")["offending_columns"]  # type: ignore[operator]
    assert not is_pit_certified(report)


def test_early_available_at_breaks_pit(
    sources_and_panel: tuple[dict[str, pl.DataFrame], pl.DataFrame],
) -> None:
    _, panel = sources_and_panel
    leaky = panel.with_columns(
        pl.when(pl.int_range(pl.len()) == 3)
        .then(pl.col("ts") + timedelta(hours=7))
        .otherwise(pl.col("available_at"))
        .alias("available_at")
    )
    report = _report(leaky)
    assert not _by_name(report)["available_at_lag"]
    assert not is_pit_certified(report)


def test_bad_ohlc_flagged(sources_and_panel: tuple[dict[str, pl.DataFrame], pl.DataFrame]) -> None:
    _, panel = sources_and_panel
    bad = panel.with_columns(
        pl.when(pl.int_range(pl.len()) == 5)
        .then(pl.col("perp_high") * 2)
        .otherwise(pl.col("perp_low"))
        .alias("perp_low")
    )
    report = _report(bad)
    assert not _by_name(report)["ohlc_consistency"]
    assert _details(report, "ohlc_consistency")["violations"] == 1
    negative = panel.with_columns(
        pl.when(pl.int_range(pl.len()) == 7)
        .then(-1.0)
        .otherwise(pl.col("spot_close"))
        .alias("spot_close")
    )
    assert not _by_name(_report(negative))["ohlc_consistency"]


def test_outlier_flagged_not_deleted(
    sources_and_panel: tuple[dict[str, pl.DataFrame], pl.DataFrame],
) -> None:
    _, panel = sources_and_panel
    spiked = panel.with_columns(
        pl.when(pl.int_range(pl.len()) == 20)
        .then(pl.col("perp_close") * 1.5)
        .otherwise(pl.col("perp_close"))
        .alias("perp_close")
    )
    report = _report(spiked)
    assert _by_name(report)["return_outliers"]
    assert _details(report, "return_outliers")["flagged"] == 2


def test_misaligned_funding_event_flagged(
    sources_and_panel: tuple[dict[str, pl.DataFrame], pl.DataFrame],
) -> None:
    funding, panel = sources_and_panel
    shifted = dict(funding)
    shifted["BTCUSDT"] = funding["BTCUSDT"].with_columns(
        pl.when(pl.int_range(pl.len()) == 4)
        .then(pl.col("funding_time") + timedelta(hours=4))
        .otherwise(pl.col("funding_time"))
        .alias("funding_time")
    )
    report = _report(panel, shifted)
    assert not _by_name(report)["funding_alignment"]
    assert _details(report, "funding_alignment")["events_off_8h_grid"] == 1


def test_funding_and_spot_coverage(
    sources_and_panel: tuple[dict[str, pl.DataFrame], pl.DataFrame],
) -> None:
    _, panel = sources_and_panel
    holes = panel.with_columns(
        pl.when(pl.int_range(pl.len()) % 10 == 0)
        .then(None)
        .otherwise(pl.col("funding_rate"))
        .alias("funding_rate"),
        pl.when(pl.int_range(pl.len()) % 20 == 0)
        .then(None)
        .otherwise(pl.col("spot_close"))
        .alias("spot_close"),
    )
    checks = _by_name(_report(holes))
    assert not checks["funding_coverage"]
    assert not checks["spot_perp_coverage"]


def test_okx_cross_check_with_recorded_okx_page(
    client: BinanceArchiveClient,
) -> None:
    okx = OkxFundingClient(http=client.http)
    okx_frame = okx.fetch_funding_history(
        "BTC-USDT-SWAP", datetime(2026, 11, 1, tzinfo=UTC), datetime(2026, 12, 1, tzinfo=UTC)
    )
    binance_like = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * okx_frame.height,
            "funding_time": okx_frame["funding_time"],
            "funding_rate": okx_frame["funding_rate"],
        }
    )
    check = okx_cross_check(binance_like, {"BTCUSDT": okx_frame}, min_overlap=5)
    assert check.passed
    assert check.details["per_symbol"]["BTCUSDT"]["correlation"] == pytest.approx(1.0)
    too_few = okx_cross_check(binance_like, {"BTCUSDT": okx_frame})
    assert not too_few.passed
