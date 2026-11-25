"""Deterministic data-quality checks for the perp/funding panel -> `DataQualityReport`.

PIT-relevant checks (all must pass for `pit_certified`): `available_at_lag`,
`no_future_timestamps`, `no_duplicate_timestamps`. The rest describe data quality; a failed
non-PIT check is reported but does not by itself block PIT certification.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

from enigma.data.panel import bar_length
from enigma.schemas.research import DataQualityCheck, DataQualityReport

PIT_CHECKS = ("available_at_lag", "no_future_timestamps", "no_duplicate_timestamps")
FUNDING_INTERVAL = timedelta(hours=8)
_MAX_EXAMPLES = 10


def _iso(values: list[Any]) -> list[str]:
    return [v.isoformat() if isinstance(v, datetime) else str(v) for v in values]


def _check(name: str, passed: bool, **details: Any) -> DataQualityCheck:
    return DataQualityCheck(name=name, passed=passed, details=details)


def _available_at_lag(panel: pl.DataFrame, length: timedelta) -> DataQualityCheck:
    bad = panel.filter(pl.col("available_at") < pl.col("ts") + length)
    return _check(
        "available_at_lag",
        bad.height == 0,
        bar_length_hours=length.total_seconds() / 3600,
        violations=bad.height,
        examples=_iso(bad["ts"].head(_MAX_EXAMPLES).to_list()),
    )


def _no_future(panel: pl.DataFrame, retrieved_at: datetime) -> DataQualityCheck:
    cols = ("ts", "available_at", "funding_time")
    latest = {
        c: panel[c].max()
        for c in cols
        if c in panel.columns and panel[c].null_count() < panel.height
    }
    offenders = {c: v for c, v in latest.items() if isinstance(v, datetime) and v > retrieved_at}
    return _check(
        "no_future_timestamps",
        not offenders,
        retrieved_at=retrieved_at.isoformat(),
        latest={c: v.isoformat() for c, v in latest.items() if isinstance(v, datetime)},
        offending_columns=sorted(offenders),
    )


def _no_duplicates(panel: pl.DataFrame) -> DataQualityCheck:
    dupes = panel.group_by(["symbol", "ts"]).len().filter(pl.col("len") > 1).sort(["symbol", "ts"])
    return _check(
        "no_duplicate_timestamps",
        dupes.height == 0,
        duplicated_keys=dupes.height,
        examples=[
            f"{s}@{t.isoformat()}"
            for s, t in dupes.select("symbol", "ts").head(_MAX_EXAMPLES).iter_rows()
        ],
    )


def _monotonic(panel: pl.DataFrame) -> DataQualityCheck:
    bad = panel.with_columns(
        (pl.col("ts").diff().over("symbol") <= timedelta(0)).fill_null(False).alias("_bad")
    ).filter(pl.col("_bad"))
    symbol_order_ok = panel["symbol"].is_sorted() if panel.height else True
    return _check(
        "monotonic_order",
        bad.height == 0 and symbol_order_ok,
        non_increasing_steps=bad.height,
        symbols_sorted=symbol_order_ok,
    )


def _gaps(panel: pl.DataFrame, length: timedelta) -> DataQualityCheck:
    steps = panel.with_columns(pl.col("ts").diff().over("symbol").alias("_step")).filter(
        pl.col("_step") > length
    )
    missing = steps.select(
        (
            (pl.col("_step").dt.total_microseconds() // (length // timedelta(microseconds=1))) - 1
        ).sum()
    ).item()
    return _check(
        "no_gaps",
        steps.height == 0,
        expected_step_hours=length.total_seconds() / 3600,
        gap_count=steps.height,
        missing_bars=int(missing or 0),
        examples=[
            f"{s}: gap before {t.isoformat()}"
            for s, t in steps.select("symbol", "ts").head(_MAX_EXAMPLES).iter_rows()
        ],
    )


def _ohlc(panel: pl.DataFrame) -> DataQualityCheck:
    o, h, lo, c = (pl.col(f"perp_{x}") for x in ("open", "high", "low", "close"))
    bad_expr = (
        (lo > pl.min_horizontal(o, c))
        | (h < pl.max_horizontal(o, c))
        | (lo > h)
        | (pl.min_horizontal(o, h, lo, c) <= 0)
        | (pl.col("perp_volume") < 0)
        | (pl.col("spot_close") <= 0).fill_null(False)
        | pl.any_horizontal(o.is_null(), h.is_null(), lo.is_null(), c.is_null())
    )
    bad = panel.filter(bad_expr)
    return _check(
        "ohlc_consistency",
        bad.height == 0,
        violations=bad.height,
        examples=[
            f"{s}@{t.isoformat()}"
            for s, t in bad.select("symbol", "ts").head(_MAX_EXAMPLES).iter_rows()
        ],
    )


def _outliers(panel: pl.DataFrame, max_abs_log_return: float) -> DataQualityCheck:
    """Informational: extreme moves are flagged, never removed; the check always passes."""
    flagged = (
        panel.with_columns(pl.col("perp_close").log().diff().over("symbol").alias("_r"))
        .filter(pl.col("_r").abs() > max_abs_log_return)
        .select("symbol", "ts", "_r")
    )
    return _check(
        "return_outliers",
        True,
        informational=True,
        threshold_abs_log_return=max_abs_log_return,
        flagged=flagged.height,
        examples=[
            {"symbol": s, "ts": t.isoformat(), "log_return": round(r, 6)}
            for s, t, r in flagged.head(_MAX_EXAMPLES).iter_rows()
        ],
    )


def _funding_alignment(
    panel: pl.DataFrame, funding_events: Mapping[str, pl.DataFrame] | None
) -> DataQualityCheck:
    joined_bad = panel.filter(
        pl.col("funding_time").is_not_null() & (pl.col("funding_time") != pl.col("available_at"))
    ).height
    off_grid = 0
    unmatched = 0
    examples: list[str] = []
    for symbol, events in sorted((funding_events or {}).items()):
        rows = panel.filter(pl.col("symbol") == symbol)
        if rows.height == 0 or events.height == 0:
            continue
        window = events.filter(
            (pl.col("funding_time") > rows["ts"].min())
            & (pl.col("funding_time") <= rows["available_at"].max())
        )
        micros = window["funding_time"].dt.epoch("us")
        grid = window.filter(micros % (FUNDING_INTERVAL // timedelta(microseconds=1)) != 0)
        off_grid += grid.height
        closes = set(rows["available_at"].to_list())
        missing = [t for t in window["funding_time"].to_list() if t not in closes]
        unmatched += len(missing)
        examples.extend(f"{symbol}@{t.isoformat()}" for t in missing[:_MAX_EXAMPLES])
    return _check(
        "funding_alignment",
        joined_bad == 0 and off_grid == 0 and unmatched == 0,
        joined_mismatch=joined_bad,
        events_off_8h_grid=off_grid,
        events_without_bar_close=unmatched,
        examples=examples[:_MAX_EXAMPLES],
    )


def _funding_coverage(panel: pl.DataFrame, min_coverage: float) -> DataQualityCheck:
    """Coverage over bars whose close is a scheduled funding time (00/08/16 UTC)."""
    expected = panel.filter(
        pl.col("available_at").dt.epoch("us") % (FUNDING_INTERVAL // timedelta(microseconds=1)) == 0
    )
    per_symbol: dict[str, float] = {}
    for (symbol,), group in expected.group_by(["symbol"], maintain_order=True):
        per_symbol[str(symbol)] = round(1 - group["funding_rate"].null_count() / group.height, 6)
    overall = (
        1 - expected["funding_rate"].null_count() / expected.height if expected.height else 0.0
    )
    return _check(
        "funding_coverage",
        expected.height > 0 and overall >= min_coverage,
        coverage=round(overall, 6),
        per_symbol=per_symbol,
        expected_events=expected.height,
        min_coverage=min_coverage,
    )


def _spot_coverage(panel: pl.DataFrame, min_coverage: float) -> DataQualityCheck:
    per_symbol: dict[str, dict[str, float | int]] = {}
    worst = 1.0
    for (symbol,), group in panel.group_by(["symbol"], maintain_order=True):
        missing = group["spot_close"].null_count()
        coverage = 1 - missing / group.height
        worst = min(worst, coverage)
        per_symbol[str(symbol)] = {
            "perp_bars": group.height,
            "missing_spot": missing,
            "coverage": round(coverage, 6),
        }
    return _check(
        "spot_perp_coverage",
        panel.height > 0 and worst >= min_coverage,
        per_symbol=per_symbol,
        min_coverage=min_coverage,
    )


def okx_cross_check(
    panel: pl.DataFrame,
    okx_funding: Mapping[str, pl.DataFrame],
    min_correlation: float = 0.3,
    min_overlap: int = 30,
) -> DataQualityCheck:
    """Pearson correlation of Binance vs OKX funding rates on common settlement times."""
    results: dict[str, dict[str, float | int | None]] = {}
    passed = True
    for symbol, okx in sorted(okx_funding.items()):
        binance = panel.filter(
            (pl.col("symbol") == symbol) & pl.col("funding_rate").is_not_null()
        ).select("funding_time", pl.col("funding_rate").alias("binance"))
        both = binance.join(
            okx.select("funding_time", pl.col("funding_rate").alias("okx")), on="funding_time"
        )
        corr: float | None = None
        if both.height >= 2:
            value = both.select(pl.corr("binance", "okx")).item()
            corr = None if value is None or math.isnan(value) else round(float(value), 6)
        ok = both.height >= min_overlap and corr is not None and corr >= min_correlation
        passed = passed and ok
        results[symbol] = {"overlap": both.height, "correlation": corr}
    return _check(
        "okx_funding_cross_check",
        passed and bool(results),
        per_symbol=results,
        min_correlation=min_correlation,
        min_overlap=min_overlap,
    )


def check_panel(
    panel: pl.DataFrame,
    *,
    frequency: str,
    retrieved_at: datetime,
    funding_events: Mapping[str, pl.DataFrame] | None = None,
    okx_funding: Mapping[str, pl.DataFrame] | None = None,
    max_abs_log_return: float = 0.25,
    min_funding_coverage: float = 0.99,
    min_spot_coverage: float = 0.99,
) -> DataQualityReport:
    length = bar_length(frequency)
    retrieved_at = retrieved_at.astimezone(UTC)
    checks = [
        _available_at_lag(panel, length),
        _no_future(panel, retrieved_at),
        _no_duplicates(panel),
        _monotonic(panel),
        _gaps(panel, length),
        _ohlc(panel),
        _outliers(panel, max_abs_log_return),
        _funding_alignment(panel, funding_events),
        _funding_coverage(panel, min_funding_coverage),
        _spot_coverage(panel, min_spot_coverage),
    ]
    if okx_funding is not None:
        checks.append(okx_cross_check(panel, okx_funding))
    return DataQualityReport(checks=tuple(checks))


def is_pit_certified(report: DataQualityReport) -> bool:
    by_name = {check.name: check.passed for check in report.checks}
    return all(by_name.get(name, False) for name in PIT_CHECKS)
