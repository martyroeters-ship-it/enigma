"""Long-format perp/spot/funding panel: the contract WS6/WS7 code against.

Row semantics (one row per `symbol` and bar):
- `ts` is the bar open (UTC); `available_at = ts + bar_length` is the bar close, the instant at
  which every field in the row is known. Strategies may only act on a row at or after
  `available_at`.
- `funding_rate` is the sum of Binance funding settlements with `ts < funding_time <=
  available_at`. BTC/ETH settle at 00:00, 08:00 and 16:00 UTC. On an 8h grid that is exactly
  one settlement, at the bar close. On a 1h grid only the bar that closes on a settlement
  carries a rate; the other hours are null, and the engine treats null as no payment. On a
  1d grid the bar sums every settlement inside the day. A single 8h print is never repeated
  on every shorter bar. `funding_time` is the last settlement included in that sum.
- `basis = perp_close / spot_close - 1`, both closes taken at `available_at`.
- Range: rows with `ts >= start` and `available_at <= end`, so no row carries information
  from after `end` (research catalogs cap `end` before the holdout).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import polars as pl

from enigma.core.config import Settings
from enigma.data.binance import ArchiveFile, ArchiveKind, BinanceArchiveClient
from enigma.data.normalize import UTC_DT

PANEL_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String(),
    "ts": UTC_DT,
    "available_at": UTC_DT,
    "perp_open": pl.Float64(),
    "perp_high": pl.Float64(),
    "perp_low": pl.Float64(),
    "perp_close": pl.Float64(),
    "perp_volume": pl.Float64(),
    "perp_quote_volume": pl.Float64(),
    "spot_close": pl.Float64(),
    "funding_rate": pl.Float64(),
    "funding_time": UTC_DT,
    "basis": pl.Float64(),
}
PANEL_COLUMNS = tuple(PANEL_SCHEMA)
PANEL_SCHEMA_VERSION = "1.0.0"

_FREQUENCIES = {
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "6h": timedelta(hours=6),
    "8h": timedelta(hours=8),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
}


def bar_length(frequency: str) -> timedelta:
    try:
        return _FREQUENCIES[frequency]
    except KeyError as exc:
        raise ValueError(f"unsupported frequency {frequency!r}") from exc


def periods_per_year(frequency: str) -> int:
    """Bars in a 365-day year. Eight-hour bars are 3 * 365 = 1095, matching protocol 0.1.0-slice."""
    seconds = bar_length(frequency).total_seconds()
    per_day = 86400 / seconds
    if abs(per_day - round(per_day)) > 1e-9:
        raise ValueError(f"frequency {frequency!r} does not divide a UTC day")
    return round(per_day) * 365


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("datetimes must be timezone-aware")
    return value.astimezone(UTC)


@dataclass
class PanelSources:
    """Normalized per-symbol inputs and the archives they were read from."""

    perp: dict[str, pl.DataFrame]
    spot: dict[str, pl.DataFrame]
    funding: dict[str, pl.DataFrame]
    files: list[ArchiveFile] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def fetch_perp_funding_sources(
    client: BinanceArchiveClient,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    frequency: str = "8h",
) -> PanelSources:
    start, end = _utc(start), _utc(end)
    bar_length(frequency)
    sources = PanelSources(perp={}, spot={}, funding={})
    for symbol in symbols:
        for kind, target in (
            (ArchiveKind.PERP_KLINES, sources.perp),
            (ArchiveKind.SPOT_KLINES, sources.spot),
            (ArchiveKind.FUNDING, sources.funding),
        ):
            # Funding settled exactly at `end` closes the last bar, so include that instant.
            range_end = end + timedelta(microseconds=1) if kind is ArchiveKind.FUNDING else end
            fetched = client.fetch_range(kind, symbol, start, range_end, frequency)
            target[symbol] = fetched.frame
            sources.files.extend(fetched.files)
            sources.missing.extend(fetched.missing)
    return sources


def _funding_at_close(frame: pl.DataFrame, events: pl.DataFrame | None) -> pl.DataFrame:
    """Historical 8h join: the settlement whose time equals the bar close, unchanged.

    Newly acquired 8h panels must keep the pinned content hash. Do not replace this
    with a sum that would perturb those bits.
    """
    if events is None or events.height == 0:
        return frame.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("funding_rate"),
            pl.lit(None, dtype=UTC_DT).alias("funding_time"),
        )
    return frame.join(
        events.select(pl.col("funding_time").alias("available_at"), "funding_rate", "funding_time"),
        on="available_at",
        how="left",
    )


def _funding_inside_bar(frame: pl.DataFrame, events: pl.DataFrame | None) -> pl.DataFrame:
    """Sum the settlements with bar open < funding_time <= bar close.

    Used for sampling frequencies other than 8h. A 1h bar therefore carries a rate
    only when a settlement falls inside that hour. A 1d bar sums the settlements in
    the day. The recorded rate is the sum of those prints, not a repeated 8h rate.
    """
    blank = frame.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("funding_rate"),
        pl.lit(None, dtype=UTC_DT).alias("funding_time"),
    )
    if events is None or events.height == 0:
        return blank
    settled = events.select("funding_time", "funding_rate").drop_nulls(
        ["funding_time", "funding_rate"]
    )
    if settled.height == 0:
        return blank
    paired = (
        frame.select("ts", "available_at")
        .join(settled, how="cross")
        .filter(
            (pl.col("funding_time") > pl.col("ts"))
            & (pl.col("funding_time") <= pl.col("available_at"))
        )
    )
    if paired.height == 0:
        return blank
    totals = paired.group_by("ts").agg(
        pl.col("funding_rate").sum(),
        pl.col("funding_time").max(),
    )
    return frame.join(totals, on="ts", how="left")


def _attach_funding(
    frame: pl.DataFrame, events: pl.DataFrame | None, length: timedelta
) -> pl.DataFrame:
    if length == timedelta(hours=8):
        return _funding_at_close(frame, events)
    return _funding_inside_bar(frame, events)


def assemble_perp_funding_panel(
    perp: Mapping[str, pl.DataFrame],
    spot: Mapping[str, pl.DataFrame],
    funding: Mapping[str, pl.DataFrame],
    start: datetime,
    end: datetime,
    frequency: str = "8h",
) -> pl.DataFrame:
    """Join canonical bar and funding frames into the panel (see module docstring)."""
    start, end = _utc(start), _utc(end)
    length = bar_length(frequency)
    parts: list[pl.DataFrame] = []
    for symbol in sorted(perp):
        bars = perp[symbol].filter((pl.col("ts") >= start) & (pl.col("ts") + length <= end))
        frame = bars.select(
            pl.lit(symbol, dtype=pl.String).alias("symbol"),
            pl.col("ts"),
            (pl.col("ts") + length).alias("available_at"),
            pl.col("open").alias("perp_open"),
            pl.col("high").alias("perp_high"),
            pl.col("low").alias("perp_low"),
            pl.col("close").alias("perp_close"),
            pl.col("volume").alias("perp_volume"),
            pl.col("quote_volume").alias("perp_quote_volume"),
        )
        spot_bars = spot.get(symbol)
        if spot_bars is not None and spot_bars.height:
            frame = frame.join(
                spot_bars.select("ts", pl.col("close").alias("spot_close")), on="ts", how="left"
            )
        else:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("spot_close"))
        frame = _attach_funding(frame, funding.get(symbol), length)
        frame = frame.with_columns(
            (pl.col("perp_close") / pl.col("spot_close") - 1.0).alias("basis")
        )
        parts.append(frame.select(PANEL_COLUMNS).cast(PANEL_SCHEMA))  # type: ignore[arg-type]
    if not parts:
        return pl.DataFrame(schema=PANEL_SCHEMA)
    return pl.concat(parts, how="vertical").sort(["symbol", "ts"], maintain_order=True)


def build_perp_funding_panel(
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    frequency: str = "8h",
    *,
    client: BinanceArchiveClient | None = None,
) -> pl.DataFrame:
    """Download (or reuse cached) archives and return the panel for `symbols`.

    Without `client`, archives are cached under `$ENIGMA_RESEARCH_DIR/raw/binance`.
    """
    if client is None:
        client = BinanceArchiveClient(Settings.from_env().research_dir / "raw" / "binance")
    sources = fetch_perp_funding_sources(client, symbols, start, end, frequency)
    return assemble_perp_funding_panel(
        sources.perp, sources.spot, sources.funding, start, end, frequency
    )
