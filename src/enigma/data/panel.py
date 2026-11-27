"""Long-format perp/spot/funding panel: the contract WS6/WS7 code against.

Row semantics (one row per `symbol` and bar):
- `ts` is the bar open (UTC); `available_at = ts + bar_length` is the bar close, the instant at
  which every field in the row is known. Strategies may only act on a row at or after
  `available_at`.
- `funding_rate` is the rate settled at the Binance funding event whose time equals
  `available_at`. Binance USD-M settles BTC/ETH funding at 00:00, 08:00 and 16:00 UTC, which are
  exactly the closes of the 8h bars opening at 16:00 (previous day), 00:00 and 08:00. So the
  event at 08:00 lands on the bar [00:00, 08:00). Bars whose close has no funding event (or
  whose event is missing from the archives) carry null `funding_rate` and `funding_time`.
  Funding events not on a bar close are not joined; `quality.check_panel` flags them.
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
    """Bars in a 365-day year. The pinned 8h protocol uses 3 * 365 = 1095."""
    per_day = timedelta(days=1) / bar_length(frequency)
    if per_day != int(per_day):
        raise ValueError(f"{frequency} does not divide a day")
    return int(per_day) * 365


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
        events = funding.get(symbol)
        if events is not None and events.height:
            frame = frame.join(
                events.select(
                    pl.col("funding_time").alias("available_at"), "funding_rate", "funding_time"
                ),
                on="available_at",
                how="left",
            )
        else:
            frame = frame.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("funding_rate"),
                pl.lit(None, dtype=UTC_DT).alias("funding_time"),
            )
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
