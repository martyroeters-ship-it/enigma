"""Parse Binance archive CSVs into canonical UTC Polars frames.

Archive quirks handled here (observed in real files):
- Kline files before ~2022 have no header row; later files start with `open_time,...`.
- Spot kline timestamps are milliseconds until 2024 and microseconds from 2025; both are
  detected per value (epoch-µs values are >= 1e14, epoch-ms values are far below).
- Funding `calc_time` carries a few milliseconds of jitter after the nominal settlement hour.
"""

from __future__ import annotations

import io
import zipfile

import polars as pl

UTC_DT = pl.Datetime("us", "UTC")

KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)
FUNDING_COLUMNS = ("calc_time", "funding_interval_hours", "last_funding_rate")

BAR_SCHEMA: dict[str, pl.DataType] = {
    "ts": UTC_DT,
    "close_time": UTC_DT,
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "quote_volume": pl.Float64(),
    "trades": pl.Int64(),
}
FUNDING_SCHEMA: dict[str, pl.DataType] = {
    "funding_time": UTC_DT,
    "funding_rate": pl.Float64(),
    "mark_price": pl.Float64(),
}

# Funding events are snapped to the nearest whole hour when within this tolerance; events
# further off keep their raw time so the quality check can flag them as misaligned.
FUNDING_SNAP_TOLERANCE_MS = 60_000
_EPOCH_US_THRESHOLD = 10**14
_HOUR_MS = 3_600_000


def read_zip_csv(data: bytes) -> bytes:
    """Return the single CSV member of a Binance archive zip."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = [n for n in archive.namelist() if n.endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one CSV member, found {members}")
        return archive.read(members[0])


def _has_header(csv: bytes) -> bool:
    first = csv.lstrip()[:1]
    return bool(first) and not first.isdigit()


def epoch_to_utc(expr: pl.Expr) -> pl.Expr:
    """Epoch integer (ms or µs, detected per value) -> UTC datetime[µs]."""
    as_int = expr.cast(pl.Int64)
    micros = pl.when(as_int >= _EPOCH_US_THRESHOLD).then(as_int).otherwise(as_int * 1000)
    return micros.cast(pl.Datetime("us")).dt.replace_time_zone("UTC")


def _read_csv(csv: bytes, columns: tuple[str, ...]) -> pl.DataFrame:
    if not csv.strip():
        return pl.DataFrame(schema=dict.fromkeys(columns, pl.String()))
    header = _has_header(csv)
    frame = pl.read_csv(
        io.BytesIO(csv),
        has_header=header,
        new_columns=None if header else list(columns),
        infer_schema=False,
    )
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"archive CSV missing columns {sorted(missing)}: {frame.columns}")
    return frame


def parse_klines_csv(csv: bytes) -> pl.DataFrame:
    """Kline CSV (spot or USD-M) -> canonical bar frame sorted by `ts`."""
    raw = _read_csv(csv, KLINE_COLUMNS)
    return (
        raw.select(
            epoch_to_utc(pl.col("open_time")).alias("ts"),
            epoch_to_utc(pl.col("close_time")).alias("close_time"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.col("quote_volume").cast(pl.Float64),
            pl.col("count").cast(pl.Int64).alias("trades"),
        )
        .cast(BAR_SCHEMA)  # type: ignore[arg-type]
        .sort("ts", maintain_order=True)
    )


def parse_funding_csv(csv: bytes) -> pl.DataFrame:
    """USD-M funding CSV -> canonical funding frame sorted by `funding_time`.

    `funding_time` is `calc_time` snapped to the nearest whole hour when it lies within
    `FUNDING_SNAP_TOLERANCE_MS` of it (observed jitter is < 100 ms). Archives carry no mark
    price, so `mark_price` is null.
    """
    raw = _read_csv(csv, FUNDING_COLUMNS)
    calc = pl.col("calc_time").cast(pl.Int64)
    calc_ms = pl.when(calc >= _EPOCH_US_THRESHOLD).then(calc // 1000).otherwise(calc)
    nearest_hour = ((calc_ms + _HOUR_MS // 2) // _HOUR_MS) * _HOUR_MS
    snapped = (
        pl.when((calc_ms - nearest_hour).abs() <= FUNDING_SNAP_TOLERANCE_MS)
        .then(nearest_hour)
        .otherwise(calc_ms)
    )
    return (
        raw.select(
            epoch_to_utc(snapped * 1000).alias("funding_time"),
            pl.col("last_funding_rate").cast(pl.Float64).alias("funding_rate"),
            pl.lit(None, dtype=pl.Float64).alias("mark_price"),
        )
        .cast(FUNDING_SCHEMA)  # type: ignore[arg-type]
        .sort("funding_time", maintain_order=True)
    )
