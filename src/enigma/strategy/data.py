"""Conversion of long-format panel frames into per-symbol strategy input arrays."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from enigma.strategy.interface import NAT_INT64


def symbol_arrays(
    frame: pl.DataFrame, symbol: str, columns: Sequence[str]
) -> dict[str, np.ndarray]:
    """Columns for one symbol, ordered by `ts`. Datetimes -> int64 ns, null floats -> NaN."""
    sub = frame.filter(pl.col("symbol") == symbol).sort("ts")
    arrays: dict[str, np.ndarray] = {}
    for column in columns:
        if column not in sub.columns:
            raise KeyError(f"column {column!r} not in frame")
        series = sub.get_column(column)
        if series.dtype == pl.Datetime or isinstance(series.dtype, pl.Datetime):
            values = series.dt.epoch("ns").fill_null(NAT_INT64).to_numpy()
            arrays[column] = np.ascontiguousarray(values, dtype=np.int64)
        elif series.dtype.is_numeric() or series.dtype == pl.Boolean:
            values = series.cast(pl.Float64).fill_null(float("nan")).to_numpy()
            arrays[column] = np.ascontiguousarray(values, dtype=np.float64)
        else:
            raise TypeError(f"column {column!r} has non-numeric dtype {series.dtype}")
    return arrays
