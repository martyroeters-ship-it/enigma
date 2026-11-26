"""Deterministic content hash of a canonical Polars frame, independent of Parquet encoding.

The frame is sorted by its key columns, then each column is serialized in order as
`name | dtype | null bitmap | values`, with values in a fixed little-endian binary form:
datetimes as int64 epoch microseconds UTC, floats as IEEE-754 float64 bits, integers as int64,
strings as length-prefixed UTF-8. Null slots serialize as zero. Identical data therefore hashes
identically across runs, machines and library versions.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence

import numpy as np
import polars as pl

HASH_SCHEME = "enigma-frame-v1"


def _column_bytes(series: pl.Series) -> bytes:
    nulls = series.is_null().to_numpy().astype(np.uint8)
    dtype = series.dtype
    if dtype == pl.String:
        parts = []
        for value in series.to_list():
            raw = b"" if value is None else str(value).encode("utf-8")
            parts.append(struct.pack("<q", len(raw)) + raw)
        values = b"".join(parts)
    elif isinstance(dtype, pl.Datetime):
        if dtype.time_zone not in (None, "UTC"):
            raise ValueError(f"column {series.name} must be UTC, got {dtype.time_zone}")
        epoch = series.dt.epoch("us").fill_null(0).to_numpy().astype("<i8")
        values = epoch.tobytes()
    elif dtype.is_float():
        values = series.cast(pl.Float64).fill_null(0.0).to_numpy().astype("<f8").tobytes()
    elif dtype.is_integer() or dtype == pl.Boolean:
        values = series.cast(pl.Int64).fill_null(0).to_numpy().astype("<i8").tobytes()
    else:
        raise TypeError(f"unsupported dtype {dtype} in column {series.name}")
    kind = "datetime_us_utc" if isinstance(dtype, pl.Datetime) else str(dtype)
    header = f"{series.name}|{kind}|{series.len()}".encode()
    return header + b"|" + nulls.tobytes() + b"|" + values


def frame_content_hash(frame: pl.DataFrame, sort_by: Sequence[str] = ("symbol", "ts")) -> str:
    """`sha256:<hex>` of the canonical serialization of `frame` (sorted by `sort_by`)."""
    canonical = frame.sort(list(sort_by), maintain_order=True) if frame.height else frame
    digest = hashlib.sha256(HASH_SCHEME.encode())
    digest.update(struct.pack("<q", canonical.height))
    for name in canonical.columns:
        digest.update(_column_bytes(canonical[name]))
    return "sha256:" + digest.hexdigest()
