"""Holdout custody (ADR-015): holdout source contract, sealed manifests, validation-plane DB.

The validation service is the only component that opens the validation DB under
`validation_dir`. Research receives only `RedactedValidationSummary` documents.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import polars as pl

from enigma.core.clock import Clock
from enigma.core.errors import IntegrityError
from enigma.core.hashing import sha256_of
from enigma.isolation.hosts import assert_may_open_validation, host_role
from enigma.schemas.validation import HoldoutManifest, ValidationReport
from enigma.store.db import Database
from enigma.store.documents import DocumentStore

VALIDATION_DB_NAME = "validation.sqlite3"
BASE_COLUMNS = ("symbol", "ts", "available_at", "perp_close", "funding_rate")
_RECORDED_BY = "validation_service"


@runtime_checkable
class HoldoutSource(Protocol):
    """Validation-plane data access. Bars with start <= ts <= end, long format (see enigma.data).

    The integrator backs this with `DataCatalog(..., plane="validation")`.
    """

    def available_range(self, symbols: Sequence[str]) -> tuple[datetime, datetime]: ...

    def load_holdout(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> pl.DataFrame: ...

    def dataset_ids(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> tuple[str, ...]: ...


class InMemoryHoldoutSource:
    """Serves a fixed frame. For wiring tests and for adapters that already hold the data."""

    def __init__(self, frame: pl.DataFrame, dataset_ids: Sequence[str]) -> None:
        if not dataset_ids:
            raise ValueError("dataset_ids must be non-empty")
        self._frame = frame.sort(["symbol", "ts"])
        self._dataset_ids = tuple(dataset_ids)

    def _subset(self, symbols: Sequence[str]) -> pl.DataFrame:
        return self._frame.filter(pl.col("symbol").is_in(list(symbols)))

    def available_range(self, symbols: Sequence[str]) -> tuple[datetime, datetime]:
        sub = self._subset(symbols)
        if sub.is_empty():
            raise IntegrityError(f"no data for {list(symbols)}")
        start, end = sub.get_column("ts").min(), sub.get_column("ts").max()
        assert isinstance(start, datetime) and isinstance(end, datetime)
        return start, end

    def load_holdout(self, symbols: Sequence[str], start: datetime, end: datetime) -> pl.DataFrame:
        return self._subset(symbols).filter(pl.col("ts").is_between(start, end, closed="both"))

    def dataset_ids(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> tuple[str, ...]:
        return self._dataset_ids


def frame_content_hash(frame: pl.DataFrame) -> str:
    """Deterministic content hash of a long-format frame (row order and column order free)."""
    sub = frame.sort(["symbol", "ts"])
    digest = hashlib.sha256()
    for name in sorted(sub.columns):
        series = sub.get_column(name)
        digest.update(f"{name}:{series.dtype}".encode())
        digest.update(series.is_null().to_numpy().astype(np.uint8).tobytes())
        if isinstance(series.dtype, pl.Datetime):
            values = series.dt.epoch("ns").fill_null(0).to_numpy().astype("<i8")
            digest.update(values.tobytes())
        elif series.dtype.is_numeric():
            values = series.cast(pl.Float64).fill_null(0.0).to_numpy().astype("<f8")
            digest.update(values.tobytes())
        else:
            digest.update("\x1f".join("" if v is None else str(v) for v in series).encode())
    return "sha256:" + digest.hexdigest()


def check_frame(frame: pl.DataFrame, symbols: Sequence[str]) -> None:
    missing = [c for c in BASE_COLUMNS if c not in frame.columns]
    if missing:
        raise IntegrityError(f"holdout frame missing columns {missing}")
    present = set(frame.get_column("symbol").unique().to_list())
    absent = sorted(set(symbols) - present)
    if absent:
        raise IntegrityError(f"holdout frame missing symbols {absent}")


def holdout_split_index(n_bars: int, holdout_fraction: float) -> int:
    """Index of the first holdout bar: the last floor(fraction * n) bars are the holdout."""
    n_hold = math.floor(holdout_fraction * n_bars)
    if n_hold < 1 or n_hold >= n_bars:
        raise IntegrityError(f"cannot carve a {holdout_fraction:.0%} holdout from {n_bars} bars")
    return n_bars - n_hold


class HoldoutCustody:
    """Owns the validation-plane SQLite DB: holdout manifests and full validation reports."""

    def __init__(self, validation_dir: Path) -> None:
        assert_may_open_validation(host_role())
        validation_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(validation_dir / VALIDATION_DB_NAME)
        self.documents = DocumentStore(self.db)

    def close(self) -> None:
        self.db.close()

    @staticmethod
    def manifest_id(symbols: Sequence[str], frequency: str, protocol_version: str) -> str:
        key = sha256_of(
            {"symbols": sorted(symbols), "frequency": frequency, "protocol": protocol_version}
        )
        return "hmf_" + key.removeprefix("sha256:")[:32]

    def seal(
        self,
        symbols: Sequence[str],
        frequency: str,
        protocol_version: str,
        holdout_fraction: float,
        source: HoldoutSource,
        clock: Clock,
    ) -> tuple[HoldoutManifest, pl.DataFrame]:
        """Return the sealed manifest (creating it on first use) and the full-period frame.

        Raises IntegrityError if the holdout rows changed since the manifest was sealed.
        """
        symbols = tuple(sorted(symbols))
        manifest_id = self.manifest_id(symbols, frequency, protocol_version)
        manifest = self.documents.get(HoldoutManifest, manifest_id)
        if manifest is None:
            start, end = source.available_range(symbols)
        else:
            # Sources select bars by availability, and the last bar becomes available one bar
            # after it opens, so reload one bar past data_end and clip back to the sealed range.
            start, end = manifest.data_start, manifest.data_end + bar_delta(frequency)
        frame = source.load_holdout(symbols, start, end).sort(["symbol", "ts"])
        if manifest is not None:
            frame = frame.filter(pl.col("ts") <= manifest.data_end)
        check_frame(frame, symbols)
        bars = frame.get_column("ts").unique().sort()
        if manifest is None:
            split = holdout_split_index(bars.len(), holdout_fraction)
            holdout_start = bars[split]
            holdout_rows = frame.filter(pl.col("ts") >= holdout_start)
            manifest = HoldoutManifest(
                id=manifest_id,
                symbols=symbols,
                frequency=frequency,
                protocol_version=protocol_version,
                holdout_fraction=holdout_fraction,
                data_start=bars[0],
                data_end=bars[-1],
                holdout_start=holdout_start,
                n_bars_total=bars.len(),
                n_holdout_bars=bars.len() - split,
                holdout_content_hash=frame_content_hash(holdout_rows),
                dataset_ids=source.dataset_ids(symbols, start, end),
                created_at=clock.now(),
            )
            self.documents.put(manifest, recorded_by=_RECORDED_BY)
            return manifest, frame
        holdout_rows = frame.filter(pl.col("ts") >= manifest.holdout_start)
        if frame_content_hash(holdout_rows) != manifest.holdout_content_hash:
            raise IntegrityError(f"holdout rows changed since manifest {manifest.id} was sealed")
        return manifest, frame

    def store_report(self, report: ValidationReport) -> None:
        self.documents.put(report, recorded_by=_RECORDED_BY)

    def get_report(self, report_id: str) -> ValidationReport | None:
        return self.documents.get(ValidationReport, report_id)


_UNITS = {"m": timedelta(minutes=1), "h": timedelta(hours=1), "d": timedelta(days=1)}


def bar_delta(frequency: str) -> timedelta:
    """Bar length for frequencies like "1m", "8h" or "1d"."""
    count, unit = frequency[:-1], frequency[-1:]
    if unit not in _UNITS or not count.isdigit() or int(count) < 1:
        raise ValueError(f"unsupported bar frequency {frequency!r}")
    return int(count) * _UNITS[unit]
