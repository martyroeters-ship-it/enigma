"""Plane-scoped catalog of immutable, content-addressed dataset versions.

Layout under `root`:
- `raw/binance/` — verified archive cache (see `enigma.data.binance`).
- `datasets/<name>/<hash16>/data.parquet` and `dataset.json` (the `DatasetVersion` manifest).

Research and validation each instantiate their own catalog with their own `root`, so the
validation plane acquires holdout data independently of the research cache. Holdout policy
(which `end` research may use) is enforced by callers, not here.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from importlib import metadata
from pathlib import Path
from typing import Any

import polars as pl

from enigma.core.clock import Clock, SystemClock
from enigma.core.errors import IntegrityError, ProviderUnavailableError
from enigma.core.ids import new_id
from enigma.core.paths import StorageRoots, bind_uri, dataset_parquet, resolve_uri, within
from enigma.data.binance import BinanceArchiveClient
from enigma.data.hashing import HASH_SCHEME, frame_content_hash
from enigma.data.instruments import PERP_VENUE, binance_perp, binance_spot
from enigma.data.okx import OkxFundingClient, okx_swap_id
from enigma.data.panel import (
    PANEL_COLUMNS,
    PANEL_SCHEMA,
    PANEL_SCHEMA_VERSION,
    assemble_perp_funding_panel,
    bar_length,
    fetch_perp_funding_sources,
)
from enigma.data.quality import check_panel, is_pit_certified
from enigma.schemas.common import Actor, Instrument, Plane
from enigma.schemas.governance import CostRecord
from enigma.schemas.research import DatasetVersion
from enigma.store.documents import DocumentStore
from enigma.store.ledger import CostLedger

SURVIVORSHIP_NOTES = (
    "Universe fixed ex ante to BTCUSDT and ETHUSDT (Binance USD-M perpetuals and spot), the "
    "largest and still-listed contracts; delisted or newer contracts are excluded, so results "
    "are conditional on these assets having survived. Not selected on performance."
)
OKX_HISTORY_DAYS = 90


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("datetimes must be timezone-aware")
    return value.astimezone(UTC)


def _library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("enigma", "polars", "pyarrow", "httpx", "numpy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def default_dataset_name(
    symbols: Sequence[str], start: datetime, end: datetime, frequency: str
) -> str:
    joined = "-".join(s.lower() for s in symbols)
    return f"binance_perp_funding_{frequency}_{joined}_{start:%Y%m%d}_{end:%Y%m%d}"


class DataCatalog:
    def __init__(
        self,
        root: Path,
        documents: DocumentStore | None,
        plane: Plane,
        actor: Actor,
        *,
        client: BinanceArchiveClient | None = None,
        okx_client: OkxFundingClient | None = None,
        okx_cross_check: bool = False,
        costs: CostLedger | None = None,
        clock: Clock | None = None,
        roots: StorageRoots | None = None,
    ) -> None:
        self.root = root
        self.roots = roots
        self.documents = documents
        self.plane = plane
        self.actor = actor
        self.clock = clock or SystemClock()
        self.client = client or BinanceArchiveClient(root / "raw" / "binance", clock=self.clock)
        self.okx_client = okx_client
        self.okx_cross_check = okx_cross_check
        self.costs = costs

    def _version_dir(self, name: str, content_hash: str) -> Path:
        return dataset_parquet(self.root, name, content_hash).parent

    def locate(self, dataset: DatasetVersion) -> Path | None:
        """Parquet for this version under this catalog. Never follows a path into another plane."""
        addressed = dataset_parquet(self.root, dataset.name, dataset.content_hash)
        if addressed.is_file():
            return addressed
        found = resolve_uri(dataset.uri, self.roots, catalog_root=self.root)
        if found is not None and within(found, self.root):
            return found
        return None

    def acquire_perp_funding_panel(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        frequency: str = "8h",
        name: str | None = None,
    ) -> DatasetVersion:
        """Download, normalize, check, and publish the panel as an immutable version.

        Re-acquiring a spec whose data is unchanged returns the existing version (same id).
        """
        symbols = sorted(set(symbols))
        if not symbols:
            raise ValueError("at least one symbol is required")
        instruments: list[Instrument] = []
        for symbol in symbols:
            instruments.extend((binance_perp(symbol), binance_spot(symbol)))
        start, end = _utc(start), _utc(end)
        if end <= start:
            raise ValueError("end must be after start")
        bar_length(frequency)
        name = name or default_dataset_name(symbols, start, end, frequency)

        requests_before = self.client.http_requests
        sources = fetch_perp_funding_sources(self.client, symbols, start, end, frequency)
        binance_requests = self.client.http_requests - requests_before
        retrieved_at = self.clock.now().astimezone(UTC)
        panel = assemble_perp_funding_panel(
            sources.perp, sources.spot, sources.funding, start, end, frequency
        )
        if panel.height == 0:
            raise ProviderUnavailableError(
                f"no perp bars published for {symbols} in [{start.isoformat()}, {end.isoformat()})"
            )

        okx_frames, okx_requests = self._okx_funding(symbols, start, end, retrieved_at)
        quality = check_panel(
            panel,
            frequency=frequency,
            retrieved_at=retrieved_at,
            funding_events=sources.funding,
            okx_funding=okx_frames,
        )
        content_hash = frame_content_hash(panel)
        self._record_costs(binance_requests, okx_requests, retrieved_at)

        existing = self._existing(name, content_hash)
        if existing is not None:
            return existing

        version_dir = self._version_dir(name, content_hash)
        data_path = version_dir / "data.parquet"
        manifest_path = version_dir / "dataset.json"
        if data_path.exists():
            if frame_content_hash(pl.read_parquet(data_path)) != content_hash:
                raise IntegrityError(f"{data_path} exists with different content; refusing")
        else:
            version_dir.mkdir(parents=True, exist_ok=True)
            tmp = version_dir / f".data.{os.getpid()}.parquet.tmp"
            panel.write_parquet(tmp, compression="zstd", statistics=True)
            os.replace(tmp, data_path)

        rows_per_symbol = {
            str(s): int(n) for s, n in panel.group_by("symbol").len().sort("symbol").iter_rows()
        }
        provenance: dict[str, Any] = {
            "source": "https://data.binance.vision",
            "spec": {
                "symbols": symbols,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "frequency": frequency,
                "range_rule": "ts >= start and available_at <= end",
            },
            "archives": [f.provenance() for f in sources.files],
            "missing_archives": sources.missing,
            "raw_row_counts": {
                f"{kind}:{symbol}": frames[symbol].height
                for kind, frames in (
                    ("perp_klines", sources.perp),
                    ("spot_klines", sources.spot),
                    ("funding", sources.funding),
                )
                for symbol in symbols
            },
            "row_counts": rows_per_symbol,
            "retrieved_at": retrieved_at.isoformat(),
            "http_requests": {"data.binance.vision": binance_requests, "okx": okx_requests},
            "okx_cross_check": _okx_cross_check_status(
                self.okx_cross_check, okx_frames is not None, OKX_HISTORY_DAYS
            ),
            "hash_scheme": HASH_SCHEME,
            "library_versions": _library_versions(),
            "user_agent": self.client.http.user_agent,
        }
        dataset = DatasetVersion(
            id=new_id("ds"),
            name=name,
            instruments=tuple(instruments),
            venue=PERP_VENUE,
            frequency=frequency,
            schema_version=PANEL_SCHEMA_VERSION,
            columns=PANEL_COLUMNS,
            time_start=panel["ts"].min(),
            time_end=panel["available_at"].max(),
            row_count=panel.height,
            content_hash=content_hash,
            uri=bind_uri(data_path, self.roots, catalog_root=self.root),
            pit_certified=is_pit_certified(quality),
            quality=quality,
            provenance=provenance,
            survivorship_notes=SURVIVORSHIP_NOTES,
            plane=self.plane.value,
            created_by=self.actor,
            created_at=retrieved_at,
        )
        tmp_manifest = version_dir / f".dataset.{os.getpid()}.json.tmp"
        tmp_manifest.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp_manifest, manifest_path)
        if self.documents is not None:
            self.documents.put(dataset, recorded_by=str(self.actor))
        return dataset

    def _existing(self, name: str, content_hash: str) -> DatasetVersion | None:
        if self.documents is not None:
            found = self.documents.list(
                DatasetVersion, where={"name": name, "content_hash": content_hash}, limit=1
            )
            if found and self.locate(found[0]) is not None:
                self.load(found[0])
                return found[0]
        manifest = self._version_dir(name, content_hash) / "dataset.json"
        if not manifest.exists():
            return None
        dataset = DatasetVersion.model_validate_json(manifest.read_text(encoding="utf-8"))
        self.load(dataset)
        if self.documents is not None and self.documents.get(DatasetVersion, dataset.id) is None:
            self.documents.put(dataset, recorded_by=str(self.actor))
        return dataset

    def _okx_funding(
        self, symbols: Sequence[str], start: datetime, end: datetime, now: datetime
    ) -> tuple[dict[str, pl.DataFrame] | None, int]:
        if not self.okx_cross_check:
            return None, 0
        window_start = max(start, now - timedelta(days=OKX_HISTORY_DAYS))
        if window_start >= end:
            return None, 0
        client = self.okx_client or OkxFundingClient()
        before = client.http_requests
        frames = {
            symbol: client.fetch_funding_history(
                okx_swap_id(symbol), window_start, end + timedelta(microseconds=1)
            )
            for symbol in symbols
        }
        return frames, client.http_requests - before

    def _record_costs(self, binance_requests: int, okx_requests: int, now: datetime) -> None:
        if self.costs is None:
            return
        for provider, count in (("data.binance.vision", binance_requests), ("okx", okx_requests)):
            if provider == "okx" and not self.okx_cross_check:
                continue
            self.costs.record(
                CostRecord(
                    id=new_id("cost"),
                    actor=self.actor,
                    task_id=None,
                    category="http",
                    provider=provider,
                    http_requests=count,
                    cached=count == 0,
                    created_at=now,
                )
            )

    def load(self, dataset: DatasetVersion) -> pl.DataFrame:
        """Read the version's Parquet and verify its content hash before returning it."""
        path = self.locate(dataset)
        if path is None:
            raise IntegrityError(f"dataset {dataset.id} data missing at {dataset.uri}")
        frame = pl.read_parquet(path)
        if tuple(frame.columns) != tuple(dataset.columns) or frame.height != dataset.row_count:
            raise IntegrityError(f"dataset {dataset.id} shape does not match its manifest")
        if dict(frame.schema) != PANEL_SCHEMA and tuple(dataset.columns) == PANEL_COLUMNS:
            raise IntegrityError(f"dataset {dataset.id} dtypes do not match the panel schema")
        actual = frame_content_hash(frame)
        if actual != dataset.content_hash:
            raise IntegrityError(
                f"dataset {dataset.id} content hash {actual} != recorded {dataset.content_hash}"
            )
        return frame

    def find(self, dataset_version_id: str) -> DatasetVersion:
        if self.documents is not None:
            found = self.documents.get(DatasetVersion, dataset_version_id)
            if found is not None:
                return found
        for manifest in sorted((self.root / "datasets").glob("*/*/dataset.json")):
            dataset = DatasetVersion.model_validate_json(manifest.read_text(encoding="utf-8"))
            if dataset.id == dataset_version_id:
                return dataset
        raise KeyError(f"DatasetVersion {dataset_version_id} not found in {self.root}")

    def get_bars(
        self,
        dataset_version_id: str,
        symbol: str | Instrument,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Verified panel rows for `symbol` with `start <= ts < end`, sorted by `ts`."""
        key = symbol.symbol if isinstance(symbol, Instrument) else symbol
        frame = self.load(self.find(dataset_version_id))
        return frame.filter(
            (pl.col("symbol") == key) & (pl.col("ts") >= _utc(start)) & (pl.col("ts") < _utc(end))
        ).sort("ts")


def _okx_cross_check_status(enabled: bool, ran: bool, history_days: int) -> str:
    if not enabled:
        return "disabled"
    if ran:
        return "ran"
    return (
        f"not_applicable: OKX publishes only the last {history_days} days of funding "
        "history and this dataset ends before that window"
    )
