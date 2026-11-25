from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from data_harness import RecordedServer, make_http

from enigma.core.clock import FrozenClock
from enigma.core.errors import IntegrityError, ProviderUnavailableError
from enigma.data.binance import BinanceArchiveClient
from enigma.data.catalog import DataCatalog
from enigma.data.instruments import binance_perp, binance_spot
from enigma.data.panel import PANEL_COLUMNS
from enigma.schemas.common import AgentRole, Plane, SystemActor
from enigma.schemas.research import DatasetVersion
from enigma.store import Stores

SYMBOLS = ["ETHUSDT", "BTCUSDT"]
START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 31, 16, tzinfo=UTC)


def _catalog(
    root: Path, server: RecordedServer, clock: FrozenClock, stores: Stores | None, plane: Plane
) -> DataCatalog:
    client = BinanceArchiveClient(root / "raw" / "binance", http=make_http(server), clock=clock)
    return DataCatalog(
        root,
        stores.documents if stores else None,
        plane,
        AgentRole.DATA_ENGINEERING,
        client=client,
        clock=clock,
        costs=stores.costs if stores else None,
    )


def test_instruments() -> None:
    perp = binance_perp("BTCUSDT")
    assert (perp.venue, perp.kind, perp.base, perp.quote) == (
        "binance_usdm",
        "perpetual",
        "BTC",
        "USDT",
    )
    assert binance_spot("ETHUSDT").key == "binance_spot:spot:ETHUSDT"
    with pytest.raises(ValueError):
        binance_perp("SOLUSDT")


def test_acquire_publishes_verified_version(
    tmp_path: Path, server: RecordedServer, clock: FrozenClock, stores: Stores
) -> None:
    catalog = _catalog(tmp_path, server, clock, stores, Plane.RESEARCH)
    ds = catalog.acquire_perp_funding_panel(SYMBOLS, START, END)

    assert ds.name == "binance_perp_funding_8h_btcusdt-ethusdt_20240101_20240131"
    assert ds.columns == PANEL_COLUMNS
    assert ds.row_count == 2 * 92
    assert ds.time_start == START and ds.time_end == END
    assert ds.venue == "binance_usdm" and ds.frequency == "8h" and ds.plane == "research"
    assert {i.key for i in ds.instruments} == {
        "binance_usdm:perpetual:BTCUSDT",
        "binance_spot:spot:BTCUSDT",
        "binance_usdm:perpetual:ETHUSDT",
        "binance_spot:spot:ETHUSDT",
    }
    assert ds.pit_certified and ds.quality.passed
    assert "survived" in ds.survivorship_notes

    path = Path(ds.uri)
    assert path == tmp_path / "datasets" / ds.name / ds.content_hash[7:23] / "data.parquet"
    assert (path.parent / "dataset.json").exists()

    archives = ds.provenance["archives"]
    assert len(archives) == 6
    assert all(len(a["sha256"]) == 64 and a["url"].startswith("https://") for a in archives)
    assert ds.provenance["http_requests"]["data.binance.vision"] == 12
    assert ds.provenance["row_counts"] == {"BTCUSDT": 92, "ETHUSDT": 92}
    assert "polars" in ds.provenance["library_versions"]

    assert stores.documents.require(DatasetVersion, ds.id) == ds
    assert stores.costs.totals()["http_requests"] == 12

    frame = catalog.load(ds)
    assert frame.height == ds.row_count
    bars = catalog.get_bars(
        ds.id, "BTCUSDT", datetime(2024, 1, 10, tzinfo=UTC), datetime(2024, 1, 11, tzinfo=UTC)
    )
    assert bars["ts"].to_list() == [datetime(2024, 1, 10, h, tzinfo=UTC) for h in (0, 8, 16)]
    assert (bars["symbol"] == "BTCUSDT").all()
    by_instrument = catalog.get_bars(ds.id, binance_perp("ETHUSDT"), START, END)
    assert by_instrument.height == 92


def test_acquire_is_idempotent(
    tmp_path: Path, server: RecordedServer, clock: FrozenClock, stores: Stores
) -> None:
    catalog = _catalog(tmp_path, server, clock, stores, Plane.RESEARCH)
    first = catalog.acquire_perp_funding_panel(SYMBOLS, START, END)
    requests = catalog.client.http_requests
    again = catalog.acquire_perp_funding_panel(list(reversed(SYMBOLS)), START, END)
    assert again == first
    assert catalog.client.http_requests == requests, "verified archives are never re-downloaded"
    assert stores.documents.count(DatasetVersion) == 1
    assert len(stores.documents.history(DatasetVersion, first.id)) == 1

    # A fresh process with no document store finds the manifest on disk.
    fresh = _catalog(tmp_path, server, clock, None, Plane.RESEARCH)
    assert fresh.acquire_perp_funding_panel(SYMBOLS, START, END).id == first.id
    assert fresh.get_bars(first.id, "BTCUSDT", START, END).height == 92


def test_same_data_same_hash_in_independent_roots(
    tmp_path: Path, server: RecordedServer, clock: FrozenClock
) -> None:
    research = _catalog(tmp_path / "research", server, clock, None, Plane.RESEARCH)
    validation = _catalog(tmp_path / "validation", server, clock, None, Plane.VALIDATION)
    a = research.acquire_perp_funding_panel(SYMBOLS, START, END)
    b = validation.acquire_perp_funding_panel(SYMBOLS, START, END)
    assert a.content_hash == b.content_hash
    assert a.id != b.id
    assert b.plane == "validation"
    assert Path(b.uri).is_relative_to(tmp_path / "validation")
    assert not (tmp_path / "validation" / "raw").samefile(tmp_path / "research" / "raw")


def test_research_end_cap_excludes_later_rows(
    tmp_path: Path, server: RecordedServer, clock: FrozenClock
) -> None:
    catalog = _catalog(tmp_path, server, clock, None, Plane.RESEARCH)
    cap = datetime(2024, 1, 20, tzinfo=UTC)
    ds = catalog.acquire_perp_funding_panel(SYMBOLS, START, cap)
    assert ds.time_end <= cap
    assert catalog.load(ds)["available_at"].max() <= cap  # type: ignore[operator]


def test_tampered_parquet_fails_integrity(
    tmp_path: Path, server: RecordedServer, clock: FrozenClock
) -> None:
    catalog = _catalog(tmp_path, server, clock, None, Plane.RESEARCH)
    ds = catalog.acquire_perp_funding_panel(SYMBOLS, START, END)
    frame = pl.read_parquet(ds.uri)
    tampered = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("funding_rate") * 10)
        .otherwise(pl.col("funding_rate"))
        .alias("funding_rate")
    )
    tampered.write_parquet(ds.uri)
    with pytest.raises(IntegrityError, match="content hash"):
        catalog.load(ds)
    with pytest.raises(IntegrityError):
        catalog.get_bars(ds.id, "BTCUSDT", START, END)
    # Re-acquiring must not silently overwrite the immutable version.
    with pytest.raises(IntegrityError):
        catalog.acquire_perp_funding_panel(SYMBOLS, START, END)


def test_truncated_parquet_fails_integrity(
    tmp_path: Path, server: RecordedServer, clock: FrozenClock
) -> None:
    catalog = _catalog(tmp_path, server, clock, None, Plane.RESEARCH)
    ds = catalog.acquire_perp_funding_panel(SYMBOLS, START, END)
    pl.read_parquet(ds.uri).head(5).write_parquet(ds.uri)
    with pytest.raises(IntegrityError, match="shape"):
        catalog.load(ds)


def test_unknown_dataset_and_empty_range(
    tmp_path: Path, server: RecordedServer, clock: FrozenClock
) -> None:
    catalog = DataCatalog(
        tmp_path,
        None,
        Plane.RESEARCH,
        SystemActor.SYSTEM,
        client=BinanceArchiveClient(tmp_path / "raw", http=make_http(server), clock=clock),
    )
    with pytest.raises(KeyError):
        catalog.find("ds_missing")
    with pytest.raises(ProviderUnavailableError):
        catalog.acquire_perp_funding_panel(
            ["BTCUSDT"], datetime(2018, 1, 1, tzinfo=UTC), datetime(2018, 2, 1, tzinfo=UTC)
        )
    with pytest.raises(ValueError):
        catalog.acquire_perp_funding_panel(["BTCUSDT"], END, START)
