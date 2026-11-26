"""An institutional home copied to another directory keeps its research."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from cycle_rig import RESEARCH_CUTOFF, Rig
from test_daemon import _daemon

from enigma.agents.contracts import ContractRegistry
from enigma.comm import MessageBus
from enigma.core.clock import FrozenClock
from enigma.core.config import Settings
from enigma.core.errors import IntegrityError
from enigma.core.paths import StorageRoots, dataset_parquet
from enigma.data.catalog import DataCatalog
from enigma.data.hashing import frame_content_hash
from enigma.data.panel import PANEL_COLUMNS, PANEL_SCHEMA, PANEL_SCHEMA_VERSION
from enigma.fsm import RESEARCH_CYCLE, StateMachineStore
from enigma.orchestrator.backlog import research_agenda
from enigma.orchestrator.cycle import CycleDriver, HoldoutPolicy
from enigma.orchestrator.quarantine import load_quarantine, prepare_runtime
from enigma.orchestrator.services import OperationalServices
from enigma.schemas.common import AgentRole, Instrument, Plane
from enigma.schemas.knowledge import Source, SourceKind
from enigma.schemas.research import (
    DataQualityCheck,
    DataQualityReport,
    DatasetVersion,
    Experiment,
    ExperimentStatus,
    ResearchObjective,
)
from enigma.schemas.validation import HoldoutManifest
from enigma.store import Stores
from enigma.store.relocate import migrate_home
from enigma.validation.holdout import HoldoutCustody
from enigma.validation.holdout import frame_content_hash as holdout_hash

OLD = Path("/tmp/m4-integration")  # noqa: S108 - path recorded on the machine that exported the home
NOW = datetime(2026, 11, 26, tzinfo=UTC)
NAME = "binance_perp_funding_8h_btcusdt_20200101_20261124"


def _settings(home: Path) -> Settings:
    root = Path(__file__).resolve().parents[2]
    return Settings(
        home=home,
        research_dir=home / "research",
        validation_dir=home / "validation",
        execution_dir=home / "execution",
        db_path=home / "control" / "enigma.sqlite3",
        registries_dir=root / "registries",
    )


def _panel() -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for index in range(4):
        opened = start + timedelta(hours=8 * index)
        rows.append(
            {
                "symbol": "BTCUSDT",
                "ts": opened,
                "available_at": opened + timedelta(hours=8),
                "perp_open": 100.0,
                "perp_high": 101.0,
                "perp_low": 99.0,
                "perp_close": 100.0 + index,
                "perp_volume": 5.0,
                "perp_quote_volume": 500.0,
                "spot_close": 100.0,
                "funding_rate": 0.0001 * (index + 1),
                "funding_time": opened + timedelta(hours=8),
                "basis": 0.001,
            }
        )
    return pl.DataFrame(rows, schema=PANEL_SCHEMA)


def _dataset(frame: pl.DataFrame, plane: str, uri: str, dataset_id: str) -> DatasetVersion:
    return DatasetVersion(
        id=dataset_id,
        name=NAME,
        instruments=(
            Instrument(
                symbol="BTCUSDT",
                base="BTC",
                quote="USDT",
                kind="perpetual",
                venue="binance_usdm",
            ),
        ),
        venue="binance_usdm",
        frequency="8h",
        schema_version=PANEL_SCHEMA_VERSION,
        columns=PANEL_COLUMNS,
        time_start=frame["ts"].min(),
        time_end=frame["available_at"].max(),
        row_count=frame.height,
        content_hash=frame_content_hash(frame),
        uri=uri,
        pit_certified=True,
        quality=DataQualityReport(checks=(DataQualityCheck(name="ok", passed=True),)),
        provenance={"source": "copied-home"},
        plane=plane,
        created_by=AgentRole.DATA_ENGINEERING,
        created_at=NOW,
    )


def _publish(catalog: Path, dataset: DatasetVersion, frame: pl.DataFrame) -> None:
    path = dataset_parquet(catalog, dataset.name, dataset.content_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    (path.parent / "dataset.json").write_text(dataset.model_dump_json(indent=2), encoding="utf-8")


def test_relocated_home_loads_and_opens_the_next_objective(tmp_path: Path) -> None:
    home = tmp_path / "linux-home"
    settings = _settings(home)
    settings.ensure_dirs()
    frame = _panel()
    digest = frame_content_hash(frame)
    research = _dataset(
        frame,
        "research",
        dataset_parquet(OLD / "research" / "data", NAME, digest).as_posix(),
        "ds_research_moved",
    )
    validation = _dataset(
        frame,
        "validation",
        dataset_parquet(OLD / "validation" / "data", NAME, digest).as_posix(),
        "ds_validation_moved",
    )
    _publish(settings.research_dir / "data", research, frame)
    _publish(settings.validation_dir / "data", validation, frame)

    digest = hashlib.sha256(b'{"title":"moved"}').hexdigest()
    raw_path = settings.research_dir / "raw" / "intelligence" / digest[:2] / f"{digest}.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(b'{"title":"moved"}')
    artifact_hash = "sha256:" + "ab" * 32
    artifact = (
        settings.research_dir / "backtests" / f"{artifact_hash.removeprefix('sha256:')[:32]}.npz"
    )
    artifact.parent.mkdir(parents=True)
    np.savez_compressed(artifact, equity=np.array([1.0, 1.01]))

    stores = Stores.open(settings.db_path)
    stores.documents.put(research, recorded_by="test")
    stores.documents.put(
        Source(
            id="src_moved",
            kind=SourceKind.WEB_PAGE,
            provider="test",
            url="https://example.invalid/moved",
            title="Moved",
            content_hash="sha256:" + digest,
            raw_uri=f"{OLD.as_posix()}/research/raw/intelligence/{digest[:2]}/{digest}.json",
            retrieved_by=AgentRole.INTELLIGENCE,
            retrieved_at=NOW,
        ),
        recorded_by="test",
    )
    stores.documents.put(
        Experiment(
            id="exp_moved",
            hypothesis_id="hyp_moved",
            family_id="fam_funding_carry",
            trial_index=1,
            strategy_package_id="pkg_moved",
            dataset_version_id=research.id,
            research_window=(NOW - timedelta(days=10), NOW),
            code_sha="abc",
            config_hash="sha256:" + "cd" * 32,
            seed=1,
            status=ExperimentStatus.SUCCEEDED,
            result_hash=artifact_hash,
            artifact_uri=f"{OLD.as_posix()}/research/backtests/{artifact.name}",
            created_by=AgentRole.QUANT_RESEARCH,
        ),
        recorded_by="test",
    )
    holdout_rows = frame.filter(pl.col("ts") == frame["ts"].max())
    custody = HoldoutCustody(settings.validation_dir)
    custody.documents.put(
        HoldoutManifest(
            id="hmf_moved",
            symbols=("BTCUSDT",),
            frequency="8h",
            protocol_version="0.1.0-slice",
            holdout_fraction=0.3,
            data_start=frame["ts"].min(),
            data_end=frame["ts"].max(),
            holdout_start=frame["ts"].max(),
            n_bars_total=4,
            n_holdout_bars=1,
            holdout_content_hash=holdout_hash(holdout_rows),
            dataset_ids=(validation.id,),
            created_at=NOW,
        ),
        recorded_by="validation_service",
    )
    custody.close()

    research_catalog = DataCatalog(
        settings.research_dir / "data",
        stores.documents,
        Plane.RESEARCH,
        AgentRole.DATA_ENGINEERING,
        roots=isolated_roots(settings),
    )
    assert research_catalog.load(research).height == 4

    preflight = prepare_runtime(settings, stores)
    assert preflight.legacy >= 1
    assert not preflight.unresolved
    assert not preflight.refused

    migrated = migrate_home(settings, stores)
    assert migrated.rewritten >= 4
    assert not migrated.refused
    stored_dataset = stores.documents.get(DatasetVersion, research.id)
    stored_source = stores.documents.get(Source, "src_moved")
    stored_experiment = stores.documents.get(Experiment, "exp_moved")
    assert stored_dataset is not None and stored_dataset.uri.startswith("inst:research/")
    assert stored_source is not None and stored_source.raw_uri is not None
    assert stored_source.raw_uri.startswith("inst:research/")
    assert stored_experiment is not None and stored_experiment.artifact_uri is not None
    assert stored_experiment.artifact_uri.startswith("inst:research/")
    history = stores.documents.history(DatasetVersion, research.id)
    assert len(history) == 2
    assert history[0].uri.startswith(f"{OLD.as_posix()}/")
    assert history[0].content_hash == history[1].content_hash
    manifest = json_manifest(settings.validation_dir / "data", validation)
    assert manifest["uri"].startswith("inst:validation/")
    assert manifest["content_hash"] == validation.content_hash
    custody = HoldoutCustody(settings.validation_dir)
    try:
        sealed = custody.documents.require(HoldoutManifest, "hmf_moved")
        assert sealed.holdout_content_hash == holdout_hash(holdout_rows)
        assert len(custody.documents.history(HoldoutManifest, "hmf_moved")) == 1
    finally:
        custody.close()
    stores.audit.verify()
    stores.db.close()

    moved = tmp_path / "enigma-run"
    shutil.move(home, moved)
    moved_settings = _settings(moved)
    moved_stores = Stores.open(moved_settings.db_path)
    again = migrate_home(moved_settings, moved_stores)
    assert again.rewritten == 0
    assert len(moved_stores.documents.history(DatasetVersion, research.id)) == 2
    moved_stores.audit.verify()

    clock = FrozenClock(NOW)
    services = OperationalServices(
        moved_settings,
        moved_stores,
        HoldoutPolicy.load(moved_settings.registries_dir / "validation" / "holdout_policy.yaml"),
        clock,
    )
    loaded = services.data_load_research(research.id)
    assert loaded.height == 4
    assert frame_content_hash(loaded) == research.content_hash
    validation_catalog = services.holdout_source.catalog
    assert validation_catalog.load(validation_catalog.find(validation.id)).height == 4

    def cutoff(_symbols: list[str]) -> datetime:
        services.data_load_research(research.id)
        return RESEARCH_CUTOFF

    driver = CycleDriver(
        moved_stores,
        MessageBus(moved_stores.db, moved_stores.audit),
        StateMachineStore(moved_stores.db, moved_stores.audit),
        ContractRegistry.load(moved_settings.registries_dir / "agents"),
        services.policy,
        cutoff,
        clock,
    )
    cycle_id = driver.submit_objective(
        "Does funding carry show a gross edge on BTCUSDT perpetuals at 8h after the home moved?",
        research.instruments,
        ("funding_carry", "BTCUSDT", "8h"),
    )
    assert driver.fsm.state(RESEARCH_CYCLE, cycle_id) == "OBJECTIVE_RECEIVED"
    moved_stores.db.close()


def test_hash_mismatch_refuses_migration(tmp_path: Path) -> None:
    home = tmp_path / "home"
    settings = _settings(home)
    settings.ensure_dirs()
    frame = _panel()
    digest = frame_content_hash(frame)
    dataset = _dataset(
        frame,
        "research",
        dataset_parquet(OLD / "research" / "data", NAME, digest).as_posix(),
        "ds_tampered",
    )
    _publish(settings.research_dir / "data", dataset, frame)
    path = dataset_parquet(settings.research_dir / "data", dataset.name, dataset.content_hash)
    broken = frame.with_columns(pl.col("perp_close") + 1)
    broken.write_parquet(path)
    stores = Stores.open(settings.db_path)
    stores.documents.put(dataset, recorded_by="test")
    report = migrate_home(settings, stores)
    assert report.refused
    assert len(stores.documents.history(DatasetVersion, dataset.id)) == 1
    kept = stores.documents.get(DatasetVersion, dataset.id)
    assert kept is not None and kept.uri.startswith(f"{OLD.as_posix()}/")
    with pytest.raises(IntegrityError):
        prepare_runtime(settings, stores)
    stores.audit.verify()
    stores.db.close()


def test_missing_dataset_quarantines_one_symbol_and_opens_another(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    daemon = _daemon(rig, max_cycles_per_day=0, continuous=True, review_interval_hours=0)

    def cutoff(symbols: list[str]) -> datetime:
        if "BTCUSDT" in symbols:
            raise IntegrityError(
                "dataset ds_gone data missing at "
                "/tmp/m4-integration/validation/data/datasets/name/hash/data.parquet"
            )
        return RESEARCH_CUTOFF

    daemon.inst.driver.research_cutoff = cutoff
    daemon.run(max_iterations=1)
    cycles = rig.stores.checkpoints.list("research_cycle")
    assert len(cycles) == 1
    objective = rig.stores.documents.get(ResearchObjective, cycles[0]["state"]["objective_id"])
    assert objective is not None
    assert "ETHUSDT" in objective.statement
    quarantine = load_quarantine(rig.stores)
    assert "BTCUSDT" in quarantine.symbols
    agenda = research_agenda(rig.stores, inputs_budget=10)
    blocked = [item for item in agenda.ranked if "BTCUSDT" in item.candidate.id and item.viable]
    assert blocked == []
    assert any(item.viable and "ETHUSDT" in item.candidate.id for item in agenda.ranked)


def test_every_missing_dataset_stays_inside_the_scheduler(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    daemon = _daemon(rig, max_cycles_per_day=0, continuous=True, review_interval_hours=0)

    def cutoff(_symbols: list[str]) -> datetime:
        raise IntegrityError("dataset ds_gone data missing at /tmp/old/data.parquet")

    daemon.inst.driver.research_cutoff = cutoff
    daemon.run(max_iterations=1)
    assert rig.stores.checkpoints.list("research_cycle") == []
    assert load_quarantine(rig.stores).candidates


def isolated_roots(settings: Settings) -> StorageRoots:
    return StorageRoots.from_settings(settings)


def json_manifest(catalog: Path, dataset: DatasetVersion) -> dict[str, object]:
    path = dataset_parquet(catalog, dataset.name, dataset.content_hash).parent / "dataset.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(body, dict)
    return body
