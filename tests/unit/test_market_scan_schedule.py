from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from cycle_rig import Rig

from enigma.core.config import Settings
from enigma.orchestrator.daemon import Daemon, Schedule
from enigma.orchestrator.institution import Institution
from enigma.schemas.market import MarketObservation, MarketScanReport
from enigma.schemas.research import ResearchObjective


def test_scan_opens_one_objective_per_window(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    now = datetime.now(UTC).replace(microsecond=0)
    start = now - timedelta(hours=30)
    for index in range(19):
        rig.stores.documents.put(
            MarketObservation(
                id=f"mob_{index:02d}",
                symbol="BTCUSDT",
                field="funding_rate",
                available_at=start + timedelta(hours=index),
                value=0.0,
            ),
            recorded_by="test",
        )
    rig.stores.documents.put(
        MarketObservation(
            id="mob_spike",
            symbol="BTCUSDT",
            field="funding_rate",
            available_at=start + timedelta(hours=19),
            value=10.0,
        ),
        recorded_by="test",
    )
    daemon = Daemon(
        _inst(rig),
        Schedule(
            poll_interval_seconds=0.0,
            heartbeat_seconds=0.0,
            auto_next_experiment=False,
            max_cycles_per_day=2,
            max_concurrent_cycles=1,
            tech_scan_interval_hours=0.0,
            tech_scan_topics=(),
            market_scan_interval_hours=100000.0,
        ),
    )
    daemon.run(max_iterations=1)
    daemon.run(max_iterations=1)
    reports = rig.stores.documents.list(MarketScanReport)
    assert len(reports) == 1
    assert reports[0].anomaly_count == 1
    assert reports[0].suggested_mechanism == "funding_crowding_reversal"
    assert reports[0].objective_id
    objective = rig.stores.documents.get(ResearchObjective, reports[0].objective_id or "")
    assert objective is not None
    assert "not a validation result" in objective.statement
    cycles = rig.stores.checkpoints.list("research_cycle")
    assert cycles == []


def _inst(rig: Rig) -> Institution:
    return Institution(
        Settings.from_env(),
        rig.stores,
        rig.bus,
        rig.fsm,
        rig.contracts,
        rig.driver.policy,
        rig.driver,
        rig.worker,
        rig.fakes.services(),
        rig.driver.clock,
    )
