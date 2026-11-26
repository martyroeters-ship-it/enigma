"""Continuous mode opens the next backlog item without a typed objective."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from cycle_rig import Rig
from test_daemon import _daemon

from enigma.orchestrator.backlog import BacklogPolicy, direction_history, research_agenda
from enigma.orchestrator.host import memavailable_mb
from enigma.schemas.common import AgentRole, Instrument
from enigma.schemas.research import Hypothesis, ResearchObjective


def test_memavailable_parser() -> None:
    text = "MemTotal: 1000000 kB\nMemAvailable: 2048000 kB\n"
    assert memavailable_mb(text) == 2000.0
    assert memavailable_mb("MemTotal: 1 kB\n") is None


def test_continuous_mode_opens_a_backlog_cycle(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    daemon = _daemon(rig, max_cycles_per_day=0, continuous=True, review_interval_hours=0)
    daemon.run(max_iterations=1)
    cycles = rig.stores.checkpoints.list("research_cycle")
    assert len(cycles) == 1
    objective_id = cycles[0]["state"]["objective_id"]
    objective = rig.stores.documents.get(ResearchObjective, objective_id)
    assert objective is not None
    assert "gross edge" in objective.statement
    assert "continuous-backlog" in objective.topics


def test_a_completed_direction_is_not_repeated(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    now = datetime(2026, 3, 1, tzinfo=UTC)
    btc = Instrument(symbol="BTCUSDT", base="BTC", quote="USDT", kind="perpetual", venue="binance")
    hyp = Hypothesis(
        id="hyp_done",
        objective_id="obj_done",
        family_id="fam_basis_reversion",
        title="Basis",
        mechanism="basis",
        prediction="basis reverts",
        formal_statement="z(basis) < -k",
        falsification_criteria=("gross sharpe <= 0",),
        instruments=(btc,),
        horizon="1d",
        features=("basis",),
        trial_budget=1,
        cost_model_version="cm-test",
        generation_method="template:test",
        methodology_version="0.1.0",
        created_by=AgentRole.QUANT_RESEARCH,
        created_at=now,
    )
    rig.stores.documents.put(hyp, recorded_by="test")
    rig.stores.checkpoints.save(
        "cyc_done",
        "research_cycle",
        "COMPLETED",
        {"hypothesis_id": hyp.id, "validation_overall": "FAIL"},
        "completed",
    )
    hist = direction_history(rig.stores)
    assert hist["basis_reversion|BTCUSDT|1d"].fails == 1
    agenda = research_agenda(
        rig.stores, inputs_budget=10, policy=BacklogPolicy(diminishing_returns_min_cycles=3)
    )
    blocked = [
        item for item in agenda.ranked if item.candidate.id == "opp_basis_reversion_BTCUSDT_1d"
    ]
    assert blocked and blocked[0].viable is False
    viable = [item.candidate.id for item in agenda.ranked if item.viable]
    assert "opp_basis_reversion_ETHUSDT_1d" in viable
