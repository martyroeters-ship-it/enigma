"""Research-cycle orchestration tests (real driver/agents/stores; service test doubles)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from cycle_rig import Rig

from enigma.fsm import RESEARCH_CYCLE, RESEARCH_CYCLE_STEPS, STRATEGY_LIFECYCLE
from enigma.schemas.common import AgentRole, Instrument, SystemActor
from enigma.schemas.cycle import NextExperimentProposal, ProposalState
from enigma.schemas.governance import ImprovementProposal, Reflection, TaskStatus
from enigma.schemas.knowledge import Lesson
from enigma.schemas.research import Experiment
from enigma.store.ledger import KILL_SWITCH


def test_full_cycle_reaches_all_steps(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    cycle = rig.submit()
    assert rig.run(cycle) == "COMPLETED", rig.stores.checkpoints.load(cycle)
    reached = [e.to_state for e in rig.fsm.history(RESEARCH_CYCLE, cycle)]
    assert reached == [*RESEARCH_CYCLE_STEPS, "COMPLETED"]
    _, state, status = rig.stores.checkpoints.load(cycle)  # type: ignore[misc]
    assert status == "completed"
    assert state["validation_overall"] == "FAIL"
    tasks = rig.stores.queue.list(correlation_id=cycle)
    assert all(t.status is TaskStatus.SUCCEEDED for t in tasks)
    assert {t.kind for t in tasks} == {
        "intelligence.research",
        "data.acquire",
        "microstructure.analyze",
        "quant.hypothesize",
        "strategy.implement",
        "validation.validate",
        "meta.reflect_cycle",
        "meta.propose_next",
    }
    exps = [rig.stores.documents.require(Experiment, e) for e in state["experiment_ids"]]
    assert len(exps) == 9 and all(e.result_hash for e in exps)
    # validation was decided only by the validation service and is terminal
    pkg = state["candidate_package_id"]
    hist = rig.fsm.history(STRATEGY_LIFECYCLE, pkg)
    assert hist[-1].to_state == "VALIDATION_FAILED"
    assert hist[-1].actor == SystemActor.VALIDATION_SERVICE.value
    lessons = [rig.stores.documents.require(Lesson, x) for x in state["lesson_ids"]]
    assert any("cost" in " ".join(les.topics) for les in lessons)
    proposal = rig.stores.documents.require(NextExperimentProposal, state["next_proposal_id"])
    assert proposal.lesson_ids
    assert len(rig.stores.documents.list(Reflection, limit=100)) >= len(tasks)
    assert rig.stores.documents.count(ImprovementProposal) >= 1
    assert len(rig.bus.conversation(cycle)) >= 2 * len(tasks)
    assert rig.stores.audit.verify() > 0


def test_passing_candidate_gets_risk_veto_while_paper_disabled(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3", fail_gate=None)
    cycle = rig.submit()
    assert rig.run(cycle) == "COMPLETED"
    risk = [t for t in rig.stores.queue.list(correlation_id=cycle) if t.kind == "risk.review"]
    assert len(risk) == 1 and risk[0].status is TaskStatus.SUCCEEDED
    assert risk[0].outputs["decision"] == "veto"
    assert "paper_trading_disabled" in risk[0].outputs["reason_codes"]
    _, state, _ = rig.stores.checkpoints.load(cycle)  # type: ignore[misc]
    assert state["risk_decision"] == "veto"
    assert state["risk_review_id"] == risk[0].outputs["review_id"]
    assert "paper" not in state.get("tasks", {})
    assert "execution.paper" not in {t.kind for t in rig.stores.queue.list(correlation_id=cycle)}


def test_paper_ack_env_does_not_bypass_risk_veto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENIGMA_PAPER_ACK_ID", "hrack_should_not_open")
    monkeypatch.setenv("ENIGMA_PAPER_NOTIONAL_USD", "10000")
    rig = Rig(tmp_path / "c.sqlite3", fail_gate=None)
    cycle = rig.submit()
    assert rig.run(cycle) == "COMPLETED"
    _, state, _ = rig.stores.checkpoints.load(cycle)  # type: ignore[misc]
    assert state["risk_decision"] == "veto"
    assert "paper" not in state.get("tasks", {})
    assert state.get("paper_session_id") is None
    kinds = {t.kind for t in rig.stores.queue.list(correlation_id=cycle)}
    assert "risk.review" in kinds
    assert "execution.paper" not in kinds


def test_resume_after_process_restart_does_not_repeat_work(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite3"
    first = Rig(db)
    cycle = first.submit()
    assert first.run(cycle, until="STRATEGY_EXECUTED") == "STRATEGY_EXECUTED"
    calls_before = list(first.fakes.calls)
    first.stores.db.close()

    second = Rig(db, fakes=first.fakes)  # new process: fresh stores, driver, worker
    assert second.run(cycle) == "COMPLETED"
    kinds = [t.kind for t in second.stores.queue.list(correlation_id=cycle)]
    assert len(kinds) == len(set(kinds)), "a step was delegated twice"
    new_calls = second.fakes.calls[len(calls_before) :]
    assert "intel" not in new_calls and "data" not in new_calls and "backtest" not in new_calls


def test_orphaned_running_task_is_requeued_and_completed(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    cycle = rig.submit()
    rig.driver.tick(cycle)
    claimed = rig.stores.queue.claim(
        "dead-worker", roles=(AgentRole.INTELLIGENCE,), lease=timedelta(hours=1)
    )
    assert claimed is not None and claimed.status is TaskStatus.RUNNING
    assert rig.stores.queue.requeue_orphans(lambda worker: worker != "dead-worker") == [claimed.id]
    assert rig.run(cycle) == "COMPLETED"


def test_kill_switch_halts_progress(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    cycle = rig.submit()
    rig.stores.flags.set(KILL_SWITCH, "on", "human")
    assert rig.run(cycle) == "OBJECTIVE_RECEIVED"
    assert rig.worker.run_once() is None


@pytest.mark.parametrize("missing", ["source_ids", "evidence_ids"])
def test_cycle_aborts_without_external_research(tmp_path: Path, missing: str) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    original = rig.fakes.intel_research

    def empty(**kwargs: Any) -> dict[str, Any]:
        out = original(**kwargs)
        out[missing] = []
        return out

    rig.fakes.intel_research = empty  # type: ignore[method-assign]
    rig2 = Rig(tmp_path / "c.sqlite3", fakes=rig.fakes)
    cycle = rig2.submit()
    assert rig2.run(cycle) == "ABORTED"
    proposals = rig2.stores.documents.list(NextExperimentProposal, where={"parent_cycle_id": cycle})
    assert len(proposals) == 1
    assert proposals[0].state is ProposalState.PROPOSED


def test_duplicate_objective_does_not_open_a_second_writer(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    first = rig.submit()
    assert rig.submit() == first
    instruments = (
        Instrument(
            symbol="BTCUSDT", base="BTC", quote="USDT", kind="perpetual", venue="binance_usdm"
        ),
    )
    with pytest.raises(ValueError, match="already running"):
        rig.driver.submit_objective("Do option-implied volatilities mean-revert?", instruments)


def test_identical_objective_reuses_prior_literature(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    first = rig.submit()
    assert rig.run(first) == "COMPLETED"
    second = rig.submit()
    assert second != first
    rig.driver.tick(second)
    state = rig.stores.checkpoints.load(second)[1]
    intel = rig.stores.queue.get(state["tasks"]["intelligence"])
    assert intel is not None
    assert intel.status is TaskStatus.SUCCEEDED
    assert intel.outputs.get("reused_prior_search") is True
    assert intel.outputs.get("http_requests") == 0
    assert intel.outputs["source_ids"]
