"""A rejected LLM generation is kept. The cycle tries one library mechanism, then defers."""

from __future__ import annotations

from pathlib import Path

import pytest
from cycle_rig import Rig
from test_daemon import _daemon

from enigma.fsm import RESEARCH_CYCLE
from enigma.schemas.common import AgentRole
from enigma.schemas.cycle import NextExperimentProposal, ProposalState
from enigma.schemas.governance import ExecutionMode
from enigma.schemas.knowledge import Lesson
from enigma.schemas.quant import GenerationResult
from enigma.schemas.research import Hypothesis


def _rejected(ctx: object) -> GenerationResult:
    task = ctx.task  # type: ignore[attr-defined]
    now = ctx.clock.now()  # type: ignore[attr-defined]
    return GenerationResult(
        id="hgen_" + task.id.replace("task_", "")[:26],
        context_id="rctx_rejected",
        context_hash="sha256:" + ("a" * 64),
        execution_mode=ExecutionMode.LLM,
        method="llm",
        prompt_refs=("quant.hypothesis@4",),
        provenance=(),
        proposals=(),
        rejected=(),
        repair_rounds=0,
        failure_reason="no_proposal_survived_checks",
        created_by=AgentRole.QUANT_RESEARCH,
        created_at=now,
    )


def test_rejected_llm_draft_recovers_onto_a_library_mechanism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENIGMA_HYPOTHESIS_MODE", "llm")
    rig = Rig(tmp_path / "c.sqlite3")
    agent = rig.worker.agents[AgentRole.QUANT_RESEARCH]
    original = agent.execute
    llm_calls = {"n": 0}

    def execute(ctx: object) -> object:
        if ctx.task.inputs.get("recovery") == "alternate":  # type: ignore[attr-defined]
            return original(ctx)  # type: ignore[arg-type]
        llm_calls["n"] += 1
        generation = _rejected(ctx)
        ctx.view.put(generation)  # type: ignore[attr-defined]
        return agent._checks_rejected(generation)

    agent.execute = execute  # type: ignore[method-assign]
    cycle = rig.submit()
    assert rig.run(cycle) == "COMPLETED"
    assert llm_calls["n"] == 1
    _, state, status = rig.stores.checkpoints.load(cycle)  # type: ignore[misc]
    assert status == "completed"
    assert state["hypothesis_recovery"] is True
    hypothesis = rig.stores.documents.require(Hypothesis, state["hypothesis_id"])
    assert hypothesis.generation_method.startswith("template:")
    generation_id = state["rejected_generation_ids"][0]
    stored = rig.stores.documents.require(GenerationResult, generation_id)
    assert stored.failure_reason == "no_proposal_survived_checks"
    assert stored.proposals == ()


def test_second_rejection_defers_with_a_chainable_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENIGMA_HYPOTHESIS_MODE", "llm")
    rig = Rig(tmp_path / "c.sqlite3")
    agent = rig.worker.agents[AgentRole.QUANT_RESEARCH]

    def execute(ctx: object) -> object:
        generation = _rejected(ctx)
        ctx.view.put(generation)  # type: ignore[attr-defined]
        return agent._checks_rejected(generation)

    agent.execute = execute  # type: ignore[method-assign]
    cycle = rig.submit()
    assert rig.run(cycle) == "DEFERRED"
    assert rig.fsm.state(RESEARCH_CYCLE, cycle) == "DEFERRED"
    _, state, status = rig.stores.checkpoints.load(cycle)  # type: ignore[misc]
    assert status == "deferred"
    assert state["defer_reason"] == "no_proposal_survived_checks"
    lesson = rig.stores.documents.require(Lesson, state["lesson_ids"][0])
    assert "no_proposal_survived_checks" in lesson.topics
    proposal = rig.stores.documents.require(NextExperimentProposal, state["next_proposal_id"])
    assert proposal.parent_cycle_id == cycle
    assert proposal.state is ProposalState.PROPOSED
    stored = rig.stores.documents.list(GenerationResult, limit=10)
    assert stored and all(item.failure_reason == "no_proposal_survived_checks" for item in stored)

    daemon = _daemon(rig, max_cycles_per_day=2, tech_scan_interval_hours=24.0, tech_scan_topics=())
    daemon.run(max_iterations=30)
    child = [
        row for row in rig.stores.checkpoints.list("research_cycle") if row["workflow_id"] != cycle
    ]
    assert len(child) == 1
    scheduled = rig.stores.documents.require(NextExperimentProposal, proposal.id)
    assert scheduled.state is ProposalState.SCHEDULED
    assert scheduled.child_cycle_id == child[0]["workflow_id"]
