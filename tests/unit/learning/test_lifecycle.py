"""A non-protected improvement is adopted only after a later, independent measurement."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from enigma.fsm import IMPROVEMENT_PROPOSAL, StateMachineStore
from enigma.learning.lifecycle import advance_improvements
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import (
    ImprovementProposal,
    ProposalStatus,
    ProposalTarget,
    Reflection,
    ReflectionOutcome,
)
from enigma.schemas.knowledge import Lesson
from enigma.store import Stores

NOW = datetime(2026, 11, 1, tzinfo=UTC)


def _proposal(proposal_id: str, target: ProposalTarget) -> ImprovementProposal:
    return ImprovementProposal(
        id=proposal_id,
        target=target,
        target_ref=AgentRole.QUANT_RESEARCH.value,
        change_summary=f"change {proposal_id}",
        rationale="measured on later work",
        success_metric="task_failure_rate",
        baseline_metric_value=0.8,
        proposed_by=AgentRole.META_LEARNING,
        status=ProposalStatus.PROPOSED,
        created_at=NOW,
    )


def _reflection(index: int) -> Reflection:
    return Reflection(
        id=f"ref_{index}",
        task_id=f"tsk_{index}",
        agent=AgentRole.QUANT_RESEARCH,
        outcome=ReflectionOutcome.SUCCESS,
        what_worked="later work",
        what_failed="",
        created_at=NOW + timedelta(days=index + 1),
    )


def test_positive_lift_is_adopted_and_a_protected_target_is_not(stores: Stores) -> None:
    fsm = StateMachineStore(stores.db, stores.audit)
    stores.documents.put(_proposal("imp_ok", ProposalTarget.WORKFLOW), recorded_by="meta_learning")
    stores.documents.put(
        _proposal("imp_held", ProposalTarget.VALIDATION_CRITERIA), recorded_by="meta_learning"
    )
    for index in range(3):
        stores.documents.put(_reflection(index), recorded_by="meta_learning")
    changed = advance_improvements(stores, fsm, NOW + timedelta(days=10))
    assert "imp_ok" in changed
    assert "imp_held" not in changed
    adopted = stores.documents.require(ImprovementProposal, "imp_ok")
    assert adopted.status is ProposalStatus.ROLLED_OUT
    assert adopted.measured_lift is not None and adopted.measured_lift > 0
    assert adopted.postmortem_ids
    assert fsm.state(IMPROVEMENT_PROPOSAL, "imp_ok") == "ROLLED_OUT"
    held = stores.documents.require(ImprovementProposal, "imp_held")
    assert held.status is ProposalStatus.PROPOSED
    assert fsm.state(IMPROVEMENT_PROPOSAL, "imp_held") == "PROPOSED"
    assert held.measured_lift is None
    again = advance_improvements(stores, fsm, NOW + timedelta(days=11))
    assert "imp_ok" not in again
    lessons = stores.documents.list(Lesson)
    assert any("imp_ok" in lesson.statement for lesson in lessons)
