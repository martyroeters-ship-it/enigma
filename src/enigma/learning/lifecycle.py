"""Move an improvement from proposal to a measured adoption or a rejection.

Protected targets (validation criteria, risk limits, permissions) stay unadopted.
A proposal is adopted only after an independent measurement on observations that
arrived after the proposal, with positive lift and a postmortem. The proposing
agent does not score its own change.
"""

from __future__ import annotations

from datetime import datetime

from enigma.core.hashing import sha256_hex
from enigma.fsm import IMPROVEMENT_PROPOSAL, StateMachineStore
from enigma.learning.adoption import measured_lift
from enigma.schemas.common import AgentRole, ClaimLevel, SystemActor
from enigma.schemas.governance import (
    PROTECTED_TARGETS,
    ImprovementProposal,
    ProposalStatus,
    Reflection,
    ReflectionOutcome,
)
from enigma.schemas.knowledge import Lesson
from enigma.schemas.learning import ErrorClass, Postmortem
from enigma.store import Stores

_MIN_LATER = 3


def advance_improvements(stores: Stores, fsm: StateMachineStore, now: datetime) -> list[str]:
    """Idempotent. Returns proposal ids whose state changed."""
    changed: list[str] = []
    for proposal in stores.documents.list(ImprovementProposal, limit=500):
        state = fsm.state(IMPROVEMENT_PROPOSAL, proposal.id)
        if state is None:
            fsm.create(IMPROVEMENT_PROPOSAL, proposal.id, AgentRole.META_LEARNING.value, "proposed")
            state = "PROPOSED"
        if proposal.target in PROTECTED_TARGETS or proposal.requires_human_approval:
            if state == "PROPOSED":
                stores.audit.record(
                    SystemActor.VALIDATION_SERVICE.value,
                    "improvement.held",
                    proposal.id,
                    {"reason": "protected target requires a human decision"},
                )
            continue
        if state == "PROPOSED":
            fsm.transition(
                IMPROVEMENT_PROPOSAL,
                proposal.id,
                "BENCHMARKING",
                AgentRole.META_LEARNING.value,
                "benchmark_started",
            )
            _put(stores, proposal, status=ProposalStatus.BENCHMARKING)
            state = "BENCHMARKING"
            changed.append(proposal.id)
        if state != "BENCHMARKING":
            if state == "ROLLED_OUT":
                _follow_up(stores, proposal, now)
            continue
        later = _later_reflections(stores, proposal)
        if len(later) < _MIN_LATER:
            continue
        baseline = proposal.baseline_metric_value
        if baseline is None:
            continue
        higher_is_better = not proposal.success_metric.endswith("failure_rate")
        current = _failure_rate(later) if not higher_is_better else _mean_metric(later, "usd")
        lift = measured_lift(baseline, current, higher_is_better=higher_is_better)
        benchmark_id = "bm_" + sha256_hex(f"{proposal.id}|{lift}|{len(later)}")[:26]
        if lift <= 0:
            fsm.transition(
                IMPROVEMENT_PROPOSAL,
                proposal.id,
                "REJECTED",
                SystemActor.VALIDATION_SERVICE.value,
                "lift_not_positive",
                {"benchmark_result": sha256_hex(str(lift))},
            )
            _put(
                stores,
                proposal,
                status=ProposalStatus.REJECTED,
                benchmark_id=benchmark_id,
                lift=lift,
            )
            changed.append(proposal.id)
            continue
        postmortem_id = _postmortem(stores, proposal, lift, benchmark_id, now)
        approved = _put(
            stores,
            proposal,
            status=ProposalStatus.APPROVED,
            benchmark_id=benchmark_id,
            lift=lift,
            postmortem_ids=(postmortem_id,),
        )
        fsm.transition(
            IMPROVEMENT_PROPOSAL,
            proposal.id,
            "APPROVED",
            SystemActor.VALIDATION_SERVICE.value,
            "lift_positive",
            {"benchmark_result": sha256_hex(f"{benchmark_id}|{lift}")},
        )
        _put(
            stores,
            approved,
            status=ProposalStatus.ROLLED_OUT,
            benchmark_id=benchmark_id,
            lift=lift,
            postmortem_ids=(postmortem_id,),
        )
        fsm.transition(
            IMPROVEMENT_PROPOSAL,
            proposal.id,
            "ROLLED_OUT",
            AgentRole.CIO.value,
            "practice_recorded",
            {"version": sha256_hex(proposal.id)},
        )
        stores.flags.set(
            f"practice.{proposal.id}",
            f"{lift:.6f}",
            SystemActor.VALIDATION_SERVICE.value,
        )
        stores.audit.record(
            SystemActor.VALIDATION_SERVICE.value,
            "improvement.adopted",
            proposal.id,
            {"lift": lift, "benchmark_id": benchmark_id, "later_observations": len(later)},
        )
        changed.append(proposal.id)
    return changed


def _later_reflections(stores: Stores, proposal: ImprovementProposal) -> list[Reflection]:
    return [
        item
        for item in stores.documents.list(Reflection, limit=2000)
        if item.created_at > proposal.created_at and item.agent.value == proposal.target_ref
    ]


def _failure_rate(items: list[Reflection]) -> float:
    failures = sum(1 for item in items if item.outcome is not ReflectionOutcome.SUCCESS)
    return failures / len(items)


def _mean_metric(items: list[Reflection], key: str) -> float:
    return sum(item.metrics.get(key, 0.0) for item in items) / len(items)


def _postmortem(
    stores: Stores, proposal: ImprovementProposal, lift: float, benchmark_id: str, now: datetime
) -> str:
    postmortem_id = "pm_" + sha256_hex(f"{proposal.id}|{benchmark_id}")[:26].upper()
    if stores.documents.get(Postmortem, postmortem_id) is None:
        stores.documents.put(
            Postmortem(
                id=postmortem_id,
                cycle_id=proposal.id,
                error_class=ErrorClass.METHOD,
                statement=(
                    f"Independent benchmark {benchmark_id} measured lift {lift:.4f} "
                    f"on {proposal.success_metric} after the proposal was filed."
                ),
                evidence_refs=proposal.evidence_ids[:20],
                counterfactual="Leave the practice unchanged and keep the baseline.",
                created_by=AgentRole.INDEPENDENT_VALIDATION,
                created_at=now,
            ),
            recorded_by=SystemActor.VALIDATION_SERVICE.value,
        )
    return postmortem_id


def _put(
    stores: Stores,
    proposal: ImprovementProposal,
    *,
    status: ProposalStatus,
    benchmark_id: str | None = None,
    lift: float | None = None,
    postmortem_ids: tuple[str, ...] | None = None,
) -> ImprovementProposal:
    updated = proposal.model_copy(
        update={
            "status": status,
            "benchmark_id": benchmark_id if benchmark_id is not None else proposal.benchmark_id,
            "measured_lift": lift if lift is not None else proposal.measured_lift,
            "postmortem_ids": postmortem_ids
            if postmortem_ids is not None
            else proposal.postmortem_ids,
        }
    )
    stores.documents.put(updated, recorded_by=SystemActor.VALIDATION_SERVICE.value)
    return updated


def _follow_up(stores: Stores, proposal: ImprovementProposal, now: datetime) -> None:
    flag = f"practice.{proposal.id}.followup"
    if stores.flags.get(flag) is not None or proposal.measured_lift is None:
        return
    later = _later_reflections(stores, proposal)
    if len(later) < _MIN_LATER:
        return
    lesson_id = "les_" + sha256_hex(f"{proposal.id}|followup")[:26].upper()
    if stores.documents.get(Lesson, lesson_id) is not None:
        stores.flags.set(flag, lesson_id, "system")
        return
    current = _failure_rate(later)
    stores.documents.put(
        Lesson(
            id=lesson_id,
            statement=(
                f"After adopting {proposal.id}, the later {proposal.success_metric} "
                f"is {current:.3f} against measured lift {proposal.measured_lift:.3f}."
            ),
            topics=("improvement_followup", proposal.id),
            derived_from_ids=(proposal.id,),
            outcome="success" if proposal.measured_lift > 0 else "failure",
            claim_level=ClaimLevel.L1_HYPOTHESIS,
            created_by=AgentRole.META_LEARNING,
            created_at=now,
        ),
        recorded_by=AgentRole.META_LEARNING.value,
    )
    stores.flags.set(flag, lesson_id, "system")
