"""An improvement is adopted only after the intervention exists and a benchmark says so.

Reflection rates are not an adoption criterion. Protected targets are never adopted
by this module. Rollback restores the previous artifact bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from enigma.core.hashing import sha256_hex, sha256_of
from enigma.fsm.definitions import IMPROVEMENT_PROPOSAL
from enigma.fsm.machine import StateMachineStore
from enigma.schemas.common import AgentRole, ClaimLevel, SystemActor
from enigma.schemas.governance import (
    PROTECTED_TARGETS,
    ImprovementProposal,
    ProposalStatus,
)
from enigma.schemas.learning import ErrorClass, Postmortem
from enigma.store import Stores

_META = AgentRole.META_LEARNING.value
_VAL = SystemActor.VALIDATION_SERVICE.value
_CIO = AgentRole.CIO.value


@dataclass(frozen=True)
class BenchmarkComparison:
    """Paired scores on a fixed task set. Infrastructure failure makes the comparison void."""

    baseline: float
    candidate: float
    task_set_hash: str
    seed: int
    infrastructure_ok: bool
    market_window: str = "held-constant"


@dataclass(frozen=True)
class InterventionResult:
    proposal_id: str
    status: str
    content_hash: str
    baseline_hash: str
    reason: str


def content_hash(text: str) -> str:
    return "sha256:" + sha256_hex(text)


def apply_intervention(
    stores: Stores,
    fsm: StateMachineStore,
    proposal: ImprovementProposal,
    *,
    baseline_text: str,
    candidate_text: str,
    evaluation: str,
    comparison: BenchmarkComparison,
    path: Path,
    now: datetime,
) -> InterventionResult:
    """Implement the candidate, benchmark it, then adopt, reject, or refuse."""
    if proposal.target in PROTECTED_TARGETS or proposal.requires_human_approval:
        return InterventionResult(
            proposal.id,
            "refused",
            "",
            content_hash(baseline_text),
            "protected targets are not self-adopted",
        )
    registered = str(proposal.proposed_change.get("evaluation") or "")
    if not registered or registered != evaluation:
        return InterventionResult(
            proposal.id,
            "refused",
            "",
            content_hash(baseline_text),
            "evaluation does not match the pre-registered record",
        )
    if not comparison.infrastructure_ok:
        return InterventionResult(
            proposal.id,
            "inconclusive",
            "",
            content_hash(baseline_text),
            "infrastructure was not healthy, so the scores are not an improvement",
        )
    if not comparison.task_set_hash:
        return InterventionResult(
            proposal.id, "refused", "", content_hash(baseline_text), "the task set was not fixed"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(candidate_text, encoding="utf-8")
    digest = content_hash(candidate_text)
    baseline = content_hash(baseline_text)
    if fsm.state(IMPROVEMENT_PROPOSAL, proposal.id) is None:
        fsm.create(IMPROVEMENT_PROPOSAL, proposal.id, _META, "intervention_recorded")
    benchmark_id = "bench_" + sha256_hex(f"{proposal.id}|{comparison.task_set_hash}|{digest}")[:26]
    lift = comparison.candidate - comparison.baseline
    record = {
        "evaluation": evaluation,
        "baseline_hash": baseline,
        "content_hash": digest,
        "lift": lift,
        "seed": comparison.seed,
        "market_window": comparison.market_window,
        "task_set_hash": comparison.task_set_hash,
        "infrastructure_ok": True,
        "recorded_at": now.isoformat(),
    }
    fsm.transition(
        IMPROVEMENT_PROPOSAL,
        proposal.id,
        "BENCHMARKING",
        _META,
        "benchmark_started",
        {"benchmark_result": sha256_of(record)},
    )
    if lift <= 0:
        fsm.transition(
            IMPROVEMENT_PROPOSAL,
            proposal.id,
            "REJECTED",
            _VAL,
            "benchmark_did_not_improve",
            {"benchmark_result": sha256_of(record)},
        )
        _save(stores, proposal, ProposalStatus.REJECTED, benchmark_id, lift, record)
        return InterventionResult(
            proposal.id, "rejected", digest, baseline, "candidate did not improve"
        )
    postmortem_id = "pm_intervention_" + sha256_hex(proposal.id)[:20]
    if stores.documents.get(Postmortem, postmortem_id) is None:
        stores.documents.put(
            Postmortem(
                id=postmortem_id,
                cycle_id=proposal.id,
                error_class=ErrorClass.METHOD,
                statement=(
                    f"Intervention {proposal.id} changed the artifact from {baseline} to {digest}. "
                    f"Paired lift was {lift}."
                ),
                evidence_refs=(benchmark_id,),
                counterfactual="The baseline artifact would have remained in place.",
                claim_level=ClaimLevel.L1_HYPOTHESIS,
                created_by=AgentRole.META_LEARNING,
                created_at=now,
            ),
            recorded_by=_META,
        )
    fsm.transition(
        IMPROVEMENT_PROPOSAL,
        proposal.id,
        "APPROVED",
        _VAL,
        "benchmark_improved",
        {"benchmark_result": sha256_of(record)},
    )
    fsm.transition(
        IMPROVEMENT_PROPOSAL,
        proposal.id,
        "ROLLED_OUT",
        _CIO,
        "artifact_installed",
        {"version": digest},
    )
    _save(
        stores,
        proposal,
        ProposalStatus.ROLLED_OUT,
        benchmark_id,
        lift,
        record,
        postmortem_ids=(postmortem_id,),
    )
    stores.flags.set(f"improvement.monitor:{proposal.id}", str(comparison.candidate), "system")
    return InterventionResult(
        proposal.id, "adopted", digest, baseline, "benchmark lift was positive"
    )


def rollback(
    stores: Stores,
    fsm: StateMachineStore,
    proposal: ImprovementProposal,
    path: Path,
    previous_text: str,
) -> InterventionResult:
    previous = content_hash(previous_text)
    path.write_text(previous_text, encoding="utf-8")
    fsm.transition(
        IMPROVEMENT_PROPOSAL,
        proposal.id,
        "REVERTED",
        _CIO,
        "rollback",
        {"previous_version": previous},
    )
    current = stores.documents.get(ImprovementProposal, proposal.id) or proposal
    _save(stores, current, ProposalStatus.REVERTED, current.benchmark_id, current.measured_lift, {})
    return InterventionResult(
        proposal.id, "reverted", content_hash(previous_text), previous, "restored"
    )


def _save(
    stores: Stores,
    proposal: ImprovementProposal,
    status: ProposalStatus,
    benchmark_id: str | None,
    lift: float | None,
    record: dict[str, object],
    postmortem_ids: tuple[str, ...] = (),
) -> None:
    updated = proposal.model_copy(
        update={
            "status": status,
            "benchmark_id": benchmark_id,
            "measured_lift": lift,
            "postmortem_ids": postmortem_ids or proposal.postmortem_ids,
            "proposed_change": {**proposal.proposed_change, "benchmark": record},
        }
    )
    stores.documents.put(updated, recorded_by=_META)
