"""Classify a finished or aborted cycle as method error or ordinary variance.

The label is an L1 institutional reading. It does not change validation `overall`
and it cannot adopt an improvement.
"""

from __future__ import annotations

from enigma.core.hashing import sha256_hex
from enigma.learning.abort import AbortClass, diagnose_abort
from enigma.schemas.common import AgentRole, UtcDatetime
from enigma.schemas.learning import ErrorClass, Postmortem

# Construction and integrity failures are not sampling noise.
_METHOD_GATES = frozenset({"G01", "G02", "G16"})

_ABORT_CLASS: dict[AbortClass, ErrorClass] = {
    AbortClass.INFRASTRUCTURE: ErrorClass.INFRASTRUCTURE,
    AbortClass.DATA: ErrorClass.DATA,
    AbortClass.BUDGET: ErrorClass.POLICY,
    AbortClass.POLICY: ErrorClass.METHOD,
    AbortClass.STRATEGY_RUNTIME: ErrorClass.METHOD,
    AbortClass.RESEARCH: ErrorClass.METHOD,
}


def _gate_ids(failed_gate_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name.split(":", 1)[0] for name in failed_gate_names)


def classify_validation_failure(failed_gate_names: tuple[str, ...]) -> ErrorClass:
    ids = _gate_ids(failed_gate_names)
    if not ids:
        raise ValueError("a passing validation is not a failure postmortem")
    if any(gate_id in _METHOD_GATES for gate_id in ids):
        return ErrorClass.METHOD
    return ErrorClass.VARIANCE


def classify_abort(reason: str, step: str | None) -> ErrorClass:
    return _ABORT_CLASS[diagnose_abort(reason, step).abort_class]


def postmortem_for_validation(
    cycle_id: str,
    *,
    summary_id: str,
    failed_gate_names: tuple[str, ...],
    now: UtcDatetime,
) -> Postmortem:
    error_class = classify_validation_failure(failed_gate_names)
    if error_class is ErrorClass.METHOD:
        counterfactual = (
            "Repair the split, leakage, or reproducibility defect and submit a new package. "
            "Do not treat this FAIL as evidence the mechanism was only unlucky."
        )
    else:
        counterfactual = (
            "Retain the failure in the trial ledger. A new package is required before any "
            "promotion; this FAIL is consistent with ordinary variance on the sealed sample."
        )
    return Postmortem(
        id="pm_" + sha256_hex(f"{cycle_id}|validation|{summary_id}")[:26],
        cycle_id=cycle_id,
        error_class=error_class,
        statement=(
            f"Validation FAIL classified as {error_class.value}. "
            f"Failed gates: {', '.join(failed_gate_names)}."
        ),
        evidence_refs=(cycle_id, summary_id),
        counterfactual=counterfactual,
        created_by=AgentRole.META_LEARNING,
        created_at=now,
    )


def postmortem_for_abort(
    cycle_id: str,
    *,
    abort_reason: str,
    step: str | None,
    now: UtcDatetime,
) -> Postmortem:
    diagnosis = diagnose_abort(abort_reason, step)
    error_class = _ABORT_CLASS[diagnosis.abort_class]
    return Postmortem(
        id="pm_" + sha256_hex(f"{cycle_id}|abort|{diagnosis.root_cause}")[:26],
        cycle_id=cycle_id,
        error_class=error_class,
        statement=(
            f"Abort at {diagnosis.step} classified as {error_class.value} ({diagnosis.root_cause})."
        ),
        evidence_refs=(cycle_id,),
        counterfactual=diagnosis.prevention,
        created_by=AgentRole.META_LEARNING,
        created_at=now,
    )
