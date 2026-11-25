"""M2.1 learning contracts: scored forecasts and structured postmortems.

A postmortem is institutional memory. It is not a validation result and cannot
raise a claim to L3.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import model_validator

from enigma.schemas.common import (
    Actor,
    ClaimLevel,
    NonEmptyStr,
    Probability,
    StrictModel,
    UtcDatetime,
)


class ErrorClass(StrEnum):
    """Methodological failure versus ordinary sampling variance, plus operational classes."""

    METHOD = "method"
    VARIANCE = "variance"
    INFRASTRUCTURE = "infrastructure"
    DATA = "data"
    POLICY = "policy"


class ForecastRecord(StrictModel):
    """One probability, the information available when it was issued, and its resolution."""

    id: NonEmptyStr
    subject_id: NonEmptyStr
    claim: NonEmptyStr
    probability: Probability
    decided_at: UtcDatetime
    information_refs: tuple[str, ...] = ()
    resolved_at: UtcDatetime | None = None
    outcome: bool | None = None
    scorer: NonEmptyStr
    created_by: Actor
    claim_level: ClaimLevel = ClaimLevel.L0_UNVERIFIED


class Postmortem(StrictModel):
    """Structured reading of one cycle. Method-error claims stay at L1 until an independent eval."""

    id: NonEmptyStr
    cycle_id: NonEmptyStr
    error_class: ErrorClass
    statement: NonEmptyStr
    evidence_refs: tuple[str, ...] = ()
    counterfactual: NonEmptyStr
    claim_level: ClaimLevel = ClaimLevel.L1_HYPOTHESIS
    created_by: Actor
    created_at: UtcDatetime

    @model_validator(mode="after")
    def _claim_is_not_validated(self) -> Postmortem:
        if self.claim_level is ClaimLevel.L3_INDEPENDENTLY_VALIDATED:
            raise ValueError("a postmortem cannot claim L3")
        return self
