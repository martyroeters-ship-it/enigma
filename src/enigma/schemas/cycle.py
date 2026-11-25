"""Research-cycle contracts: market analyses, next-experiment proposals, risk reviews."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from enigma.schemas.common import Actor, NonEmptyStr, StrictModel, UtcDatetime


class MarketAnalysis(StrictModel):
    """Enigma-computed measurements on a research dataset (not external evidence)."""

    id: NonEmptyStr
    dataset_version_id: NonEmptyStr
    method: NonEmptyStr = Field(description="e.g. deterministic:funding-basis-stats-v1")
    per_symbol: dict[str, dict[str, float]]
    findings: tuple[str, ...]
    created_by: Actor
    created_at: UtcDatetime


class ProposalState(StrEnum):
    PROPOSED = "proposed"
    SCHEDULED = "scheduled"
    EXECUTED = "executed"
    DISMISSED = "dismissed"


class NextExperimentProposal(StrictModel):
    """IMPLEMENTATION_PLAN §2 WS8: `{parent_run_id, hypothesis_delta, rationale_refs}`."""

    id: NonEmptyStr
    parent_cycle_id: NonEmptyStr
    parent_hypothesis_id: NonEmptyStr | None
    mechanism_id: NonEmptyStr
    hypothesis_delta: dict[str, Any] = Field(
        description="Changes vs parent: mechanism, parameters, search space, horizon"
    )
    rationale: NonEmptyStr
    lesson_ids: tuple[str, ...] = Field(min_length=1)
    validation_summary_id: str | None = None
    state: ProposalState = ProposalState.PROPOSED
    child_cycle_id: str | None = None
    proposed_by: Actor
    created_at: UtcDatetime


class RiskDecision(StrEnum):
    VETO = "veto"
    CLEARANCE = "clearance"


class RiskReview(StrictModel):
    id: NonEmptyStr
    strategy_package_id: NonEmptyStr
    decision: RiskDecision
    reason_codes: tuple[str, ...]
    checks: dict[str, bool]
    limits_version: NonEmptyStr
    created_by: Actor
    created_at: UtcDatetime
