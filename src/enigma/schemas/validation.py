"""Validation-plane contracts. `overall` is computed by a pure function, never by a model."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from enigma.schemas.common import NonEmptyStr, Sha256, StrictModel, UtcDatetime


class GateOutcome(StrEnum):
    PASS = "PASS"  # noqa: S105 - gate outcome, not a credential
    FAIL = "FAIL"


class GateResult(StrictModel):
    gate_id: NonEmptyStr = Field(description="G01..G16")
    name: NonEmptyStr
    result: GateOutcome
    stat: float | None
    threshold: float | None
    comparator: NonEmptyStr = Field(description="e.g. '>=', '<=', '==' — how stat meets threshold")
    params: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


def aggregate(gates: tuple[GateResult, ...], required: tuple[str, ...]) -> GateOutcome:
    """PASS iff every required gate is present and PASS. Missing gates fail closed."""
    by_id = {gate.gate_id: gate for gate in gates}
    for gate_id in required:
        gate = by_id.get(gate_id)
        if gate is None or gate.result is not GateOutcome.PASS:
            return GateOutcome.FAIL
    return GateOutcome.PASS if required else GateOutcome.FAIL


class ValidationRequest(StrictModel):
    id: NonEmptyStr
    strategy_package_id: NonEmptyStr
    experiment_ids: tuple[str, ...] = Field(min_length=1)
    family_id: NonEmptyStr
    dataset_version_id: NonEmptyStr = Field(description="Research dataset the package was built on")
    requested_by: NonEmptyStr
    requested_at: UtcDatetime


class ValidationReport(StrictModel):
    id: NonEmptyStr
    request_id: NonEmptyStr
    strategy_package_id: NonEmptyStr
    protocol_version: NonEmptyStr
    required_gates: tuple[str, ...]
    gates: tuple[GateResult, ...]
    overall: GateOutcome
    holdout_dataset_ids: tuple[str, ...] = Field(min_length=1)
    trial_count: int = Field(ge=1)
    deflated_sharpe: float | None = None
    pbo: float | None = None
    net_sharpe_holdout: float | None = None
    cost_assumptions_hash: Sha256
    validator_build_sha: NonEmptyStr
    summary: str = Field(default="", description="Non-authoritative human-readable summary")
    created_at: UtcDatetime

    @model_validator(mode="after")
    def _overall_is_aggregate(self) -> ValidationReport:
        if aggregate(self.gates, self.required_gates) is not self.overall:
            raise ValueError("overall must equal aggregate(gates, required_gates)")
        return self


class RedactedValidationSummary(StrictModel):
    """What the research plane may see: pass/fail per gate, no holdout path structure."""

    id: NonEmptyStr
    report_id: NonEmptyStr
    strategy_package_id: NonEmptyStr
    protocol_version: NonEmptyStr
    overall: GateOutcome
    gate_outcomes: dict[str, GateOutcome]
    failed_gate_names: tuple[str, ...]
