"""Research-plane contracts: objectives, datasets, hypotheses, experiments, strategy packages."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from enigma.schemas.common import (
    Actor,
    ClaimLevel,
    Instrument,
    NonEmptyStr,
    Sha256,
    StrictModel,
    UtcDatetime,
)


class ResearchObjective(StrictModel):
    id: NonEmptyStr
    statement: NonEmptyStr
    instruments: tuple[Instrument, ...]
    topics: tuple[str, ...] = ()
    max_trials: int = Field(default=20, ge=1)
    max_usd: float = Field(default=0.0, ge=0.0)
    created_by: Actor
    created_at: UtcDatetime


class DataQualityCheck(StrictModel):
    name: NonEmptyStr
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class DataQualityReport(StrictModel):
    checks: tuple[DataQualityCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


class DatasetVersion(StrictModel):
    """Immutable, content-addressed dataset version (Parquet on disk)."""

    id: NonEmptyStr
    name: NonEmptyStr
    instruments: tuple[Instrument, ...]
    venue: NonEmptyStr
    frequency: NonEmptyStr = Field(description="e.g. 1h, 8h, 1d")
    schema_version: NonEmptyStr
    columns: tuple[str, ...]
    time_start: UtcDatetime
    time_end: UtcDatetime
    row_count: int = Field(ge=0)
    content_hash: Sha256
    uri: NonEmptyStr
    pit_certified: bool
    quality: DataQualityReport
    provenance: dict[str, Any] = Field(description="Endpoints, download URLs, retrieval time")
    survivorship_notes: str = ""
    plane: NonEmptyStr = "research"
    created_by: Actor
    created_at: UtcDatetime


class HypothesisStatus(StrEnum):
    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED_IN_RESEARCH = "supported_in_research"
    REJECTED_IN_RESEARCH = "rejected_in_research"
    VALIDATED = "validated"
    INVALIDATED = "invalidated"


class Hypothesis(StrictModel):
    id: NonEmptyStr
    objective_id: NonEmptyStr
    family_id: NonEmptyStr = Field(description="Multiple-testing family")
    title: NonEmptyStr
    mechanism: NonEmptyStr = Field(description="Causal/structural explanation")
    prediction: NonEmptyStr
    formal_statement: NonEmptyStr = Field(description="Mathematical, falsifiable statement")
    falsification_criteria: tuple[str, ...] = Field(min_length=1)
    instruments: tuple[Instrument, ...] = Field(min_length=1)
    horizon: NonEmptyStr
    features: tuple[str, ...] = Field(min_length=1)
    parameters: dict[str, float | int | str] = Field(default_factory=dict)
    search_space: dict[str, tuple[float | int | str, ...]] = Field(default_factory=dict)
    trial_budget: int = Field(ge=1)
    cost_model_version: NonEmptyStr
    evidence_ids: tuple[str, ...] = ()
    contradictory_evidence_ids: tuple[str, ...] = ()
    lesson_ids: tuple[str, ...] = Field(default=(), description="Prior lessons consulted")
    parent_hypothesis_id: str | None = None
    uncertainty_notes: str = ""
    generation_method: NonEmptyStr = Field(description="e.g. template:funding-carry-v1, llm:<id>")
    claim_level: ClaimLevel = ClaimLevel.L1_HYPOTHESIS
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    methodology_version: NonEmptyStr
    created_by: Actor
    created_at: UtcDatetime

    @model_validator(mode="after")
    def _claim_level_bounded(self) -> Hypothesis:
        if self.claim_level == ClaimLevel.L3_INDEPENDENTLY_VALIDATED:
            raise ValueError("Hypotheses cannot carry L3; only ValidationReports confer L3")
        return self


class CostModel(StrictModel):
    """Versioned execution-cost assumptions; validation re-applies these independently."""

    version: NonEmptyStr
    taker_fee_bps: float = Field(ge=0)
    maker_fee_bps: float = Field(ge=0)
    half_spread_bps: float = Field(ge=0)
    impact_bps_per_million: float = Field(ge=0, description="Linear impact proxy per $1M notional")
    apply_funding: bool = True
    capacity_notional_usd: float = Field(gt=0)
    latency_bars: int = Field(default=1, ge=0, description="Signal-to-fill delay in bars")


class StrategyPackage(StrictModel):
    """Content-addressed strategy code + config, executed only inside the sandbox."""

    id: NonEmptyStr
    hypothesis_id: NonEmptyStr
    name: NonEmptyStr
    source_code: NonEmptyStr = Field(description="Python module source implementing the interface")
    entrypoint: NonEmptyStr = Field(description="Class name implementing Strategy")
    params: dict[str, float | int | str] = Field(default_factory=dict)
    required_columns: tuple[str, ...]
    content_hash: Sha256
    interface_version: NonEmptyStr = "1.0.0"
    cost_model: CostModel
    label_horizon_bars: int = Field(
        default=1, ge=1, description="Bars the signal targets; sets the G02 embargo"
    )
    sandbox_profile: NonEmptyStr = "default"
    created_by: Actor
    created_at: UtcDatetime


class ExperimentStatus(StrEnum):
    REGISTERED = "registered"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


class Experiment(StrictModel):
    """Registered before it runs; every trial counts toward multiple-testing accounting."""

    id: NonEmptyStr
    hypothesis_id: NonEmptyStr
    family_id: NonEmptyStr
    trial_index: int = Field(ge=1)
    strategy_package_id: NonEmptyStr
    dataset_version_id: NonEmptyStr
    research_window: tuple[UtcDatetime, UtcDatetime]
    code_sha: NonEmptyStr
    config_hash: Sha256
    seed: int
    status: ExperimentStatus = ExperimentStatus.REGISTERED
    metrics: dict[str, float] = Field(default_factory=dict)
    result_hash: Sha256 | None = None
    artifact_uri: str | None = None
    error: str | None = None
    created_by: Actor
    started_at: UtcDatetime | None = None
    ended_at: UtcDatetime | None = None
