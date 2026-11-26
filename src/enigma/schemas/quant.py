"""Autonomous quant-research contracts: research context, hypothesis proposals and their
pre-registration, strategy specs, and LLM generation records.

Everything an LLM sees or produces is typed and carries ids, so citations, columns and
parameters can be checked programmatically before anything is registered or executed.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from enigma.core.hashing import sha256_of
from enigma.schemas.common import (
    Actor,
    ClaimLevel,
    NonEmptyStr,
    Sha256,
    StrictModel,
    UtcDatetime,
)
from enigma.schemas.governance import ExecutionMode, ModelProvenance

ComplexityClass = Literal["rule", "statistical", "ml"]
ColumnRole = Literal["signal", "target", "control"]
NonEmptyStrTuple = tuple[NonEmptyStr, ...]


# --------------------------------------------------------------------------- research context
class EvidenceExcerpt(StrictModel):
    id: NonEmptyStr
    source_id: NonEmptyStr
    text: NonEmptyStr = Field(description="Verbatim retrieved excerpt (untrusted data)")
    claim_level: ClaimLevel
    topics: tuple[str, ...] = ()


class PriorHypothesisOutcome(StrictModel):
    hypothesis_id: NonEmptyStr
    family_id: NonEmptyStr
    title: NonEmptyStr
    mechanism: NonEmptyStr
    signal: NonEmptyStr = Field(description="Formal statement / signal definition")
    columns: tuple[str, ...]
    horizon_bars: int = Field(ge=1)
    generation_method: NonEmptyStr
    validation: Literal["pass", "fail", "untested"]
    failed_gates: tuple[str, ...] = ()
    trials: int = Field(default=0, ge=0)


class LessonNote(StrictModel):
    id: NonEmptyStr
    statement: NonEmptyStr
    outcome: NonEmptyStr
    topics: tuple[str, ...] = ()


class FailureNote(StrictModel):
    id: NonEmptyStr
    kind: NonEmptyStr
    description: NonEmptyStr
    gate_id: str | None = None


class DatasetInfo(StrictModel):
    dataset_version_id: NonEmptyStr
    name: NonEmptyStr
    frequency: NonEmptyStr
    bar_hours: float = Field(gt=0)
    symbols: tuple[str, ...]
    columns: tuple[str, ...]
    coverage_start: UtcDatetime
    coverage_end: UtcDatetime = Field(description="Last bar usable for research (pre-cutoff)")
    rows: int = Field(ge=0)


class StatObservation(StrictModel):
    """Deterministic statistic computed from research-window data strictly before the cutoff."""

    id: NonEmptyStr
    symbol: NonEmptyStr
    name: NonEmptyStr
    definition: NonEmptyStr
    value: float
    sample_size: int = Field(ge=0)
    t_stat: float | None = None
    data_end: UtcDatetime = Field(description="Latest available_at used")


class ContradictionPair(StrictModel):
    evidence_a: NonEmptyStr
    evidence_b: NonEmptyStr
    basis: NonEmptyStr
    method: NonEmptyStr


class DomainGap(StrictModel):
    domain: NonEmptyStr
    priority: NonEmptyStr
    prior_hypotheses: int = Field(ge=0)
    data_available: bool
    note: str = ""


class ResearchContext(StrictModel):
    """Everything the hypothesis generator may use. Nothing outside this object is citable."""

    id: NonEmptyStr
    objective_id: NonEmptyStr
    objective: NonEmptyStr
    research_cutoff: UtcDatetime
    instruments: tuple[str, ...] = Field(min_length=1)
    cost_model_version: NonEmptyStr
    cost_bps_per_unit_turnover: float = Field(ge=0)
    evidence: tuple[EvidenceExcerpt, ...] = ()
    prior_hypotheses: tuple[PriorHypothesisOutcome, ...] = ()
    lessons: tuple[LessonNote, ...] = ()
    failures: tuple[FailureNote, ...] = ()
    datasets: tuple[DatasetInfo, ...] = Field(min_length=1)
    observations: tuple[StatObservation, ...] = ()
    contradictions: tuple[ContradictionPair, ...] = ()
    underexplored: tuple[DomainGap, ...] = ()

    def content_hash(self) -> str:
        return sha256_of(self.model_dump(mode="json", exclude={"id"}))

    def evidence_ids(self) -> frozenset[str]:
        return frozenset(e.id for e in self.evidence)

    def columns_by_dataset(self) -> dict[str, frozenset[str]]:
        return {d.dataset_version_id: frozenset(d.columns) for d in self.datasets}


# --------------------------------------------------------------------------- hypothesis proposal
class ColumnUse(StrictModel):
    dataset_id: NonEmptyStr
    column: NonEmptyStr
    role: ColumnRole
    min_lag_bars: int = Field(
        description="Most recent bar used relative to decision bar t (0 = bar t at its close). "
        "Signals and controls must be >= 0; targets are forward by definition."
    )


class ParameterGrid(StrictModel):
    name: NonEmptyStr = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")
    values: tuple[float | int, ...] = Field(min_length=1, max_length=6)


class ExperimentalDesign(StrictModel):
    parameters: tuple[ParameterGrid, ...] = Field(min_length=1)
    grid_size: int = Field(ge=1)
    train_test_protocol: NonEmptyStr
    primary_metric: NonEmptyStr

    @model_validator(mode="after")
    def _grid_size_consistent(self) -> ExperimentalDesign:
        size = 1
        for p in self.parameters:
            size *= len(p.values)
        if size != self.grid_size:
            raise ValueError(f"grid_size {self.grid_size} != product of grid lengths {size}")
        if len({p.name for p in self.parameters}) != len(self.parameters):
            raise ValueError("parameter names must be unique")
        return self


class ExecutionCostEstimate(StrictModel):
    turnover_per_bar: float = Field(ge=0, le=2, description="Expected mean |position change|")
    bars_per_year: float = Field(gt=0)
    cost_bps_per_unit_turnover: float = Field(ge=0)
    annual_cost_bps: float = Field(ge=0)
    cost_model_version: NonEmptyStr
    rationale: NonEmptyStr


class HypothesisProposal(StrictModel):
    """A fully specified, checkable research hypothesis. Every field is required and non-empty."""

    id: NonEmptyStr
    title: NonEmptyStr
    domain: NonEmptyStr
    market_mechanism: NonEmptyStr
    inefficiency_rationale: NonEmptyStr = Field(description="Why the inefficiency exists")
    persistence_rationale: NonEmptyStr = Field(description="Why it is not arbitraged away")
    signal_definition: NonEmptyStr
    target_definition: NonEmptyStr
    test_statistic: NonEmptyStr
    required_data: tuple[ColumnUse, ...] = Field(min_length=1)
    horizon_bars: int = Field(ge=1)
    horizon: NonEmptyStr
    null_hypothesis: NonEmptyStr
    alternative_hypothesis: NonEmptyStr
    design: ExperimentalDesign
    confounders: NonEmptyStrTuple = Field(min_length=1)
    execution_costs: ExecutionCostEstimate
    failure_conditions: NonEmptyStrTuple = Field(min_length=1)
    supporting_evidence_ids: NonEmptyStrTuple = Field(min_length=1)
    # Empty is allowed: local models often have no genuine contrary citation.
    contradictory_evidence_ids: tuple[NonEmptyStr, ...] = Field(default=())
    novelty_statement: NonEmptyStr
    complexity_class: ComplexityClass
    complexity_rationale: NonEmptyStr
    instruments: NonEmptyStrTuple = Field(min_length=1)

    def columns(self, role: ColumnRole | None = None) -> tuple[str, ...]:
        return tuple(
            sorted({c.column for c in self.required_data if role is None or c.role == role})
        )

    def search_space(self) -> dict[str, tuple[float | int, ...]]:
        return {p.name: p.values for p in sorted(self.design.parameters, key=lambda p: p.name)}


class CheckFailure(StrictModel):
    check: NonEmptyStr
    detail: NonEmptyStr


class RejectedDraft(StrictModel):
    title: str
    round: int = Field(ge=0)
    reasons: tuple[CheckFailure, ...] = Field(min_length=1)
    draft: dict[str, Any]


class DatasetAcquisitionRequest(StrictModel):
    """A draft needed data the institution does not hold; Data Engineering decides on it."""

    id: NonEmptyStr
    requested_by: Actor
    description: NonEmptyStr
    columns: tuple[str, ...]
    rationale: NonEmptyStr
    source_draft_title: str = ""
    generation_id: str = ""
    created_at: UtcDatetime


class GenerationResult(StrictModel):
    """One hypothesis-generation run: survivors, rejections, and full provenance."""

    id: NonEmptyStr
    context_id: NonEmptyStr
    context_hash: Sha256
    execution_mode: ExecutionMode
    method: NonEmptyStr
    prompt_refs: tuple[str, ...]
    provenance: tuple[ModelProvenance, ...]
    proposals: tuple[HypothesisProposal, ...]
    rejected: tuple[RejectedDraft, ...]
    acquisition_requests: tuple[DatasetAcquisitionRequest, ...] = ()
    repair_rounds: int = Field(ge=0, le=1)
    failure_reason: str | None = None
    created_by: Actor
    created_at: UtcDatetime

    @property
    def ok(self) -> bool:
        return bool(self.proposals)

    @property
    def prompt_ref(self) -> str:
        return self.prompt_refs[0] if self.prompt_refs else ""

    @model_validator(mode="after")
    def _failure_is_explicit(self) -> GenerationResult:
        if not self.proposals and not self.failure_reason:
            raise ValueError("a generation with no surviving proposal must state failure_reason")
        return self


# --------------------------------------------------------------------------- registration
class RegisteredHypothesis(StrictModel):
    """Pre-registered, immutable hypothesis spec. Linked to the cycle's `Hypothesis` by id.

    `content_hash` covers every other field; a record whose hash does not verify is rejected.
    Amendments are new records (`amends=<old id>`) in the same multiple-testing family.
    """

    id: NonEmptyStr
    hypothesis_id: NonEmptyStr
    objective_id: NonEmptyStr
    family_id: NonEmptyStr
    proposal: HypothesisProposal
    trial_budget: int = Field(ge=1)
    cost_model_version: NonEmptyStr
    dataset_version_ids: tuple[str, ...] = Field(min_length=1)
    amends: str | None = None
    lineage: tuple[str, ...] = Field(default=(), description="Ancestor spec ids, oldest first")
    family_members: tuple[str, ...] = Field(
        min_length=1, description="Hypothesis ids in the family, including this one"
    )
    cumulative_trial_budget: int = Field(ge=1)
    generation_id: NonEmptyStr
    execution_mode: ExecutionMode
    prompt_ref: str = ""
    provenance: tuple[ModelProvenance, ...] = ()
    research_cutoff: UtcDatetime
    registered_by: Actor
    registered_at: UtcDatetime
    content_hash: Sha256

    @staticmethod
    def hash_fields(data: dict[str, Any]) -> str:
        return sha256_of({k: v for k, v in data.items() if k != "content_hash"})

    @model_validator(mode="after")
    def _hash_verifies(self) -> RegisteredHypothesis:
        expected = self.hash_fields(self.model_dump(mode="json"))
        if expected != self.content_hash:
            raise ValueError("RegisteredHypothesis content_hash does not match its content")
        return self


# --------------------------------------------------------------------------- strategy spec
class FeatureDef(StrictModel):
    name: NonEmptyStr = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")
    formula: NonEmptyStr = Field(description="Math over input columns at bars <= t")
    inputs: NonEmptyStrTuple = Field(min_length=1)
    lookback_bars: int = Field(ge=0)
    min_lag_bars: int = Field(description="Most recent bar used relative to t; must be >= 0")


class BaselineSpec(StrictModel):
    name: NonEmptyStr
    signal_math: NonEmptyStr
    features: tuple[FeatureDef, ...] = Field(min_length=1)
    parameters: tuple[ParameterGrid, ...] = ()
    position_rule: NonEmptyStr
    complexity_class: ComplexityClass = "rule"


class StrategySpec(StrictModel):
    id: NonEmptyStr
    hypothesis_spec_id: NonEmptyStr
    hypothesis_id: NonEmptyStr
    name: NonEmptyStr = Field(pattern=r"^[a-z][a-z0-9_]{0,48}$")
    signal_math: NonEmptyStr
    features: tuple[FeatureDef, ...] = Field(min_length=1)
    parameters: tuple[ParameterGrid, ...] = Field(min_length=1)
    position_rule: NonEmptyStr
    position_bounds: tuple[float, float] = (-1.0, 1.0)
    rebalance_bars: int = Field(ge=1)
    holding_bars: int = Field(ge=1)
    complexity_class: ComplexityClass
    complexity_justification: NonEmptyStr
    baseline: BaselineSpec
    required_columns: NonEmptyStrTuple = Field(min_length=1)
    label_horizon_bars: int = Field(ge=1)
    execution_mode: ExecutionMode
    prompt_ref: str = ""
    provenance: tuple[ModelProvenance, ...] = ()
    content_hash: Sha256
    created_by: Actor
    created_at: UtcDatetime

    @staticmethod
    def hash_fields(data: dict[str, Any]) -> str:
        keep = {
            k: v
            for k, v in data.items()
            if k not in {"content_hash", "provenance", "created_at", "created_by", "id"}
        }
        return sha256_of(keep)

    @model_validator(mode="after")
    def _checks(self) -> StrategySpec:
        lo, hi = self.position_bounds
        if not (-1.0 <= lo <= 0.0 <= hi <= 1.0):
            raise ValueError("position_bounds must lie within [-1, 1] and contain 0")
        if self.hash_fields(self.model_dump(mode="json")) != self.content_hash:
            raise ValueError("StrategySpec content_hash does not match its content")
        return self

    def search_space(self) -> dict[str, tuple[float | int, ...]]:
        return {p.name: p.values for p in sorted(self.parameters, key=lambda p: p.name)}


class ConsistencyReport(StrictModel):
    spec_id: NonEmptyStr
    hypothesis_spec_id: NonEmptyStr
    checks: dict[str, bool]
    failures: tuple[CheckFailure, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures


class CodegenAttempt(StrictModel):
    index: int = Field(ge=1)
    entrypoint: str
    source_hash: str
    stage_reached: Literal["llm", "static", "sandbox", "truncation", "passed"]
    errors: tuple[str, ...] = ()
    provenance: ModelProvenance | None = None


class StrategyGenerationRecord(StrictModel):
    """Generation trail for one strategy package (candidate or its mandatory baseline)."""

    id: NonEmptyStr
    spec_id: NonEmptyStr
    hypothesis_id: NonEmptyStr
    role: Literal["candidate", "baseline"]
    execution_mode: ExecutionMode
    prompt_ref: str = ""
    attempts: tuple[CodegenAttempt, ...]
    ok: bool
    package_id: str | None = None
    package_hash: str | None = None
    checks: dict[str, bool] = Field(default_factory=dict)
    required_comparisons: tuple[str, ...] = ()
    baseline_record_id: str | None = None
    failure_reason: str | None = None
    created_by: Actor
    created_at: UtcDatetime

    @model_validator(mode="after")
    def _consistent(self) -> StrategyGenerationRecord:
        if self.ok and (self.package_id is None or self.failure_reason):
            raise ValueError("a successful generation needs a package and no failure_reason")
        if not self.ok and not self.failure_reason:
            raise ValueError("a failed generation must state failure_reason")
        return self
