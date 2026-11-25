"""Institution contracts: tasks, decisions, reflections, improvement proposals, budgets, costs."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from enigma.schemas.common import Actor, AgentRole, NonEmptyStr, StrictModel, UtcDatetime


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(StrictModel):
    id: NonEmptyStr
    kind: NonEmptyStr = Field(description="e.g. intelligence.research, data.acquire")
    assigned_to: AgentRole
    created_by: Actor
    correlation_id: NonEmptyStr = Field(description="Binds work for one research cycle")
    parent_task_id: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 100
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    attempts: int = 0
    max_attempts: int = Field(default=3, ge=1)
    created_at: UtcDatetime
    updated_at: UtcDatetime


class ExecutionMode(StrEnum):
    """How a decision was produced; shown on the dashboard so fallbacks are never mistaken for
    model reasoning."""

    LLM = "llm"
    DETERMINISTIC = "deterministic"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"


class ModelProvenance(StrictModel):
    provider: NonEmptyStr
    model_id: NonEmptyStr
    model_version: NonEmptyStr
    request_hash: NonEmptyStr
    raw_response_hash: NonEmptyStr
    prompt_ref: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    cached: bool = False
    structured_mode: str | None = None
    repaired: bool = False
    finish_reason: str | None = None


class AgentDecision(StrictModel):
    """Structured decision record with a reasoning summary (never hidden chain-of-thought)."""

    id: NonEmptyStr
    agent: Actor
    task_id: str | None
    decision_type: NonEmptyStr
    outcome: NonEmptyStr
    reason_code: NonEmptyStr
    reasoning_summary: NonEmptyStr
    input_refs: tuple[str, ...] = ()
    output_refs: tuple[str, ...] = ()
    instructions_version: NonEmptyStr
    created_at: UtcDatetime
    execution_mode: ExecutionMode = ExecutionMode.DETERMINISTIC
    fallback_reason: str | None = None
    model_calls: tuple[ModelProvenance, ...] = ()


class ReflectionOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    ABORTED = "aborted"


class Reflection(StrictModel):
    """Post-task reflection (LEARNING_ARCHITECTURE.md §8)."""

    id: NonEmptyStr
    task_id: NonEmptyStr
    agent: AgentRole
    outcome: ReflectionOutcome
    what_worked: str
    what_failed: str
    suspected_systemic_issue: str | None = None
    proposed_followup: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: UtcDatetime


class ProposalStatus(StrEnum):
    PROPOSED = "proposed"
    BENCHMARKING = "benchmarking"
    APPROVED = "approved"
    REJECTED = "rejected"
    ROLLED_OUT = "rolled_out"
    REVERTED = "reverted"


class ProposalTarget(StrEnum):
    PROMPT = "prompt"
    METHODOLOGY = "methodology"
    TOOL = "tool"
    WORKFLOW = "workflow"
    SOFTWARE = "software"
    RESEARCH_DIRECTION = "research_direction"
    TECHNOLOGY = "technology"
    # Changes to these require human approval and can never be auto-adopted.
    VALIDATION_CRITERIA = "validation_criteria"
    RISK_LIMITS = "risk_limits"
    SECURITY_PERMISSIONS = "security_permissions"


PROTECTED_TARGETS: frozenset[ProposalTarget] = frozenset(
    {
        ProposalTarget.VALIDATION_CRITERIA,
        ProposalTarget.RISK_LIMITS,
        ProposalTarget.SECURITY_PERMISSIONS,
    }
)


class ImprovementProposal(StrictModel):
    id: NonEmptyStr
    target: ProposalTarget
    target_ref: NonEmptyStr = Field(description="Agent role, methodology id, module path, ...")
    change_summary: NonEmptyStr
    rationale: NonEmptyStr
    proposed_change: dict[str, Any] = Field(default_factory=dict)
    success_metric: NonEmptyStr
    baseline_metric_value: float | None = None
    evidence_ids: tuple[str, ...] = ()
    proposed_by: Actor
    status: ProposalStatus = ProposalStatus.PROPOSED
    requires_human_approval: bool = False
    created_at: UtcDatetime


class Budget(StrictModel):
    """Hard ceilings per agent session or task. Zero means 'none allowed'."""

    max_usd: float = Field(default=0.0, ge=0)
    max_tokens: int = Field(default=0, ge=0)
    max_http_requests: int = Field(default=200, ge=0)
    max_wall_seconds: float = Field(default=900.0, gt=0)
    max_cpu_seconds: float = Field(default=600.0, gt=0)


class CostRecord(StrictModel):
    id: NonEmptyStr
    actor: Actor
    task_id: str | None
    category: NonEmptyStr = Field(description="llm | search | http | compute")
    provider: NonEmptyStr
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    http_requests: int = 0
    cpu_seconds: float = 0.0
    cached: bool = False
    created_at: UtcDatetime
