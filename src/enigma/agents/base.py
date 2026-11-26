"""Agent base class, execution context, and task delegation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from enigma.agents.budget import BudgetTracker
from enigma.agents.contracts import RoleContract
from enigma.agents.tools import ToolGateway
from enigma.comm.bus import MessageBus
from enigma.core.clock import Clock
from enigma.core.errors import PermissionDeniedError
from enigma.core.hashing import sha256_hex
from enigma.schemas.common import Actor, AgentRole
from enigma.schemas.governance import Reflection, ReflectionOutcome, Task
from enigma.schemas.messages import MessageType
from enigma.store.documents import PlaneView
from enigma.store.queue import TaskQueue


@dataclass(frozen=True)
class DecisionRecord:
    """Structured reasoning summary; persisted as an AgentDecision by the runtime."""

    decision_type: str
    outcome: str
    reason_code: str
    reasoning_summary: str
    input_refs: tuple[str, ...] = ()
    output_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskRequest:
    kind: str
    assigned_to: AgentRole
    inputs: dict[str, Any]
    priority: int = 100
    idempotency_key: str | None = None


@dataclass
class TaskOutcome:
    outputs: dict[str, Any]
    decisions: list[DecisionRecord] = field(default_factory=list)
    delegate: list[TaskRequest] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    what_worked: str = ""
    what_failed: str = ""
    suspected_systemic_issue: str | None = None
    proposed_followup: str | None = None


@dataclass
class AgentContext:
    role: AgentRole
    contract: RoleContract
    task: Task
    view: PlaneView
    tools: ToolGateway
    budget: BudgetTracker
    bus: MessageBus
    clock: Clock


def deterministic_task_id(correlation_id: str, kind: str, key: str = "") -> str:
    return "tsk_" + sha256_hex(f"{correlation_id}|{kind}|{key}")[:26].upper()


def delegate(
    queue: TaskQueue,
    bus: MessageBus,
    contracts: dict[AgentRole, RoleContract],
    from_actor: Actor,
    request: TaskRequest,
    correlation_id: str,
    parent_task_id: str | None = None,
    now: datetime | None = None,
) -> Task:
    """Create a task idempotently and publish TaskAssigned. Re-delegation returns the original."""
    if isinstance(from_actor, AgentRole):
        allowed = contracts[from_actor].may_delegate_to
        if request.assigned_to not in allowed:
            raise PermissionDeniedError(
                f"{from_actor.value} may not delegate to {request.assigned_to.value}"
            )
    target = contracts[request.assigned_to]
    if request.kind not in target.task_kinds:
        raise PermissionDeniedError(
            f"{request.assigned_to.value} does not execute task kind {request.kind}"
        )
    task_id = deterministic_task_id(correlation_id, request.kind, request.idempotency_key or "")
    existing = queue.get(task_id)
    if existing is not None:
        return existing
    now = now or datetime.now(UTC)
    task = Task(
        id=task_id,
        kind=request.kind,
        assigned_to=request.assigned_to,
        created_by=from_actor,
        correlation_id=correlation_id,
        parent_task_id=parent_task_id,
        priority=request.priority,
        inputs=request.inputs,
        created_at=now,
        updated_at=now,
    )
    queue.enqueue(task)
    bus.publish(
        from_actor,
        request.assigned_to,
        MessageType.TASK_ASSIGNED,
        {
            "task_id": task.id,
            "kind": task.kind,
            "budget": target.budget.model_dump(mode="json"),
            "inputs": task.inputs,
        },
        correlation_id=correlation_id,
        causation_id=parent_task_id,
        idempotency_key=f"{task.id}:assigned",
    )
    return task


class Agent(ABC):
    """A logical institutional role. Subclasses implement `execute` for their task kinds."""

    role: AgentRole

    @abstractmethod
    def execute(self, ctx: AgentContext) -> TaskOutcome: ...

    def reflect(
        self, ctx: AgentContext, outcome: TaskOutcome | None, error: str | None
    ) -> Reflection:
        """Default structured post-task reflection; roles may override with richer analysis."""
        status = ReflectionOutcome.SUCCESS if error is None else ReflectionOutcome.FAILURE
        metrics = dict(outcome.metrics) if outcome else {}
        metrics["wall_seconds"] = round(ctx.budget.elapsed, 3)
        metrics["http_requests"] = float(ctx.budget.http_requests)
        metrics["usd"] = ctx.budget.usd
        return Reflection(
            id="ref_" + sha256_hex(f"{ctx.task.id}|{ctx.task.attempts}")[:26].upper(),
            task_id=ctx.task.id,
            agent=self.role,
            outcome=status,
            what_worked=(outcome.what_worked if outcome else "") or "",
            what_failed=(outcome.what_failed if outcome else "") or (error or ""),
            suspected_systemic_issue=outcome.suspected_systemic_issue if outcome else None,
            proposed_followup=outcome.proposed_followup if outcome else None,
            metrics=metrics,
            created_at=ctx.clock.now(),
        )
