"""Agent worker: claims tasks, enforces contracts and budgets, records decisions + reflections."""

from __future__ import annotations

import os
import socket
import traceback
from datetime import timedelta

from enigma.agents.base import Agent, AgentContext, TaskOutcome, delegate
from enigma.agents.budget import BudgetTracker
from enigma.agents.contracts import ContractRegistry
from enigma.agents.tools import ToolGateway, ToolRegistry
from enigma.comm.bus import MessageBus
from enigma.core.clock import Clock, SystemClock
from enigma.core.errors import (
    BudgetExceededError,
    ConstitutionViolationError,
    ProviderUnavailableError,
)
from enigma.core.hashing import sha256_hex
from enigma.core.logging import get_logger
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import AgentDecision, Task
from enigma.schemas.messages import MessageType
from enigma.store import PlaneView, Stores

log = get_logger(__name__)

# Errors that indicate governance problems: never retried, always audited as incidents.
_NON_RETRYABLE = (ConstitutionViolationError, BudgetExceededError, ProviderUnavailableError)


class AgentWorker:
    def __init__(
        self,
        stores: Stores,
        bus: MessageBus,
        contracts: ContractRegistry,
        tools: ToolRegistry,
        agents: dict[AgentRole, Agent],
        roles: tuple[AgentRole, ...] | None = None,
        clock: Clock | None = None,
        worker_id: str | None = None,
        lease: timedelta = timedelta(minutes=15),
    ) -> None:
        self.stores = stores
        self.bus = bus
        self.contracts = contracts
        self.tools = tools
        self.agents = agents
        self.roles = roles or tuple(agents)
        self.clock = clock or SystemClock()
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self.lease = lease

    def run_once(self) -> Task | None:
        """Claim and execute at most one task. Returns the finished task, or None if idle."""
        if self.stores.flags.kill_switch_active():
            return None
        task = self.stores.queue.claim(self.worker_id, roles=self.roles, lease=self.lease)
        if task is None:
            return None
        return self._execute(task)

    def _context(self, task: Task) -> AgentContext:
        contract = self.contracts.get(task.assigned_to)
        readable = frozenset(contract.readable_planes)
        view = PlaneView(
            self.stores.documents,
            readable,
            frozenset(contract.writable_planes),
            actor=task.assigned_to.value,
        )
        gateway = ToolGateway(
            self.tools,
            contract.tools,
            readable | frozenset(contract.writable_planes),
            task.assigned_to.value,
            self.stores.audit,
            task.id,
        )
        budget = BudgetTracker(contract.budget, self.stores.costs, task.assigned_to, task.id)
        return AgentContext(
            role=task.assigned_to,
            contract=contract,
            task=task,
            view=view,
            tools=gateway,
            budget=budget,
            bus=self.bus,
            clock=self.clock,
        )

    def _execute(self, task: Task) -> Task:
        agent = self.agents.get(task.assigned_to)
        ctx = self._context(task)
        audit = self.stores.audit
        if agent is None or task.kind not in ctx.contract.task_kinds:
            audit.record(task.assigned_to.value, "task.rejected", task.id, {"kind": task.kind})
            return self.stores.queue.fail(task.id, f"no agent executes {task.kind}", retry=False)
        audit.record(
            task.assigned_to.value,
            "task.started",
            task.id,
            {"kind": task.kind, "attempt": task.attempts, "worker": self.worker_id},
        )
        outcome: TaskOutcome | None = None
        error: str | None = None
        retry = True
        try:
            outcome = agent.execute(ctx)
            ctx.budget.check()
        except _NON_RETRYABLE as exc:
            error, retry = f"{type(exc).__name__}: {exc}", False
            audit.record(task.assigned_to.value, "task.incident", task.id, {"error": error})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.warning(
                "task_failed", task_id=task.id, error=error, trace=traceback.format_exc(limit=5)
            )

        self._record_decisions(ctx, outcome, error)
        self.stores.documents.put(agent.reflect(ctx, outcome, error), recorded_by=self.worker_id)

        if error is None and outcome is not None:
            for request in outcome.delegate:
                delegate(
                    self.stores.queue,
                    self.bus,
                    {c.role: c for c in self.contracts.all()},
                    task.assigned_to,
                    request,
                    task.correlation_id,
                    parent_task_id=task.id,
                )
            done = self.stores.queue.complete(task.id, outcome.outputs)
            self.bus.publish(
                task.assigned_to,
                task.created_by,
                MessageType.TASK_COMPLETED,
                {"task_id": task.id, "kind": task.kind, "outputs": outcome.outputs},
                correlation_id=task.correlation_id,
                causation_id=task.id,
                idempotency_key=f"{task.id}:completed",
            )
            audit.record(task.assigned_to.value, "task.completed", task.id, {"kind": task.kind})
            return done

        failed = self.stores.queue.fail(task.id, error or "unknown error", retry=retry)
        if failed.status.value == "failed":
            self.bus.publish(
                task.assigned_to,
                task.created_by,
                MessageType.TASK_FAILED,
                {"task_id": task.id, "kind": task.kind, "error": error},
                correlation_id=task.correlation_id,
                causation_id=task.id,
                idempotency_key=f"{task.id}:failed",
            )
        audit.record(
            task.assigned_to.value,
            "task.failed",
            task.id,
            {"error": error, "final": failed.status.value == "failed"},
        )
        return failed

    def _record_decisions(
        self, ctx: AgentContext, outcome: TaskOutcome | None, error: str | None
    ) -> None:
        records = list(outcome.decisions) if outcome else []
        for index, record in enumerate(records):
            self.stores.documents.put(
                AgentDecision(
                    id="dec_"
                    + sha256_hex(f"{ctx.task.id}|{ctx.task.attempts}|{index}")[:26].upper(),
                    agent=ctx.role,
                    task_id=ctx.task.id,
                    decision_type=record.decision_type,
                    outcome=record.outcome,
                    reason_code=record.reason_code,
                    reasoning_summary=record.reasoning_summary,
                    input_refs=record.input_refs,
                    output_refs=record.output_refs,
                    instructions_version=ctx.contract.instructions_version,
                    created_at=self.clock.now(),
                ),
                recorded_by=self.worker_id,
            )
        if error is not None:
            self.stores.documents.put(
                AgentDecision(
                    id="dec_" + sha256_hex(f"{ctx.task.id}|{ctx.task.attempts}|error")[:26].upper(),
                    agent=ctx.role,
                    task_id=ctx.task.id,
                    decision_type="task_execution",
                    outcome="failed",
                    reason_code="exception",
                    reasoning_summary=error[:2000],
                    instructions_version=ctx.contract.instructions_version,
                    created_at=self.clock.now(),
                ),
                recorded_by=self.worker_id,
            )
