from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from enigma.agents import (
    Agent,
    AgentContext,
    AgentWorker,
    BudgetTracker,
    ContractRegistry,
    DecisionRecord,
    TaskOutcome,
    TaskRequest,
    Tool,
    ToolRegistry,
    delegate,
)
from enigma.comm import MessageBus
from enigma.core import BudgetExceededError, PermissionDeniedError
from enigma.schemas.common import AgentRole, Plane, SystemActor
from enigma.schemas.governance import AgentDecision, Budget, Reflection, TaskStatus
from enigma.schemas.messages import MessageType
from enigma.store import Stores

REGISTRY_DIR = Path(__file__).resolve().parents[2] / "registries" / "agents"


@pytest.fixture
def contracts() -> ContractRegistry:
    return ContractRegistry.load(REGISTRY_DIR)


@pytest.fixture
def bus(stores: Stores) -> MessageBus:
    return MessageBus(stores.db, stores.audit)


class RecordingIntelligence(Agent):
    """Test double that exercises the runtime; not an operational agent."""

    role = AgentRole.INTELLIGENCE
    calls: ClassVar[list[str]] = []

    def execute(self, ctx: AgentContext) -> TaskOutcome:
        self.calls.append(ctx.task.id)
        result = ctx.tools.call("intel.research", topic=ctx.task.inputs["topic"])
        ctx.budget.charge("http", "test", http_requests=3)
        return TaskOutcome(
            outputs={"n": result},
            decisions=[DecisionRecord("search_plan", "executed", "ok", "Ran one query.")],
            what_worked="query returned results",
        )


class ForbiddenToolAgent(Agent):
    role = AgentRole.INTELLIGENCE

    def execute(self, ctx: AgentContext) -> TaskOutcome:
        ctx.tools.call("validation.run")
        return TaskOutcome(outputs={})


class FlakyAgent(Agent):
    role = AgentRole.INTELLIGENCE

    def execute(self, ctx: AgentContext) -> TaskOutcome:
        raise RuntimeError("transient upstream error")


def _topic_length(topic: str) -> int:
    return len(topic)


def _tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool("intel.research", "test", Plane.RESEARCH, _topic_length))
    reg.register(Tool("validation.run", "test", Plane.VALIDATION, lambda: "PASS"))
    return reg


def _request(topic: str = "funding") -> TaskRequest:
    return TaskRequest("intelligence.research", AgentRole.INTELLIGENCE, {"topic": topic})


def test_all_twelve_contracts_load_with_unique_task_owners(contracts: ContractRegistry) -> None:
    assert [c.number for c in contracts.all()] == list(range(1, 13))
    kinds = [k for c in contracts.all() for k in c.task_kinds]
    assert len(kinds) == len(set(kinds))
    assert contracts.role_for_task("validation.validate") is AgentRole.INDEPENDENT_VALIDATION
    val = contracts.get(AgentRole.INDEPENDENT_VALIDATION)
    assert val.deterministic_core and Plane.RESEARCH not in val.readable_planes
    for role in (AgentRole.QUANT_RESEARCH, AgentRole.INTELLIGENCE, AgentRole.DATA_ENGINEERING):
        assert Plane.VALIDATION not in contracts.get(role).readable_planes
    assert contracts.get(AgentRole.CIO).instructions_version.startswith("sha256:")


def test_delegation_is_permissioned_and_idempotent(
    stores: Stores, bus: MessageBus, contracts: ContractRegistry
) -> None:
    cmap = {c.role: c for c in contracts.all()}
    t1 = delegate(stores.queue, bus, cmap, AgentRole.CIO, _request(), "cyc_1")
    t2 = delegate(stores.queue, bus, cmap, AgentRole.CIO, _request(), "cyc_1")
    assert t1.id == t2.id
    assert len(stores.queue.list()) == 1
    assigned = bus.poll(AgentRole.INTELLIGENCE)
    assert len(assigned) == 1 and assigned[0].message_type is MessageType.TASK_ASSIGNED
    with pytest.raises(PermissionDeniedError):
        delegate(
            stores.queue,
            bus,
            cmap,
            AgentRole.INTELLIGENCE,
            TaskRequest("data.acquire", AgentRole.DATA_ENGINEERING, {}),
            "cyc_1",
        )
    with pytest.raises(PermissionDeniedError):
        delegate(
            stores.queue,
            bus,
            cmap,
            AgentRole.CIO,
            TaskRequest("validation.validate", AgentRole.INTELLIGENCE, {}),
            "cyc_1",
        )


def test_worker_executes_records_decision_reflection_and_reply(
    stores: Stores, bus: MessageBus, contracts: ContractRegistry
) -> None:
    cmap = {c.role: c for c in contracts.all()}
    task = delegate(stores.queue, bus, cmap, AgentRole.CIO, _request("basis"), "cyc_1")
    worker = AgentWorker(
        stores, bus, contracts, _tools(), {AgentRole.INTELLIGENCE: RecordingIntelligence()}
    )
    done = worker.run_once()
    assert done is not None and done.status is TaskStatus.SUCCEEDED
    assert done.outputs == {"n": 5}
    assert worker.run_once() is None
    decisions = stores.documents.list(AgentDecision, where={"task_id": task.id})
    assert decisions[0].instructions_version == cmap[AgentRole.INTELLIGENCE].instructions_version
    reflections = stores.documents.list(Reflection, where={"task_id": task.id})
    assert reflections[0].outcome.value == "success"
    replies = bus.poll(AgentRole.CIO)
    assert [m.message_type for m in replies] == [MessageType.TASK_COMPLETED]
    assert stores.costs.totals(task_id=task.id)["http_requests"] == 3
    stores.audit.verify()


def test_tool_outside_allowlist_is_denied_and_not_retried(
    stores: Stores, bus: MessageBus, contracts: ContractRegistry
) -> None:
    cmap = {c.role: c for c in contracts.all()}
    task = delegate(stores.queue, bus, cmap, AgentRole.CIO, _request(), "cyc_2")
    worker = AgentWorker(
        stores, bus, contracts, _tools(), {AgentRole.INTELLIGENCE: ForbiddenToolAgent()}
    )
    done = worker.run_once()
    assert done is not None and done.status is TaskStatus.FAILED
    actions = [e.action for e in stores.audit.entries(subject_id=task.id)]
    assert "tool.denied" in actions and "task.incident" in actions
    assert [m.message_type for m in bus.poll(AgentRole.CIO)] == [MessageType.TASK_FAILED]


def test_transient_failures_retry_then_fail(
    stores: Stores, bus: MessageBus, contracts: ContractRegistry
) -> None:
    cmap = {c.role: c for c in contracts.all()}
    delegate(stores.queue, bus, cmap, AgentRole.CIO, _request(), "cyc_3")
    worker = AgentWorker(stores, bus, contracts, _tools(), {AgentRole.INTELLIGENCE: FlakyAgent()})
    statuses = [worker.run_once() for _ in range(3)]
    assert [s.status for s in statuses if s] == [
        TaskStatus.PENDING,
        TaskStatus.PENDING,
        TaskStatus.FAILED,
    ]


def test_kill_switch_stops_claiming(
    stores: Stores, bus: MessageBus, contracts: ContractRegistry
) -> None:
    cmap = {c.role: c for c in contracts.all()}
    delegate(stores.queue, bus, cmap, AgentRole.CIO, _request(), "cyc_4")
    stores.flags.set("kill_switch", "on", SystemActor.HUMAN.value)
    worker = AgentWorker(
        stores, bus, contracts, _tools(), {AgentRole.INTELLIGENCE: RecordingIntelligence()}
    )
    assert worker.run_once() is None
    assert stores.queue.counts() == {"pending": 1}


def test_budget_tracker_refuses_and_records(stores: Stores) -> None:
    tracker = BudgetTracker(
        Budget(max_usd=1.0, max_tokens=100, max_http_requests=2),
        stores.costs,
        AgentRole.QUANT_RESEARCH,
        "tsk_x",
    )
    tracker.authorize(usd=0.5, tokens=50)
    with pytest.raises(BudgetExceededError):
        tracker.authorize(usd=2.0)
    tracker.charge("http", "p", http_requests=2)
    with pytest.raises(BudgetExceededError):
        tracker.charge("http", "p", http_requests=1)
    assert stores.costs.totals(task_id="tsk_x")["http_requests"] == 3


def test_bus_idempotency_and_conversation(stores: Stores, bus: MessageBus) -> None:
    for _ in range(3):
        bus.publish(
            AgentRole.CIO,
            AgentRole.QUANT_RESEARCH,
            MessageType.TASK_ASSIGNED,
            {"x": 1},
            "cyc_9",
            idempotency_key="k1",
        )
    bus.publish(
        AgentRole.QUANT_RESEARCH, AgentRole.CIO, MessageType.TASK_COMPLETED, {"x": 2}, "cyc_9"
    )
    assert len(bus.conversation("cyc_9")) == 2
    pending = bus.poll(AgentRole.QUANT_RESEARCH)
    bus.ack(pending[0].message_id)
    assert bus.poll(AgentRole.QUANT_RESEARCH) == []
