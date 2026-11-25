"""Tool bindings: the only place agent tool names are connected to implementations.

Core tools (lifecycle, knowledge index/retrieval, tasks, improvement proposals, risk) are bound
to the stores here. Service tools (research engine, data catalog, sandbox + backtest, validation,
model gateway) are supplied by the composition root as `Services`, which keeps each plane's
implementation behind its own object and lets tests substitute labelled test doubles.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from enigma.agents.tools import Tool, ToolRegistry
from enigma.core.clock import Clock
from enigma.core.errors import (
    InvalidTransitionError,
    PermissionDeniedError,
    ProviderUnavailableError,
)
from enigma.execution.paper import PaperBroker
from enigma.fsm import IMPROVEMENT_PROPOSAL, STRATEGY_LIFECYCLE, StateMachineStore
from enigma.risk.service import RiskService
from enigma.schemas.common import Plane
from enigma.schemas.governance import ImprovementProposal
from enigma.schemas.knowledge import EvidenceItem, FailureCondition, Lesson
from enigma.schemas.research import Hypothesis
from enigma.store import Stores
from enigma.store.documents import kind_of

ToolFn = Callable[..., Any]


@dataclass(frozen=True)
class Services:
    intel_research: ToolFn
    data_acquire_research: ToolFn
    data_load_research: ToolFn
    strategy_build: ToolFn
    backtest_run: ToolFn
    validation_run: ToolFn
    gateway_status: ToolFn
    research_cutoff: Callable[[Sequence[str]], datetime]
    llm_structured: ToolFn | None = None
    quant_hypothesis_generate: ToolFn | None = None
    strategy_codegen: ToolFn | None = None
    gateway_benchmark: ToolFn | None = None


_RETRIEVE_KINDS = {
    "lesson_ids": "lesson",
    "failure_ids": "failure_condition",
    "hypothesis_ids": "hypothesis",
    "evidence_ids": "evidence",
}


def _index_body(doc: BaseModel) -> tuple[str, str, tuple[str, ...]]:
    if isinstance(doc, Lesson):
        return doc.statement[:120], doc.statement, tuple(doc.topics)
    if isinstance(doc, Hypothesis):
        body = f"{doc.mechanism}\n{doc.prediction}\n{doc.formal_statement}"
        return doc.title, body, tuple(doc.features)
    if isinstance(doc, FailureCondition):
        return doc.description[:120], doc.description, ()
    if isinstance(doc, EvidenceItem):
        return doc.claim[:120], f"{doc.claim}\n{doc.excerpt}", tuple(doc.topics)
    raise TypeError(f"{type(doc).__name__} is not indexable")


class CoreTools:
    def __init__(self, stores: Stores, fsm: StateMachineStore, risk: RiskService, clock: Clock):
        self.stores = stores
        self.fsm = fsm
        self.risk = risk
        self.clock = clock

    def knowledge_retrieve(
        self, query: str, topics: list[str] | tuple[str, ...] = ()
    ) -> dict[str, list[str]]:
        text = " ".join([query, *topics])
        hits = self.stores.documents.search_text(
            text, [Plane.CONTROL, Plane.RESEARCH], kinds=list(_RETRIEVE_KINDS.values()), limit=200
        )
        out: dict[str, list[str]] = {k: [] for k in _RETRIEVE_KINDS}
        reverse = {v: k for k, v in _RETRIEVE_KINDS.items()}
        for doc_id, kind, _ in hits:
            out[reverse[kind]].append(doc_id)
        return out

    def knowledge_index(self, doc: BaseModel) -> str:
        title, body, topics = _index_body(doc)
        kind, plane = kind_of(type(doc))
        doc_id = str(doc.id)  # type: ignore[attr-defined]
        self.stores.documents.index_text(doc_id, kind, plane, title, body, topics)
        return doc_id

    def lifecycle_transition(
        self,
        subject_id: str,
        target: str,
        reason: str,
        actor: str,
        artifacts: dict[str, str] | None = None,
    ) -> str:
        current = self.fsm.state(STRATEGY_LIFECYCLE, subject_id)
        if current == target:
            return target
        if current is None:
            if target != STRATEGY_LIFECYCLE.initial:
                raise InvalidTransitionError(f"{subject_id} has no lifecycle; create it first")
            self.fsm.create(STRATEGY_LIFECYCLE, subject_id, actor, reason_code=reason)
            return target
        self.fsm.transition(STRATEGY_LIFECYCLE, subject_id, target, actor, reason, artifacts)
        return target

    def tasks_for_cycle(self, cycle_id: str) -> list[dict[str, Any]]:
        return [
            {
                "id": t.id,
                "kind": t.kind,
                "status": t.status.value,
                "assigned_to": str(t.assigned_to),
            }
            for t in self.stores.queue.list(correlation_id=cycle_id, limit=1000)
        ]

    def improvement_propose(self, proposal: ImprovementProposal, actor: str) -> str:
        if str(proposal.proposed_by) != actor:
            raise PermissionDeniedError("proposals must be filed by the proposing role")
        if self.stores.documents.get(ImprovementProposal, proposal.id) is None:
            self.stores.documents.put(proposal, recorded_by=actor)
        if self.fsm.state(IMPROVEMENT_PROPOSAL, proposal.id) is None:
            self.fsm.create(IMPROVEMENT_PROPOSAL, proposal.id, actor, reason_code="proposed")
        return proposal.id

    def risk_evaluate(self, package_id: str, summary_id: str) -> dict[str, Any]:
        return self.risk.evaluate(package_id, summary_id)

    def paper_open_session(
        self,
        package_id: str,
        risk_review_id: str,
        acknowledge_id: str,
        notional_usd: float = 10_000.0,
    ) -> dict[str, Any]:
        session = PaperBroker(self.stores, self.clock).open_session(
            package_id=package_id,
            risk_review_id=risk_review_id,
            acknowledge_id=acknowledge_id,
            notional_usd=notional_usd,
        )
        return {
            "session_id": session.id,
            "status": session.status.value,
            "equity_usd": session.equity_usd,
        }


def _no_model_gateway(**_: Any) -> Any:
    raise ProviderUnavailableError("no model gateway is configured")


def build_registry(core: CoreTools, services: Services) -> ToolRegistry:
    reg = ToolRegistry()
    r, c, v = Plane.RESEARCH, Plane.CONTROL, Plane.VALIDATION
    for tool in (
        Tool(
            "knowledge.retrieve",
            "BM25 retrieval of prior lessons/failures/evidence",
            c,
            core.knowledge_retrieve,
        ),
        Tool(
            "knowledge.index",
            "Index a lesson/hypothesis/evidence for retrieval",
            c,
            core.knowledge_index,
        ),
        Tool(
            "lifecycle.transition",
            "Strategy lifecycle transition as the calling role",
            r,
            core.lifecycle_transition,
            pass_actor=True,
        ),
        Tool("tasks.for_cycle", "Tasks of one research cycle", c, core.tasks_for_cycle),
        Tool(
            "improvement.propose",
            "File an approval-gated improvement proposal",
            c,
            core.improvement_propose,
            pass_actor=True,
        ),
        Tool(
            "risk.evaluate", "Risk service decision for a validated package", c, core.risk_evaluate
        ),
        Tool(
            "execution.paper",
            "Open a paper session after risk CLEARANCE + human acknowledge",
            Plane.EXECUTION,
            core.paper_open_session,
        ),
        Tool("intel.research", "Recursive external research", r, services.intel_research),
        Tool(
            "data.acquire_research",
            "Acquire a PIT-checked research dataset",
            r,
            services.data_acquire_research,
        ),
        Tool(
            "data.load_research", "Load a verified research dataset", r, services.data_load_research
        ),
        Tool(
            "strategy.build",
            "Statically check and package strategy code",
            r,
            services.strategy_build,
        ),
        Tool(
            "backtest.run",
            "Sandboxed signal run + deterministic bar backtest",
            r,
            services.backtest_run,
        ),
        Tool("validation.run", "Independent validation protocol run", v, services.validation_run),
        Tool("gateway.status", "Model/search provider availability", c, services.gateway_status),
    ):
        reg.register(tool)
    reg.register(
        Tool(
            "llm.structured",
            "Typed model call via the gateway",
            c,
            services.llm_structured or _no_model_gateway,
            pass_actor=True,
        )
    )
    if services.quant_hypothesis_generate is not None:
        reg.register(
            Tool(
                "quant.hypothesis_generate",
                "LLM HypothesisGenerator over a sealed ResearchContext",
                r,
                services.quant_hypothesis_generate,
                pass_actor=True,
            )
        )
    if services.strategy_codegen is not None:
        reg.register(
            Tool(
                "strategy.codegen",
                "LLM StrategyCodeGenerator from a cycle Hypothesis",
                r,
                services.strategy_codegen,
                pass_actor=True,
            )
        )
    if services.gateway_benchmark is not None:
        reg.register(
            Tool("gateway.benchmark", "Benchmark available models", c, services.gateway_benchmark)
        )
    return reg
