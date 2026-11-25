"""Read models for the dashboard and JSON API. Everything shown is read from the stores."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

from enigma.agents.contracts import ContractRegistry
from enigma.comm.bus import MessageBus
from enigma.core.config import Settings
from enigma.core.errors import IntegrityError
from enigma.fsm import (
    ALL_MACHINES,
    IMPROVEMENT_PROPOSAL,
    RESEARCH_CYCLE,
    RESEARCH_CYCLE_STEPS,
    STRATEGY_LIFECYCLE,
    StateMachineStore,
)
from enigma.schemas.cycle import MarketAnalysis, NextExperimentProposal, RiskReview
from enigma.schemas.governance import AgentDecision, ImprovementProposal, Reflection
from enigma.schemas.knowledge import EvidenceItem, FailureCondition, KnowledgeEdge, Lesson, Source
from enigma.schemas.research import (
    DatasetVersion,
    Experiment,
    Hypothesis,
    ResearchObjective,
    StrategyPackage,
)
from enigma.schemas.validation import RedactedValidationSummary, ValidationReport
from enigma.store import Stores

HEARTBEAT_PREFIX = "heartbeat:"


class ReadModels:
    def __init__(self, stores: Stores) -> None:
        self.s = stores
        self.fsm = StateMachineStore(stores.db, stores.audit)
        self.bus = MessageBus(stores.db, stores.audit)

    def health(self) -> dict[str, Any]:
        try:
            audit_entries = self.s.audit.verify()
            audit_ok = True
        except IntegrityError:
            audit_entries, audit_ok = -1, False
        beats = self.s.db.query(
            "SELECT name, value, updated_at FROM flags WHERE name LIKE ? ORDER BY updated_at DESC",
            (HEARTBEAT_PREFIX + "%",),
        )
        return {
            "kill_switch": self.s.flags.kill_switch_active(),
            "audit_chain_ok": audit_ok,
            "audit_entries": audit_entries,
            "schema_version": self.s.db.schema_version,
            "workers": [
                {
                    "worker": r["name"].removeprefix(HEARTBEAT_PREFIX),
                    "status": r["value"],
                    "last_seen": r["updated_at"],
                }
                for r in beats
            ],
            "tasks": self.s.queue.counts(),
        }

    def cycles(self) -> list[dict[str, Any]]:
        out = []
        for c in self.s.checkpoints.list("research_cycle"):
            state = c["state"]
            objective = self.s.documents.get(ResearchObjective, state.get("objective_id", ""))
            step = self.fsm.state(RESEARCH_CYCLE, c["workflow_id"]) or c["step"]
            done = (
                (RESEARCH_CYCLE_STEPS.index(step) + 1)
                if step in RESEARCH_CYCLE_STEPS
                else (len(RESEARCH_CYCLE_STEPS) + 1 if step == "COMPLETED" else 0)
            )
            out.append(
                {
                    "id": c["workflow_id"],
                    "step": step,
                    "status": c["status"],
                    "progress": done,
                    "total": len(RESEARCH_CYCLE_STEPS) + 1,
                    "objective": objective.statement if objective else "",
                    "validation": state.get("validation_overall"),
                    "abort_reason": state.get("abort_reason"),
                    "updated_at": c["updated_at"],
                    "hypothesis_id": state.get("hypothesis_id"),
                }
            )
        return out

    def cycle_detail(self, cycle_id: str) -> dict[str, Any]:
        loaded = self.s.checkpoints.load(cycle_id)
        if loaded is None:
            raise KeyError(cycle_id)
        _, state, status = loaded
        events = self.fsm.history(RESEARCH_CYCLE, cycle_id)
        tasks = self.s.queue.list(correlation_id=cycle_id)
        messages = self.bus.conversation(cycle_id)
        hyp = self.s.documents.get(Hypothesis, state.get("hypothesis_id", ""))
        summary = self.s.documents.get(
            RedactedValidationSummary, state.get("validation_summary_id", "")
        )
        report = self.s.documents.get(ValidationReport, summary.report_id) if summary else None
        experiments = [
            e
            for e in (self.s.documents.get(Experiment, x) for x in state.get("experiment_ids", []))
            if e
        ]
        lessons = [
            x for x in (self.s.documents.get(Lesson, i) for i in state.get("lesson_ids", [])) if x
        ]
        proposal = self.s.documents.get(NextExperimentProposal, state.get("next_proposal_id", ""))
        return {
            "id": cycle_id,
            "status": status,
            "state": state,
            "events": [asdict(e) for e in events],
            "tasks": [t.model_dump(mode="json") for t in sorted(tasks, key=lambda t: t.created_at)],
            "messages": [m.model_dump(mode="json") for m in messages],
            "hypothesis": hyp.model_dump(mode="json") if hyp else None,
            "experiments": [e.model_dump(mode="json") for e in experiments],
            "validation": report.model_dump(mode="json") if report else None,
            "lessons": [x.model_dump(mode="json") for x in lessons],
            "next_proposal": proposal.model_dump(mode="json") if proposal else None,
            "sources": len(state.get("source_ids", [])),
            "evidence": len(state.get("evidence_ids", [])),
        }

    def agents(self) -> list[dict[str, Any]]:
        contracts = ContractRegistry.load(Settings.from_env().registries_dir / "agents")
        running = {
            t.assigned_to.value: t
            for t in self.s.queue.list(limit=1000)
            if t.status.value == "running"
        }
        refl = self.s.documents.list(Reflection, limit=2000)
        per_agent: dict[str, Counter[str]] = {}
        for r in refl:
            per_agent.setdefault(r.agent.value, Counter())[r.outcome.value] += 1
        costs = self.s.costs.by_actor()
        out = []
        for c in contracts.all():
            counts = per_agent.get(c.role.value, Counter())
            total = sum(counts.values())
            current = running.get(c.role.value)
            out.append(
                {
                    "number": c.number,
                    "role": c.role.value,
                    "name": c.name,
                    "plane": c.plane.value,
                    "mission": c.mission,
                    "tools": list(c.tools),
                    "task_kinds": list(c.task_kinds),
                    "deterministic": c.deterministic_core,
                    "instructions_version": c.instructions_version[:19],
                    "current_task": current.kind if current else None,
                    "tasks_done": total,
                    "success_rate": (counts["success"] / total) if total else None,
                    "costs": costs.get(c.role.value, {}),
                }
            )
        return out

    def knowledge(self) -> dict[str, Any]:
        sources = self.s.documents.list(Source, limit=5000)
        evidence_n = self.s.documents.count(EvidenceItem)
        return {
            "sources": len(sources),
            "by_provider": dict(Counter(s.provider for s in sources)),
            "by_kind": dict(Counter(s.kind.value for s in sources)),
            "evidence": evidence_n,
            "edges": self.s.documents.count(KnowledgeEdge),
            "lessons": self.s.documents.count(Lesson),
            "failures": self.s.documents.count(FailureCondition),
            "hypotheses": self.s.documents.count(Hypothesis),
            "analyses": self.s.documents.count(MarketAnalysis),
            "datasets": [
                d.model_dump(mode="json", exclude={"provenance", "quality"})
                for d in self.s.documents.list(DatasetVersion, limit=50)
            ],
            "recent_sources": [
                s.model_dump(
                    mode="json",
                    include={"id", "title", "url", "provider", "published_at", "retrieved_at"},
                )
                for s in sources[:40]
            ],
            "recent_evidence": [
                e.model_dump(
                    mode="json", include={"id", "source_id", "excerpt", "extraction_method"}
                )
                for e in self.s.documents.list(EvidenceItem, limit=20)
            ],
        }

    def strategies(self) -> dict[str, Any]:
        packages = self.s.documents.list(StrategyPackage, limit=500)
        return {
            "lifecycle": self.fsm.state_counts(STRATEGY_LIFECYCLE),
            "packages": [
                {
                    "id": p.id,
                    "name": p.name,
                    "params": p.params,
                    "state": self.fsm.state(STRATEGY_LIFECYCLE, p.id),
                }
                for p in packages
            ],
            "experiments": [
                e.model_dump(
                    mode="json",
                    include={
                        "id",
                        "hypothesis_id",
                        "family_id",
                        "trial_index",
                        "status",
                        "metrics",
                        "result_hash",
                    },
                )
                for e in self.s.documents.list(Experiment, limit=500)
            ],
            "hypotheses": [
                h.model_dump(
                    mode="json",
                    include={
                        "id",
                        "title",
                        "family_id",
                        "formal_statement",
                        "trial_budget",
                        "status",
                        "created_at",
                    },
                )
                for h in self.s.documents.list(Hypothesis, limit=100)
            ],
            "validations": [
                r.model_dump(mode="json")
                for r in self.s.documents.list(ValidationReport, limit=100)
            ],
            "risk_reviews": [
                r.model_dump(mode="json") for r in self.s.documents.list(RiskReview, limit=100)
            ],
        }

    def learning(self) -> dict[str, Any]:
        return {
            "lessons": [
                x.model_dump(mode="json") for x in self.s.documents.list(Lesson, limit=200)
            ],
            "next_experiments": [
                x.model_dump(mode="json")
                for x in self.s.documents.list(NextExperimentProposal, limit=100)
            ],
            "improvements": [
                {
                    **x.model_dump(mode="json"),
                    "fsm_state": self.fsm.state(IMPROVEMENT_PROPOSAL, x.id),
                }
                for x in self.s.documents.list(ImprovementProposal, limit=200)
            ],
            "reflections": [
                x.model_dump(mode="json") for x in self.s.documents.list(Reflection, limit=100)
            ],
        }

    def decisions(self, limit: int = 100) -> list[dict[str, Any]]:
        return [
            d.model_dump(mode="json") for d in self.s.documents.list(AgentDecision, limit=limit)
        ]

    def messages(self, limit: int = 150) -> list[dict[str, Any]]:
        return [m.model_dump(mode="json") for m in self.bus.recent(limit)]

    def costs(self) -> dict[str, Any]:
        return {"totals": self.s.costs.totals(), "by_actor": self.s.costs.by_actor()}

    def audit(self, limit: int = 200) -> list[dict[str, Any]]:
        return [asdict(e) for e in self.s.audit.entries(limit=limit)]

    def machines(self) -> dict[str, dict[str, int]]:
        return {m.name: self.fsm.state_counts(m) for m in ALL_MACHINES}
