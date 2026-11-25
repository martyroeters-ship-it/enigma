"""CIO research-cycle driver: the 13-step M1 workflow as an idempotent, resumable state machine.

The CIO is deterministic policy (docs/AGENTS.md challenge, ADR-004). `tick` inspects persisted
state only, so a crashed process resumes exactly where it stopped: tasks have deterministic ids,
delegation is idempotent, and every step is checkpointed before the next begins.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from enigma.agents.base import TaskRequest, delegate
from enigma.agents.contracts import ContractRegistry
from enigma.comm.bus import MessageBus
from enigma.core.clock import Clock
from enigma.core.errors import InvalidTransitionError
from enigma.core.hashing import sha256_hex, sha256_of
from enigma.core.ids import new_id
from enigma.fsm import RESEARCH_CYCLE, STRATEGY_LIFECYCLE, StateMachineStore
from enigma.learning.abort import lessons_from_aborted_cycle
from enigma.schemas.common import AgentRole, Instrument
from enigma.schemas.governance import Task, TaskStatus
from enigma.schemas.research import ResearchObjective
from enigma.schemas.validation import RedactedValidationSummary, ValidationRequest
from enigma.store import Stores

WORKFLOW = "research_cycle"
CIO = AgentRole.CIO


@dataclass(frozen=True)
class HoldoutPolicy:
    """Research data window start. The holdout boundary itself is owned by the validation plane:
    `ValidationService.seal_holdout` seals it once per symbol set and returns only the research
    cutoff, which the driver receives through `research_cutoff`."""

    data_start: date

    @property
    def research_start(self) -> datetime:
        d = self.data_start
        return datetime(d.year, d.month, d.day, tzinfo=UTC)

    @classmethod
    def load(cls, path: Path) -> HoldoutPolicy:
        raw = yaml.safe_load(path.read_text("utf-8"))
        return cls(data_start=date.fromisoformat(str(raw["data_start"])))


ResearchCutoff = Callable[[Sequence[str]], datetime]


class CycleDriver:
    def __init__(
        self,
        stores: Stores,
        bus: MessageBus,
        fsm: StateMachineStore,
        contracts: ContractRegistry,
        policy: HoldoutPolicy,
        research_cutoff: ResearchCutoff,
        clock: Clock,
    ) -> None:
        self.stores = stores
        self.bus = bus
        self.fsm = fsm
        self.contracts = contracts
        self.cmap = {c.role: c for c in contracts.all()}
        self.policy = policy
        self.research_cutoff = research_cutoff
        self.clock = clock

    # ---------------------------------------------------------------- intake (step 1)
    def submit_objective(
        self,
        statement: str,
        instruments: tuple[Instrument, ...],
        topics: tuple[str, ...] = (),
        max_trials: int = 9,
        proposal_id: str | None = None,
        created_by: AgentRole | str = CIO,
    ) -> str:
        now = self.clock.now()
        objective = ResearchObjective(
            id=new_id("obj"),
            statement=statement,
            instruments=instruments,
            topics=topics,
            max_trials=max_trials,
            created_by=created_by,
            created_at=now,
        )
        self.stores.documents.put(objective, recorded_by=CIO.value)
        symbols = sorted({i.symbol for i in instruments})
        research_end = self.research_cutoff(symbols)
        if research_end <= self.policy.research_start:
            raise ValueError("research cutoff precedes the research data start")
        cycle_id = new_id("cyc")
        self.fsm.create(RESEARCH_CYCLE, cycle_id, CIO.value, reason_code="objective_received")
        state = {
            "objective_id": objective.id,
            "proposal_id": proposal_id,
            "symbols": symbols,
            "research_start": self.policy.research_start.isoformat(),
            "research_end": research_end.isoformat(),
            "started_at": now.isoformat(),
        }
        self.stores.checkpoints.save(cycle_id, WORKFLOW, "OBJECTIVE_RECEIVED", state, "running")
        self.stores.audit.record(
            CIO.value,
            "cycle.objective_received",
            cycle_id,
            {"objective_id": objective.id, "statement": statement},
        )
        return cycle_id

    # ---------------------------------------------------------------- helpers
    def _state(self, cycle_id: str) -> dict[str, Any]:
        loaded = self.stores.checkpoints.load(cycle_id)
        if loaded is None:
            raise KeyError(cycle_id)
        return loaded[1]

    def _save(self, cycle_id: str, state: dict[str, Any], status: str = "running") -> None:
        step = self.fsm.state(RESEARCH_CYCLE, cycle_id) or "UNKNOWN"
        self.stores.checkpoints.save(cycle_id, WORKFLOW, step, state, status)

    def _task(self, cycle_id: str, kind: str, inputs: dict[str, Any], key: str = "") -> Task:
        request = TaskRequest(kind, self.contracts.role_for_task(kind), inputs, idempotency_key=key)
        return delegate(self.stores.queue, self.bus, self.cmap, CIO, request, cycle_id)

    def _advance(
        self,
        cycle_id: str,
        target: str,
        reason: str,
        state: dict[str, Any],
        artifacts: dict[str, str] | None = None,
    ) -> None:
        self.fsm.transition(RESEARCH_CYCLE, cycle_id, target, CIO.value, reason, artifacts)
        self._save(cycle_id, state, "completed" if target == "COMPLETED" else "running")

    def abort(self, cycle_id: str, reason: str) -> None:
        state = self._state(cycle_id)
        state["abort_reason"] = reason
        step = self.fsm.state(RESEARCH_CYCLE, cycle_id)
        try:
            self.fsm.transition(RESEARCH_CYCLE, cycle_id, "ABORTED", CIO.value, reason[:200])
        except InvalidTransitionError:
            return
        lessons = lessons_from_aborted_cycle(
            cycle_id,
            reason,
            step=step,
            hypothesis_id=state.get("hypothesis_id"),
            now=self.clock.now(),
        )
        for les in lessons:
            self.stores.documents.put(les, recorded_by=CIO.value)
        state["abort_lesson_ids"] = [les.id for les in lessons]
        self._save(cycle_id, state, "aborted")
        self.stores.audit.record(
            CIO.value,
            "cycle.aborted_lessons",
            cycle_id,
            {"lesson_ids": state["abort_lesson_ids"], "step": step, "reason": reason[:200]},
        )

    @staticmethod
    def _failed(*tasks: Task) -> Task | None:
        return next((t for t in tasks if t.status is TaskStatus.FAILED), None)

    @staticmethod
    def _done(*tasks: Task) -> bool:
        return all(t.status is TaskStatus.SUCCEEDED for t in tasks)

    def active_cycles(self) -> list[str]:
        return [c["workflow_id"] for c in self.stores.checkpoints.list(WORKFLOW, "running")]

    # ---------------------------------------------------------------- the 13 steps
    def tick(self, cycle_id: str) -> bool:
        """Advance one cycle as far as current task results allow. Returns True on progress."""
        if self.stores.flags.kill_switch_active():
            return False
        step = self.fsm.state(RESEARCH_CYCLE, cycle_id)
        if step in (None, "COMPLETED", "ABORTED"):
            return False
        state = self._state(cycle_id)
        handler = getattr(self, f"_from_{step.lower()}")
        progressed: bool = handler(cycle_id, state)
        return progressed

    def _research_inputs(self, state: dict[str, Any]) -> dict[str, Any]:
        objective = self.stores.documents.require(ResearchObjective, state["objective_id"])
        return {
            "topic": objective.statement,
            "keywords": list(objective.topics),
            "objective_id": objective.id,
        }

    def _data_inputs(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "symbols": state["symbols"],
            "start": state["research_start"],
            "end": state["research_end"],
            "frequency": "8h",
        }

    def _from_objective_received(self, cycle_id: str, state: dict[str, Any]) -> bool:
        intel = self._task(cycle_id, "intelligence.research", self._research_inputs(state))
        data = self._task(cycle_id, "data.acquire", self._data_inputs(state))
        state["tasks"] = {"intelligence": intel.id, "data": data.id}
        self._advance(
            cycle_id,
            "DELEGATED",
            "tasks_assigned",
            state,
            {"intelligence_task": intel.id, "data_task": data.id},
        )
        return True

    def _get(self, task_id: str) -> Task:
        task = self.stores.queue.get(task_id)
        if task is None:
            raise KeyError(task_id)
        return task

    def _from_delegated(self, cycle_id: str, state: dict[str, Any]) -> bool:
        intel = self._get(state["tasks"]["intelligence"])
        if (failed := self._failed(intel)) is not None:
            self.abort(cycle_id, f"intelligence task failed: {failed.error}")
            return True
        if not self._done(intel):
            return False
        sources = list(intel.outputs.get("source_ids", []))
        if not sources:
            self.abort(cycle_id, "no external sources retrieved")
            return True
        state["source_ids"] = sources
        self._advance(
            cycle_id,
            "RESEARCH_RETRIEVED",
            "sources_ingested",
            state,
            {"sources": sha256_of(sorted(sources))},
        )
        return True

    def _from_research_retrieved(self, cycle_id: str, state: dict[str, Any]) -> bool:
        intel = self._get(state["tasks"]["intelligence"])
        evidence = list(intel.outputs.get("evidence_ids", []))
        if not evidence:
            self.abort(cycle_id, "no source-backed evidence extracted")
            return True
        state["evidence_ids"] = evidence
        self._advance(
            cycle_id,
            "KNOWLEDGE_STORED",
            "evidence_stored",
            state,
            {"evidence": sha256_of(sorted(evidence))},
        )
        return True

    def _from_knowledge_stored(self, cycle_id: str, state: dict[str, Any]) -> bool:
        data = self._get(state["tasks"]["data"])
        if (failed := self._failed(data)) is not None:
            self.abort(cycle_id, f"data task failed: {failed.error}")
            return True
        if not self._done(data):
            return False
        dataset_id = str(data.outputs["dataset_version_id"])
        micro = self._task(cycle_id, "microstructure.analyze", {"dataset_version_id": dataset_id})
        state["tasks"]["microstructure"] = micro.id
        if (failed := self._failed(micro)) is not None:
            self.abort(cycle_id, f"microstructure task failed: {failed.error}")
            return True
        if not self._done(micro):
            self._save(cycle_id, state)
            return False
        quant_inputs: dict[str, Any] = {
            "objective_id": state["objective_id"],
            "evidence_ids": state["evidence_ids"],
            "dataset_version_id": dataset_id,
            "analysis_id": micro.outputs["analysis_id"],
            "proposal_id": state.get("proposal_id"),
            "cycle_id": cycle_id,
        }
        # Opt-in LLM hypothesis path (default remains deterministic templates).
        if os.environ.get("ENIGMA_HYPOTHESIS_MODE", "").strip().lower() == "llm":
            quant_inputs["hypothesis_mode"] = "llm"
        quant = self._task(cycle_id, "quant.hypothesize", quant_inputs)
        state["tasks"]["quant"] = quant.id
        if (failed := self._failed(quant)) is not None:
            self.abort(cycle_id, f"hypothesis task failed: {failed.error}")
            return True
        if not self._done(quant):
            self._save(cycle_id, state)
            return False
        state["hypothesis_id"] = quant.outputs["hypothesis_id"]
        state["dataset_version_id"] = dataset_id
        self._advance(
            cycle_id,
            "HYPOTHESIS_GENERATED",
            "hypothesis_registered",
            state,
            {"hypothesis": str(quant.outputs["hypothesis_id"])},
        )
        return True

    def _from_hypothesis_generated(self, cycle_id: str, state: dict[str, Any]) -> bool:
        data = self._get(state["tasks"]["data"])
        if not data.outputs.get("pit_certified", False):
            self.abort(cycle_id, "research dataset is not PIT certified")
            return True
        self._advance(
            cycle_id,
            "DATA_ACQUIRED",
            "dataset_bound",
            state,
            {"dataset_version": str(data.outputs["content_hash"])},
        )
        return True

    def _from_data_acquired(self, cycle_id: str, state: dict[str, Any]) -> bool:
        impl = self._task(
            cycle_id,
            "strategy.implement",
            {
                "hypothesis_id": state["hypothesis_id"],
                "dataset_version_id": state["dataset_version_id"],
                "cycle_id": cycle_id,
            },
        )
        state["tasks"]["strategy"] = impl.id
        if (failed := self._failed(impl)) is not None:
            self.abort(cycle_id, f"strategy task failed: {failed.error}")
            return True
        if not self._done(impl):
            self._save(cycle_id, state)
            return False
        out = impl.outputs
        state["candidate_package_id"] = out.get("candidate_package_id")
        state["experiment_ids"] = out.get("experiment_ids", [])
        self._advance(
            cycle_id,
            "STRATEGY_EXECUTED",
            "sandboxed_runs_completed",
            state,
            {"packages": sha256_of(sorted(out.get("package_ids", [])))},
        )
        return True

    def _from_strategy_executed(self, cycle_id: str, state: dict[str, Any]) -> bool:
        impl = self._get(state["tasks"]["strategy"])
        if not impl.outputs.get("candidate_package_id"):
            errors = impl.outputs.get("trial_errors") or ["unknown"]
            self.abort(cycle_id, f"no strategy trial ran successfully: {errors[0]}"[:500])
            return True
        if not impl.outputs.get("reproducible", False):
            self.abort(cycle_id, "twin backtests were not reproducible")
            return True
        self._advance(
            cycle_id,
            "BACKTEST_RECORDED",
            "reproducible_backtests_recorded",
            state,
            {"results": sha256_of(impl.outputs.get("result_hashes", []))},
        )
        return True

    def _from_backtest_recorded(self, cycle_id: str, state: dict[str, Any]) -> bool:
        package_id = state.get("candidate_package_id")
        if not package_id:
            self.abort(cycle_id, "no strategy package ran successfully")
            return True
        request_id = "vrq_" + sha256_hex(f"{cycle_id}|{package_id}")[:26].upper()
        if self.stores.documents.get(ValidationRequest, request_id) is None:
            hyp_family = self._get(state["tasks"]["quant"]).outputs["family_id"]
            request = ValidationRequest(
                id=request_id,
                strategy_package_id=package_id,
                experiment_ids=tuple(state["experiment_ids"]),
                family_id=hyp_family,
                dataset_version_id=state["dataset_version_id"],
                requested_by=CIO.value,
                requested_at=self.clock.now(),
            )
            self.stores.documents.put(request, recorded_by=CIO.value)
            if self.fsm.state(STRATEGY_LIFECYCLE, package_id) == "RESEARCH_CANDIDATE":
                self.fsm.transition(
                    STRATEGY_LIFECYCLE,
                    package_id,
                    "VALIDATION_QUEUED",
                    CIO.value,
                    "submitted",
                    {"strategy_package": package_id},
                )
        val = self._task(cycle_id, "validation.validate", {"request_id": request_id})
        state["tasks"]["validation"] = val.id
        state["validation_request_id"] = request_id
        if (failed := self._failed(val)) is not None:
            self.abort(cycle_id, f"validation task failed: {failed.error}")
            return True
        if not self._done(val):
            self._save(cycle_id, state)
            return False
        summary_id = str(val.outputs["summary_id"])
        summary = self.stores.documents.require(RedactedValidationSummary, summary_id)
        state["validation_summary_id"] = summary_id
        state["validation_overall"] = summary.overall.value
        self._advance(
            cycle_id,
            "VALIDATION_DECIDED",
            f"validation_{summary.overall.value}",
            state,
            {"validation_report": summary.report_id},
        )
        return True

    def _from_validation_decided(self, cycle_id: str, state: dict[str, Any]) -> bool:
        if state.get("validation_overall") == "PASS":
            self._task(
                cycle_id,
                "risk.review",
                {
                    "package_id": state["candidate_package_id"],
                    "summary_id": state["validation_summary_id"],
                },
            )
        meta = self._task(
            cycle_id,
            "meta.reflect_cycle",
            {
                "cycle_id": cycle_id,
                "hypothesis_id": state["hypothesis_id"],
                "experiment_ids": state["experiment_ids"],
                "summary_id": state.get("validation_summary_id"),
            },
        )
        state["tasks"]["reflect"] = meta.id
        if (failed := self._failed(meta)) is not None:
            self.abort(cycle_id, f"reflection task failed: {failed.error}")
            return True
        if not self._done(meta):
            self._save(cycle_id, state)
            return False
        state["reflection_id"] = meta.outputs["reflection_id"]
        state["lesson_ids"] = meta.outputs["lesson_ids"]
        self._advance(
            cycle_id,
            "REFLECTED",
            "cycle_reflection_recorded",
            state,
            {"reflection": str(meta.outputs["reflection_id"])},
        )
        return True

    def _from_reflected(self, cycle_id: str, state: dict[str, Any]) -> bool:
        if not state.get("lesson_ids"):
            self.abort(cycle_id, "no lessons were stored")
            return True
        self._advance(
            cycle_id,
            "LESSONS_STORED",
            "lessons_stored",
            state,
            {"lessons": sha256_of(sorted(state["lesson_ids"]))},
        )
        return True

    def _from_lessons_stored(self, cycle_id: str, state: dict[str, Any]) -> bool:
        nxt = self._task(
            cycle_id,
            "meta.propose_next",
            {
                "cycle_id": cycle_id,
                "hypothesis_id": state["hypothesis_id"],
                "experiment_ids": state["experiment_ids"],
                "summary_id": state.get("validation_summary_id"),
                "lesson_ids": state["lesson_ids"],
            },
        )
        state["tasks"]["next"] = nxt.id
        if (failed := self._failed(nxt)) is not None:
            self.abort(cycle_id, f"next-experiment task failed: {failed.error}")
            return True
        if not self._done(nxt):
            self._save(cycle_id, state)
            return False
        state["next_proposal_id"] = nxt.outputs["proposal_id"]
        self._advance(
            cycle_id,
            "NEXT_EXPERIMENT_PROPOSED",
            "next_experiment_proposed",
            state,
            {"proposal": str(nxt.outputs["proposal_id"])},
        )
        return True

    def _from_next_experiment_proposed(self, cycle_id: str, state: dict[str, Any]) -> bool:
        self._advance(cycle_id, "COMPLETED", "cycle_completed", state)
        self.stores.audit.record(
            CIO.value,
            "cycle.completed",
            cycle_id,
            {
                "validation_overall": state.get("validation_overall"),
                "next_proposal_id": state.get("next_proposal_id"),
            },
        )
        return True
