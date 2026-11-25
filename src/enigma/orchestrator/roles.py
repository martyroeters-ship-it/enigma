"""Runtime implementations of the institutional roles used by the research cycle.

Every capability is reached through `ctx.tools` (contract allowlist + audit). Roles whose
contracts mark `deterministic_core` never call a model.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from enigma.agents.base import Agent, AgentContext, DecisionRecord, TaskOutcome
from enigma.core.build_info import code_sha
from enigma.core.errors import BudgetExceededError, ProviderUnavailableError
from enigma.core.hashing import sha256_hex, sha256_of
from enigma.learning.lessons import CycleEvidence, derive_lessons, propose_next
from enigma.learning.review import review_institution
from enigma.research.hypothesis import build_hypothesis, score_mechanisms
from enigma.research.mechanisms import MECHANISMS
from enigma.research.microstructure import METHOD as MICRO_METHOD
from enigma.research.microstructure import analyze_panel, findings
from enigma.research.trials import trial_plan
from enigma.schemas.common import AgentRole
from enigma.schemas.cycle import MarketAnalysis, NextExperimentProposal
from enigma.schemas.governance import Reflection, ReflectionOutcome
from enigma.schemas.knowledge import EvidenceItem, FailureCondition, Lesson
from enigma.schemas.research import (
    DatasetVersion,
    Experiment,
    ExperimentStatus,
    Hypothesis,
    ResearchObjective,
    StrategyPackage,
)
from enigma.schemas.validation import RedactedValidationSummary

REPRODUCIBILITY_EPS = 1e-9


def _finite(value: float | None) -> float:
    return value if value is not None and math.isfinite(value) else -math.inf


def _hid(prefix: str, *parts: object) -> str:
    return f"{prefix}_" + sha256_hex("|".join(str(p) for p in parts))[:26].upper()


class IntelligenceAgent(Agent):
    role = AgentRole.INTELLIGENCE

    def execute(self, ctx: AgentContext) -> TaskOutcome:
        inputs = ctx.task.inputs
        prior = ctx.tools.call(
            "knowledge.retrieve", query=inputs["topic"], topics=inputs.get("keywords", [])
        )
        result: dict[str, Any] = ctx.tools.call(
            "intel.research",
            topic=inputs["topic"],
            keywords=inputs.get("keywords", []),
            max_sources=int(inputs.get("max_sources", 30)),
            depth=int(inputs.get("depth", 1)),
            budget=ctx.budget,
        )
        n_src, n_ev = len(result["source_ids"]), len(result["evidence_ids"])
        return TaskOutcome(
            outputs=result,
            decisions=[
                DecisionRecord(
                    "research_plan",
                    "executed",
                    "multi_provider_search",
                    f"Searched {', '.join(sorted(result.get('providers', {})))} with "
                    f"{len(result.get('queries', []))} queries (depth {inputs.get('depth', 1)}); "
                    f"{n_src} sources, {n_ev} evidence items, {result.get('duplicates', 0)} "
                    f"duplicates merged; {len(prior.get('evidence_ids', []))} prior evidence items "
                    "were already in the registry.",
                    output_refs=tuple(result["source_ids"][:50]),
                )
            ],
            metrics={"sources": float(n_src), "evidence": float(n_ev)},
            what_worked=f"{n_src} sources with provenance" if n_src else "",
            what_failed="" if n_ev else "no evidence extracted",
        )


class DataEngineeringAgent(Agent):
    role = AgentRole.DATA_ENGINEERING

    def execute(self, ctx: AgentContext) -> TaskOutcome:
        i = ctx.task.inputs
        result: dict[str, Any] = ctx.tools.call(
            "data.acquire_research",
            symbols=list(i["symbols"]),
            start=datetime.fromisoformat(i["start"]),
            end=datetime.fromisoformat(i["end"]),
            frequency=i.get("frequency", "8h"),
            budget=ctx.budget,
        )
        failed = result.get("failed_checks", [])
        return TaskOutcome(
            outputs=result,
            decisions=[
                DecisionRecord(
                    "dataset_publication",
                    "published",
                    "pit_certified" if result["pit_certified"] else "pit_not_certified",
                    f"Published {result['rows']} rows for {', '.join(i['symbols'])} "
                    f"{i['start'][:10]}..{i['end'][:10]}; failed checks: {failed or 'none'}.",
                    output_refs=(result["dataset_version_id"],),
                )
            ],
            metrics={"rows": float(result["rows"])},
            what_worked="checksummed public archives",
            what_failed=", ".join(failed),
        )


class MarketMicrostructureAgent(Agent):
    role = AgentRole.MARKET_MICROSTRUCTURE

    def execute(self, ctx: AgentContext) -> TaskOutcome:
        dataset_id = str(ctx.task.inputs["dataset_version_id"])
        frame = ctx.tools.call("data.load_research", dataset_version_id=dataset_id)
        stats = analyze_panel(frame)
        analysis = MarketAnalysis(
            id=_hid("mka", dataset_id, MICRO_METHOD),
            dataset_version_id=dataset_id,
            method=MICRO_METHOD,
            per_symbol=stats,
            findings=findings(stats),
            created_by=self.role,
            created_at=ctx.clock.now(),
        )
        ctx.view.put(analysis)
        return TaskOutcome(
            outputs={"analysis_id": analysis.id, "findings": list(analysis.findings)},
            decisions=[
                DecisionRecord(
                    "market_analysis",
                    "recorded",
                    "funding_basis_stats",
                    " ".join(analysis.findings)[:2000],
                    input_refs=(dataset_id,),
                    output_refs=(analysis.id,),
                )
            ],
        )


class _Critique(BaseModel):
    weaknesses: list[str] = Field(max_length=5)
    additional_falsification: list[str] = Field(max_length=3)


class QuantResearchAgent(Agent):
    role = AgentRole.QUANT_RESEARCH

    def execute(self, ctx: AgentContext) -> TaskOutcome:
        i = ctx.task.inputs
        objective = ctx.view.require(ResearchObjective, i["objective_id"])
        evidence = [e for e in (ctx.view.get(EvidenceItem, x) for x in i["evidence_ids"]) if e]
        prior = ctx.tools.call(
            "knowledge.retrieve", query=objective.statement, topics=list(MECHANISMS)
        )
        lessons = [x for x in (ctx.view.get(Lesson, lid) for lid in prior["lesson_ids"]) if x]
        failures = [
            x for x in (ctx.view.get(FailureCondition, f) for f in prior["failure_ids"]) if x
        ]
        tested = Counter(h.family_id.removeprefix("fam_") for h in ctx.view.list(Hypothesis))
        analysis = ctx.view.require(MarketAnalysis, i["analysis_id"])
        scores = score_mechanisms(evidence, analysis, lessons, failures, dict(tested))
        proposal = (
            ctx.view.require(NextExperimentProposal, i["proposal_id"])
            if i.get("proposal_id")
            else None
        )
        chosen = (
            next((s for s in scores if s.mechanism_id == proposal.mechanism_id), scores[0])
            if proposal
            else scores[0]
        )
        hypothesis = build_hypothesis(
            objective,
            MECHANISMS[chosen.mechanism_id],
            chosen,
            objective.instruments,
            cycle_id=i["cycle_id"],
            now=ctx.clock.now(),
            dataset_version_ids=(i["dataset_version_id"],),
            lesson_ids=tuple(les.id for les in lessons),
            parent=proposal,
        )
        decisions = [
            DecisionRecord(
                "hypothesis_selection",
                chosen.mechanism_id,
                "next_experiment_proposal" if proposal else "highest_support",
                "Mechanism scores (support - 0.5*contradiction + data - lessons): "
                + "; ".join(f"{s.mechanism_id}={s.total:.2f}" for s in scores)
                + f". Consulted {len(lessons)} lessons and {len(failures)} failure records.",
                input_refs=(objective.id, analysis.id, *chosen.supporting_evidence[:10]),
                output_refs=(hypothesis.id,),
            )
        ]
        hypothesis = self._critique(ctx, hypothesis, decisions)
        ctx.view.put(hypothesis)
        ctx.tools.call("knowledge.index", doc=hypothesis)
        return TaskOutcome(
            outputs={
                "hypothesis_id": hypothesis.id,
                "family_id": hypothesis.family_id,
                "mechanism_id": chosen.mechanism_id,
                "trial_budget": hypothesis.trial_budget,
                "scores": {s.mechanism_id: round(s.total, 4) for s in scores},
            },
            decisions=decisions,
            what_worked=f"{len(chosen.supporting_evidence)} supporting evidence items",
        )

    def _critique(
        self, ctx: AgentContext, hyp: Hypothesis, decisions: list[DecisionRecord]
    ) -> Hypothesis:
        """Optional model critique; recorded as AI interpretation, never changes parameters."""
        try:
            critique: _Critique = ctx.tools.call(
                "llm.structured",
                model_cls=_Critique,
                system="You critique quantitative trading hypotheses. Be specific and brief.",
                user=f"Mechanism: {hyp.mechanism}\nFormal statement: {hyp.formal_statement}",
                task_id=ctx.task.id,
                budget=ctx.budget,
            )
        except (ProviderUnavailableError, BudgetExceededError) as exc:
            decisions.append(
                DecisionRecord(
                    "llm_critique",
                    "skipped",
                    "provider_unavailable",
                    f"Deterministic path only: {exc}"[:500],
                )
            )
            return hyp
        notes = " | AI critique (unverified): " + "; ".join(critique.weaknesses)
        decisions.append(
            DecisionRecord("llm_critique", "recorded", "ai_interpretation", notes[:2000])
        )
        return hyp.model_copy(update={"uncertainty_notes": hyp.uncertainty_notes + notes})


class StrategyEngineeringAgent(Agent):
    role = AgentRole.STRATEGY_ENGINEERING

    def execute(self, ctx: AgentContext) -> TaskOutcome:
        i = ctx.task.inputs
        hyp = ctx.view.require(Hypothesis, i["hypothesis_id"])
        dataset = ctx.view.require(DatasetVersion, i["dataset_version_id"])
        mech = MECHANISMS[hyp.family_id.removeprefix("fam_")]
        plan = trial_plan(hyp)
        experiments: list[Experiment] = []
        packages: list[str] = []
        for index, params in enumerate(plan, start=1):
            exp_id = _hid("exp", hyp.id, index)
            existing = ctx.view.get(Experiment, exp_id)
            if existing is not None and existing.status in (
                ExperimentStatus.SUCCEEDED,
                ExperimentStatus.FAILED,
            ):
                experiments.append(existing)
                packages.append(existing.strategy_package_id)
                continue
            package: StrategyPackage = ctx.tools.call(
                "strategy.build",
                source_code=mech.source_template,
                entrypoint=mech.entrypoint,
                params=params,
                hypothesis_id=hyp.id,
                name=f"{mech.id}-{index}",
                required_columns=mech.features,
                cost_model_version=hyp.cost_model_version,
            )
            ctx.view.put(package)
            packages.append(package.id)
            ctx.tools.call(
                "lifecycle.transition",
                subject_id=package.id,
                target="DRAFT_HYPOTHESIS",
                reason="package_built",
            )
            config_hash = sha256_of(
                {
                    "params": params,
                    "cost_model": package.cost_model,
                    "dataset": dataset.id,
                    "seed": 0,
                }
            )
            exp = Experiment(
                id=exp_id,
                hypothesis_id=hyp.id,
                family_id=hyp.family_id,
                trial_index=index,
                strategy_package_id=package.id,
                dataset_version_id=dataset.id,
                research_window=(dataset.time_start, dataset.time_end),
                code_sha=code_sha(),
                config_hash=config_hash,
                seed=0,
                created_by=self.role,
            )
            ctx.view.put(exp)
            ctx.tools.call(
                "lifecycle.transition",
                subject_id=package.id,
                target="RESEARCH_EXPERIMENT_RUNNING",
                reason="trial_registered",
            )
            exp = exp.model_copy(
                update={"status": ExperimentStatus.RUNNING, "started_at": ctx.clock.now()}
            )
            ctx.view.put(exp)
            try:
                first = ctx.tools.call(
                    "backtest.run", package_id=package.id, dataset_version_id=dataset.id, seed=0
                )
                twin = ctx.tools.call(
                    "backtest.run", package_id=package.id, dataset_version_id=dataset.id, seed=0
                )
            except Exception as exc:  # sandbox violations and runtime errors are trial results
                exp = exp.model_copy(
                    update={
                        "status": ExperimentStatus.FAILED,
                        "error": f"{type(exc).__name__}: {exc}"[:1000],
                        "ended_at": ctx.clock.now(),
                    }
                )
                ctx.view.put(exp)
                experiments.append(exp)
                ctx.tools.call(
                    "lifecycle.transition",
                    subject_id=package.id,
                    target="RESEARCH_REJECTED",
                    reason="trial_failed",
                )
                continue
            delta = max(
                (
                    abs(first["metrics"][k] - twin["metrics"].get(k, math.inf))
                    for k in first["metrics"]
                ),
                default=0.0,
            )
            reproducible = (
                delta <= REPRODUCIBILITY_EPS and first["result_hash"] == twin["result_hash"]
            )
            metrics = {k: float(v) for k, v in first["metrics"].items()}
            metrics["twin_max_abs_delta"] = float(delta)
            metrics["twin_reproducible"] = 1.0 if reproducible else 0.0
            exp = exp.model_copy(
                update={
                    "status": ExperimentStatus.SUCCEEDED,
                    "metrics": metrics,
                    "result_hash": first["result_hash"],
                    "artifact_uri": first.get("artifact_uri"),
                    "ended_at": ctx.clock.now(),
                }
            )
            ctx.view.put(exp)
            experiments.append(exp)
        succeeded = [e for e in experiments if e.status is ExperimentStatus.SUCCEEDED]
        candidate = max(
            succeeded,
            key=lambda e: (_finite(e.metrics.get("net_sharpe")), -e.trial_index),
            default=None,
        )
        for exp in succeeded:
            target = (
                "RESEARCH_CANDIDATE"
                if candidate and exp.id == candidate.id
                else "RESEARCH_REJECTED"
            )
            ctx.tools.call(
                "lifecycle.transition",
                subject_id=exp.strategy_package_id,
                target=target,
                reason="trial_selection",
                artifacts={"experiment_result": exp.result_hash or ""},
            )
        table = {e.id: round(e.metrics.get("net_sharpe", float("nan")), 4) for e in succeeded}
        reproducible = bool(succeeded) and all(
            e.metrics.get("twin_reproducible") == 1.0 for e in succeeded
        )
        return TaskOutcome(
            outputs={
                "experiment_ids": [e.id for e in experiments],
                "package_ids": packages,
                "candidate_package_id": candidate.strategy_package_id if candidate else None,
                "candidate_experiment_id": candidate.id if candidate else None,
                "result_hashes": [e.result_hash for e in succeeded],
                "reproducible": reproducible,
                "trial_errors": [e.error for e in experiments if e.error][:3],
                "net_sharpe_by_experiment": table,
            },
            decisions=[
                DecisionRecord(
                    "candidate_selection",
                    candidate.id if candidate else "none",
                    "max_research_net_sharpe",
                    f"Ran {len(experiments)} registered trials in the sandbox "
                    f"({len(succeeded)} succeeded). Research-window net Sharpe by trial: {table}. "
                    "Selection on the research window is accounted for by the trial ledger.",
                    input_refs=(hyp.id, dataset.id),
                    output_refs=tuple(e.id for e in experiments),
                )
            ],
            metrics={"trials": float(len(experiments)), "succeeded": float(len(succeeded))},
            what_worked="all twin backtests reproducible" if reproducible else "",
            what_failed=(
                ""
                if reproducible
                else "; ".join(e.error or "" for e in experiments if e.error)[:500]
                or "non-reproducible backtest detected"
            ),
        )


class IndependentValidationAgent(Agent):
    role = AgentRole.INDEPENDENT_VALIDATION

    def execute(self, ctx: AgentContext) -> TaskOutcome:
        result: dict[str, Any] = ctx.tools.call(
            "validation.run", request_id=ctx.task.inputs["request_id"]
        )
        return TaskOutcome(
            outputs=result,
            decisions=[
                DecisionRecord(
                    "validation_outcome",
                    str(result["overall"]),
                    "programmatic_gates",
                    f"Validation service computed overall={result['overall']} under protocol "
                    f"{result['protocol_version']}; failed gates: "
                    f"{result.get('failed_gates') or []}. "
                    "This record restates the service output and does not alter it.",
                    output_refs=(str(result["summary_id"]),),
                )
            ],
        )


class RiskManagementAgent(Agent):
    role = AgentRole.RISK_MANAGEMENT

    def execute(self, ctx: AgentContext) -> TaskOutcome:
        review: dict[str, Any] = ctx.tools.call(
            "risk.evaluate",
            package_id=ctx.task.inputs["package_id"],
            summary_id=ctx.task.inputs["summary_id"],
        )
        return TaskOutcome(
            outputs=review,
            decisions=[
                DecisionRecord(
                    "deployment_review",
                    str(review["decision"]),
                    ",".join(review["reason_codes"]),
                    f"Deterministic limit checks: {review['checks']}.",
                )
            ],
        )


class MetaLearningAgent(Agent):
    role = AgentRole.META_LEARNING

    def _evidence(self, ctx: AgentContext) -> CycleEvidence:
        i = ctx.task.inputs
        hyp = ctx.view.require(Hypothesis, i["hypothesis_id"])
        exps = tuple(ctx.view.require(Experiment, e) for e in i["experiment_ids"])
        summary = (
            ctx.view.get(RedactedValidationSummary, i["summary_id"])
            if i.get("summary_id")
            else None
        )
        return CycleEvidence(i["cycle_id"], hyp, exps, summary, hyp.family_id.removeprefix("fam_"))

    def execute(self, ctx: AgentContext) -> TaskOutcome:
        if ctx.task.kind == "meta.propose_next":
            return self._propose(ctx)
        ev = self._evidence(ctx)
        now = ctx.clock.now()
        lessons = derive_lessons(ev, now)
        for les in lessons:
            ctx.view.put(les)
            ctx.tools.call("knowledge.index", doc=les)
        tasks: list[dict[str, Any]] = ctx.tools.call("tasks.for_cycle", cycle_id=ev.cycle_id)
        task_refl = [
            r for t in tasks for r in ctx.view.list(Reflection, where={"task_id": t["id"]})
        ]
        failed_tasks = [t["kind"] for t in tasks if t["status"] == "failed"]
        passed = ev.summary is not None and ev.summary.overall.value == "PASS"
        reflection = Reflection(
            id=_hid("ref", ev.cycle_id, "cycle"),
            task_id=ctx.task.id,
            agent=self.role,
            outcome=ReflectionOutcome.SUCCESS if passed else ReflectionOutcome.FAILURE,
            what_worked=(
                f"Cycle executed {len(tasks)} delegated tasks across "
                f"{len({t['assigned_to'] for t in tasks})} roles; "
                f"{len(ev.experiments)} trials registered before running."
            ),
            what_failed=(
                "; ".join(les.statement for les in lessons if les.outcome == "failure") or "none"
            )[:4000],
            suspected_systemic_issue=next(
                (
                    t
                    for t in ("cost_fragility", "overfit", "multiple_testing", "no_signal")
                    if any(t in les.topics for les in lessons)
                ),
                None,
            ),
            proposed_followup="See next-experiment proposal for this cycle.",
            metrics={
                "task_reflections": float(len(task_refl)),
                "failed_tasks": float(len(failed_tasks)),
                "wall_seconds": sum(r.metrics.get("wall_seconds", 0.0) for r in task_refl),
                "http_requests": sum(r.metrics.get("http_requests", 0.0) for r in task_refl),
            },
            created_at=now,
        )
        ctx.view.put(reflection)
        status = ctx.tools.call("gateway.status")
        proposals = review_institution(
            ctx.view.list(Reflection, limit=500),
            ctx.view.list(Lesson, limit=500),
            status.get("missing_credentials", []),
            now,
            window_key=ev.cycle_id,
        )
        for proposal in proposals:
            ctx.tools.call("improvement.propose", proposal=proposal)
        return TaskOutcome(
            outputs={
                "reflection_id": reflection.id,
                "lesson_ids": [x.id for x in lessons],
                "improvement_proposal_ids": [p.id for p in proposals],
            },
            decisions=[
                DecisionRecord(
                    "cycle_reflection",
                    reflection.outcome.value,
                    reflection.suspected_systemic_issue or "none",
                    " | ".join(les.statement for les in lessons)[:2000],
                    output_refs=tuple(x.id for x in lessons),
                )
            ],
        )

    def _propose(self, ctx: AgentContext) -> TaskOutcome:
        ev = self._evidence(ctx)
        lessons = [ctx.view.require(Lesson, x) for x in ctx.task.inputs["lesson_ids"]]
        tested = Counter(h.family_id.removeprefix("fam_") for h in ctx.view.list(Hypothesis))
        proposal = propose_next(ev, lessons, dict(tested), ctx.clock.now())
        ctx.view.put(proposal)
        return TaskOutcome(
            outputs={"proposal_id": proposal.id, "mechanism_id": proposal.mechanism_id},
            decisions=[
                DecisionRecord(
                    "next_experiment",
                    proposal.mechanism_id,
                    "lesson_driven",
                    proposal.rationale,
                    input_refs=proposal.lesson_ids,
                    output_refs=(proposal.id,),
                )
            ],
        )


ALL_ROLE_AGENTS: tuple[type[Agent], ...] = (
    IntelligenceAgent,
    DataEngineeringAgent,
    MarketMicrostructureAgent,
    QuantResearchAgent,
    StrategyEngineeringAgent,
    IndependentValidationAgent,
    RiskManagementAgent,
    MetaLearningAgent,
)
