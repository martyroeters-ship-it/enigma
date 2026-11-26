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
from enigma.research.bridge import hypothesis_from_proposal
from enigma.research.context import build_research_context
from enigma.research.hybrid import compare_funding_to_returns
from enigma.research.hypothesis import build_hypothesis, score_mechanisms
from enigma.research.mechanisms import MECHANISMS, family_id_for_domain
from enigma.research.microstructure import METHOD as MICRO_METHOD
from enigma.research.microstructure import analyze_panel, findings
from enigma.research.trials import trial_plan
from enigma.schemas.common import AgentRole
from enigma.schemas.cycle import MarketAnalysis, NextExperimentProposal
from enigma.schemas.governance import ExecutionMode, Reflection, ReflectionOutcome
from enigma.schemas.knowledge import EvidenceItem, FailureCondition, Lesson
from enigma.schemas.quant import (
    DatasetInfo,
    GenerationResult,
    HypothesisProposal,
    StrategyGenerationRecord,
)
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


def _hybrid_on_frame(frame: Any) -> dict[str, Any]:
    columns = set(getattr(frame, "columns", ()))
    if not {"funding_rate", "perp_close", "symbol", "available_at"} <= columns:
        return {"hybrid_status": "unavailable"}
    symbols = sorted(frame.get_column("symbol").unique().to_list())
    if not symbols:
        return {"hybrid_status": "unavailable"}
    part = frame.filter(frame.get_column("symbol") == symbols[0]).sort("available_at")
    comparison = compare_funding_to_returns(
        part.get_column("funding_rate").to_list(),
        part.get_column("perp_close").to_list(),
    )
    if comparison is None:
        return {"hybrid_status": "insufficient_rows", "hybrid_symbol": symbols[0]}
    return {
        "hybrid_status": "compared",
        "hybrid_symbol": symbols[0],
        "hybrid_winner": comparison.winner,
        "hybrid_loser_failure": comparison.loser_failure,
    }


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
        hybrid = _hybrid_on_frame(frame)
        return TaskOutcome(
            outputs={
                "analysis_id": analysis.id,
                "findings": list(analysis.findings),
                **hybrid,
            },
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
        if i.get("recovery") == "alternate":
            return self._execute_recovery(ctx)
        if i.get("hypothesis_mode") == "llm":
            return self._execute_llm(ctx)
        return self._execute_deterministic(ctx)

    def _execute_deterministic(self, ctx: AgentContext) -> TaskOutcome:
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
                execution_mode=ExecutionMode.DETERMINISTIC,
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
                "execution_mode": ExecutionMode.DETERMINISTIC,
            },
            decisions=decisions,
            what_worked=f"{len(chosen.supporting_evidence)} supporting evidence items",
        )

    def _execute_llm(self, ctx: AgentContext) -> TaskOutcome:
        """LLM primary path. Never falls back to mechanism templates on failure."""
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
        dataset = ctx.view.require(DatasetVersion, i["dataset_version_id"])
        priors = list(ctx.view.list(Hypothesis))
        context = build_research_context(
            objective=objective,
            research_cutoff=dataset.time_end,
            datasets=(
                DatasetInfo(
                    dataset_version_id=dataset.id,
                    name=dataset.name,
                    frequency=dataset.frequency,
                    bar_hours=1.0,
                    symbols=tuple(inst.symbol for inst in dataset.instruments),
                    columns=tuple(dataset.columns),
                    coverage_start=dataset.time_start,
                    coverage_end=dataset.time_end,
                    rows=dataset.row_count,
                ),
            ),
            evidence=evidence,
            lessons=lessons,
            failures=failures,
            prior_hypotheses=priors,
            cost_model_version="cm-binance-usdm-v1",
        )
        try:
            raw = ctx.tools.call(
                "quant.hypothesis_generate",
                context=context.model_dump(mode="json"),
                task_id=ctx.task.id,
                budget=ctx.budget,
            )
        except (ProviderUnavailableError, BudgetExceededError, KeyError) as exc:
            raise ProviderUnavailableError(
                f"LLM hypothesis generation unavailable (no silent template fallback): {exc}"
            ) from exc
        generation = GenerationResult.model_validate(raw)
        # Persist even on failure so operators can inspect rejected drafts.
        ctx.view.put(generation)
        if not generation.ok:
            if generation.failure_reason == "no_proposal_survived_checks":
                return self._checks_rejected(generation)
            raise RuntimeError(
                generation.failure_reason or "LLM hypothesis generation produced no survivors"
            )
        proposal: HypothesisProposal = generation.proposals[0]
        hypothesis = hypothesis_from_proposal(
            proposal,
            objective_id=objective.id,
            instruments=objective.instruments,
            generation=generation,
            dataset_version_ids=(dataset.id,),
            lesson_ids=tuple(les.id for les in lessons),
            created_by=self.role,
            created_at=ctx.clock.now(),
            cycle_id=i["cycle_id"],
        )
        decisions = [
            DecisionRecord(
                "hypothesis_selection",
                proposal.id,
                "llm_hypothesis_generator",
                f"LLM generation {generation.id} via {generation.prompt_ref}; "
                f"consulted {len(lessons)} lessons, {len(evidence)} evidence items.",
                input_refs=(objective.id, dataset.id, *proposal.supporting_evidence_ids[:10]),
                output_refs=(hypothesis.id,),
                execution_mode=ExecutionMode.LLM,
            )
        ]
        ctx.view.put(hypothesis)
        ctx.tools.call("knowledge.index", doc=hypothesis)
        return TaskOutcome(
            outputs={
                "hypothesis_id": hypothesis.id,
                "family_id": hypothesis.family_id,
                "mechanism_id": proposal.domain,
                "trial_budget": hypothesis.trial_budget,
                "generation_id": generation.id,
                "execution_mode": ExecutionMode.LLM,
                "prompt_ref": generation.prompt_ref,
            },
            decisions=decisions,
            what_worked=f"LLM proposal '{proposal.title}' passed programmatic checks",
        )

    def _checks_rejected(self, generation: GenerationResult) -> TaskOutcome:
        domains: list[str] = []
        for draft in generation.rejected:
            raw = draft.draft.get("domain") or draft.draft.get("mechanism_id")
            if isinstance(raw, str) and raw:
                domains.append(raw)
        return TaskOutcome(
            outputs={
                "hypothesis_id": None,
                "failure_reason": "no_proposal_survived_checks",
                "generation_id": generation.id,
                "rejected_domains": domains,
                "execution_mode": ExecutionMode.LLM,
            },
            decisions=[
                DecisionRecord(
                    "hypothesis_selection",
                    "none",
                    "no_proposal_survived_checks",
                    f"Generation {generation.id} is kept. No proposal survived checks.",
                    output_refs=(generation.id,),
                    execution_mode=ExecutionMode.LLM,
                )
            ],
            what_failed="no_proposal_survived_checks",
        )

    def _execute_recovery(self, ctx: AgentContext) -> TaskOutcome:
        """One alternate direction after LLM checks reject every draft.

        Recorded as a deterministic fallback, not as model reasoning. The rejected
        generation stays in the document store.
        """
        i = ctx.task.inputs
        excluded = {str(item) for item in i.get("excluded_mechanisms") or []}
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
        ranked = sorted(scores, key=lambda s: (tested.get(s.mechanism_id, 0), s.mechanism_id))
        chosen = next((s for s in ranked if s.mechanism_id not in excluded), None)
        if chosen is None:
            return TaskOutcome(
                outputs={
                    "hypothesis_id": None,
                    "failure_reason": "no_proposal_survived_checks",
                    "generation_id": i.get("rejected_generation_id"),
                    "execution_mode": ExecutionMode.DETERMINISTIC_FALLBACK,
                },
                what_failed="no registered mechanism left after check rejection",
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
        )
        hypothesis = hypothesis.model_copy(
            update={
                "uncertainty_notes": (
                    hypothesis.uncertainty_notes
                    + " | Deterministic fallback after no_proposal_survived_checks."
                    " The rejected LLM generation is retained."
                )
            }
        )
        ctx.view.put(hypothesis)
        ctx.tools.call("knowledge.index", doc=hypothesis)
        generation_id = str(i.get("rejected_generation_id") or "")
        return TaskOutcome(
            outputs={
                "hypothesis_id": hypothesis.id,
                "family_id": hypothesis.family_id,
                "mechanism_id": chosen.mechanism_id,
                "trial_budget": hypothesis.trial_budget,
                "generation_id": generation_id or None,
                "execution_mode": ExecutionMode.DETERMINISTIC_FALLBACK,
                "recovery": "alternate",
            },
            decisions=[
                DecisionRecord(
                    "hypothesis_selection",
                    chosen.mechanism_id,
                    "no_proposal_survived_checks",
                    "LLM drafts failed programmatic checks. One library mechanism was "
                    f"registered instead: {chosen.mechanism_id}.",
                    input_refs=tuple(ref for ref in (objective.id, generation_id) if ref),
                    output_refs=(hypothesis.id,),
                    execution_mode=ExecutionMode.DETERMINISTIC_FALLBACK,
                    fallback_reason="no_proposal_survived_checks",
                )
            ],
            what_worked=f"fallback mechanism {chosen.mechanism_id}",
            what_failed="no_proposal_survived_checks",
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


def _mechanism_id(family_id: str) -> str:
    """``fam_fam_cryptocurrency`` and ``fam_funding_carry`` both strip to one id."""
    return family_id_for_domain(family_id).removeprefix("fam_")


class StrategyEngineeringAgent(Agent):
    role = AgentRole.STRATEGY_ENGINEERING

    def execute(self, ctx: AgentContext) -> TaskOutcome:
        i = ctx.task.inputs
        hyp = ctx.view.require(Hypothesis, i["hypothesis_id"])
        mech_id = _mechanism_id(hyp.family_id)
        # Explicit ENIGMA_STRATEGY_MODE=llm is the only path that code-generates a
        # registered mechanism. An LLM hypothesis that names a library mechanism
        # is compiled from that mechanism's template. Domain "novel" is not in
        # MECHANISMS, so it still goes through codegen. There is no silent
        # substitution after a codegen failure.
        if i.get("strategy_mode") == "llm" or mech_id not in MECHANISMS:
            return self._execute_llm_codegen(ctx)
        return self._execute_template(ctx, compiled_choice=i.get("strategy_mode") == "compile")

    def _execute_llm_codegen(self, ctx: AgentContext) -> TaskOutcome:
        """LLM codegen path. Never falls back to mechanism templates on failure."""
        i = ctx.task.inputs
        hyp = ctx.view.require(Hypothesis, i["hypothesis_id"])
        dataset = ctx.view.require(DatasetVersion, i["dataset_version_id"])
        try:
            raw = ctx.tools.call(
                "strategy.codegen",
                hypothesis_id=hyp.id,
                dataset_version_id=dataset.id,
                task_id=ctx.task.id,
                budget=ctx.budget,
            )
        except (ProviderUnavailableError, BudgetExceededError, KeyError) as exc:
            raise ProviderUnavailableError(
                f"LLM strategy codegen unavailable (no silent template fallback): {exc}"
            ) from exc
        if not raw.get("ok"):
            raise RuntimeError(raw.get("failure_reason") or "strategy codegen failed")
        if isinstance(raw.get("record"), dict):
            ctx.view.put(StrategyGenerationRecord.model_validate(raw["record"]))
        base = ctx.view.require(StrategyPackage, raw["package_id"])
        return self._run_trials(
            ctx,
            hyp,
            dataset,
            source_code=base.source_code,
            entrypoint=base.entrypoint,
            required_columns=tuple(base.required_columns),
            name_prefix="llm",
            execution_mode=ExecutionMode.LLM,
            generation_id=raw.get("generation_id"),
        )

    def _execute_template(self, ctx: AgentContext, *, compiled_choice: bool = False) -> TaskOutcome:
        i = ctx.task.inputs
        hyp = ctx.view.require(Hypothesis, i["hypothesis_id"])
        dataset = ctx.view.require(DatasetVersion, i["dataset_version_id"])
        mech = MECHANISMS[_mechanism_id(hyp.family_id)]
        outcome = self._run_trials(
            ctx,
            hyp,
            dataset,
            source_code=mech.source_template,
            entrypoint=mech.entrypoint,
            required_columns=mech.features,
            name_prefix=mech.id,
            execution_mode=ExecutionMode.DETERMINISTIC,
            implementation=(
                "compiled_registered_mechanism" if compiled_choice else "library_template"
            ),
        )
        if compiled_choice and outcome.decisions:
            decision = outcome.decisions[0]
            outcome.decisions[0] = DecisionRecord(
                decision.decision_type,
                decision.outcome,
                "compiled_registered_mechanism",
                "Compiled the registered template for the mechanism the hypothesis "
                "selected. No unrestricted Python was generated. " + decision.reasoning_summary,
                input_refs=decision.input_refs,
                output_refs=decision.output_refs,
                execution_mode=decision.execution_mode,
            )
        return outcome

    def _run_trials(
        self,
        ctx: AgentContext,
        hyp: Hypothesis,
        dataset: DatasetVersion,
        *,
        source_code: str,
        entrypoint: str,
        required_columns: tuple[str, ...],
        name_prefix: str,
        execution_mode: ExecutionMode,
        generation_id: str | None = None,
        implementation: str | None = None,
    ) -> TaskOutcome:
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
                source_code=source_code,
                entrypoint=entrypoint,
                params=params,
                hypothesis_id=hyp.id,
                name=f"{name_prefix}-{index}",
                required_columns=required_columns,
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
                    "backtest.run",
                    package_id=package.id,
                    dataset_version_id=dataset.id,
                    seed=0,
                    budget=ctx.budget,
                )
                twin = ctx.tools.call(
                    "backtest.run",
                    package_id=package.id,
                    dataset_version_id=dataset.id,
                    seed=0,
                    budget=ctx.budget,
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
            metrics.update({k: float(v) for k, v in first.get("resources", {}).items()})
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
        outputs: dict[str, Any] = {
            "experiment_ids": [e.id for e in experiments],
            "package_ids": packages,
            "candidate_package_id": candidate.strategy_package_id if candidate else None,
            "candidate_experiment_id": candidate.id if candidate else None,
            "result_hashes": [e.result_hash for e in succeeded],
            "reproducible": reproducible,
            "trial_errors": [e.error for e in experiments if e.error][:3],
            "net_sharpe_by_experiment": table,
            "execution_mode": execution_mode,
        }
        if generation_id:
            outputs["generation_id"] = generation_id
        if implementation:
            outputs["implementation"] = implementation
        return TaskOutcome(
            outputs=outputs,
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
                    execution_mode=execution_mode,
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


class ExecutionMonitoringAgent(Agent):
    role = AgentRole.EXECUTION_MONITORING

    def execute(self, ctx: AgentContext) -> TaskOutcome:
        i = ctx.task.inputs
        result: dict[str, Any] = ctx.tools.call(
            "execution.paper",
            package_id=i["package_id"],
            risk_review_id=i["risk_review_id"],
            acknowledge_id=i["acknowledge_id"],
            notional_usd=float(i.get("notional_usd", 10_000.0)),
        )
        return TaskOutcome(
            outputs=result,
            decisions=[
                DecisionRecord(
                    "paper_session",
                    str(result["session_id"]),
                    "risk_clearance_and_human_ack",
                    f"Opened paper session status={result['status']} "
                    f"equity={result['equity_usd']}.",
                    output_refs=(str(result["session_id"]),),
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
                    for t in (
                        "absent_gross_edge",
                        "cost_fragility",
                        "overfit",
                        "multiple_testing",
                        "no_signal",
                    )
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
    ExecutionMonitoringAgent,
    MetaLearningAgent,
)
