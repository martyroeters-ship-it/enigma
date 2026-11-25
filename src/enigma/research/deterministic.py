"""Explicit deterministic hypothesis path (M1 mechanism library). Never used as silent fallback."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from enigma.core.ids import new_id
from enigma.research.hypothesis import (
    METHOD as LEGACY_METHOD,
)
from enigma.research.hypothesis import (
    MechanismScore,
    build_hypothesis,
    score_mechanisms,
)
from enigma.research.mechanisms import MECHANISMS
from enigma.schemas.common import AgentRole, Instrument
from enigma.schemas.cycle import MarketAnalysis, NextExperimentProposal
from enigma.schemas.governance import ExecutionMode
from enigma.schemas.knowledge import EvidenceItem, FailureCondition, Lesson
from enigma.schemas.quant import GenerationResult, ResearchContext
from enigma.schemas.research import Hypothesis, ResearchObjective


class DeterministicHypothesisSource:
    """Labelled M1 path (`execution_mode=deterministic_fallback`).

    Produces a legacy `Hypothesis` via the mechanism library. The accompanying
    `GenerationResult` carries no `HypothesisProposal`s — coordinators that need the new
    typed proposal path must call `HypothesisGenerator` instead.
    """

    def propose_legacy(
        self,
        *,
        objective: ResearchObjective,
        evidence: Sequence[EvidenceItem],
        analysis: MarketAnalysis | None,
        lessons: Sequence[Lesson],
        failures: Sequence[FailureCondition],
        tested_mechanisms: dict[str, int],
        instruments: Sequence[Instrument],
        cycle_id: str,
        now: datetime,
        dataset_version_ids: Sequence[str] = (),
        lesson_ids: Sequence[str] = (),
        parent: NextExperimentProposal | None = None,
        cost_model_version: str = "cm-binance-usdm-v1",
    ) -> tuple[Hypothesis, MechanismScore]:
        scores = score_mechanisms(evidence, analysis, lessons, failures, tested_mechanisms)
        if not scores:
            raise RuntimeError("mechanism library is empty")
        best = scores[0]
        mechanism = MECHANISMS[best.mechanism_id]
        hypothesis = build_hypothesis(
            objective=objective,
            mechanism=mechanism,
            score=best,
            instruments=instruments,
            cycle_id=cycle_id,
            now=now,
            dataset_version_ids=dataset_version_ids,
            lesson_ids=lesson_ids,
            parent=parent,
            cost_model_version=cost_model_version,
        )
        return hypothesis, best

    def record_fallback(
        self,
        context: ResearchContext,
        hypothesis: Hypothesis,
        *,
        now: datetime,
        actor: AgentRole = AgentRole.QUANT_RESEARCH,
    ) -> GenerationResult:
        return GenerationResult(
            id=new_id("hgen"),
            context_id=context.id,
            context_hash=context.content_hash(),
            execution_mode=ExecutionMode.DETERMINISTIC_FALLBACK,
            method=LEGACY_METHOD,
            prompt_refs=(),
            provenance=(),
            proposals=(),
            rejected=(),
            repair_rounds=0,
            failure_reason=f"deterministic_fallback_legacy_hypothesis:{hypothesis.id}",
            created_by=actor,
            created_at=now,
        )
