"""Build a sealed `ResearchContext` from institutional documents (lessons, evidence, data)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from enigma.core.ids import new_id
from enigma.schemas.knowledge import EvidenceItem, FailureCondition, Lesson
from enigma.schemas.quant import (
    DatasetInfo,
    DomainGap,
    EvidenceExcerpt,
    FailureNote,
    LessonNote,
    PriorHypothesisOutcome,
    ResearchContext,
    StatObservation,
)
from enigma.schemas.research import Hypothesis, ResearchObjective

_BARS = re.compile(r"(\d+)\s*bar", re.I)


def evidence_excerpt(item: EvidenceItem) -> EvidenceExcerpt:
    return EvidenceExcerpt(
        id=item.id,
        source_id=item.source_id,
        text=item.excerpt,
        claim_level=item.claim_level,
        topics=tuple(item.topics),
    )


def lesson_note(lesson: Lesson) -> LessonNote:
    return LessonNote(
        id=lesson.id,
        statement=lesson.statement,
        outcome=str(lesson.outcome),
        topics=tuple(lesson.topics),
    )


def failure_note(failure: FailureCondition) -> FailureNote:
    return FailureNote(
        id=failure.id,
        kind=failure.kind,
        description=failure.description,
        gate_id=failure.gate_id,
    )


def _horizon_bars(hyp: Hypothesis) -> int:
    match = _BARS.search(hyp.horizon or "")
    return int(match.group(1)) if match else 1


def prior_from_hypothesis(
    hyp: Hypothesis, *, validation: Literal["pass", "fail", "untested"] = "untested"
) -> PriorHypothesisOutcome:
    return PriorHypothesisOutcome(
        hypothesis_id=hyp.id,
        family_id=hyp.family_id,
        title=hyp.title,
        mechanism=hyp.mechanism,
        signal=hyp.formal_statement,
        columns=tuple(hyp.features),
        horizon_bars=_horizon_bars(hyp),
        generation_method=hyp.generation_method,
        validation=validation,
        failed_gates=(),
        trials=0,
    )


def build_research_context(
    *,
    objective: ResearchObjective,
    research_cutoff: datetime,
    datasets: Sequence[DatasetInfo],
    evidence: Sequence[EvidenceItem] = (),
    lessons: Sequence[Lesson] = (),
    failures: Sequence[FailureCondition] = (),
    prior_hypotheses: Sequence[Hypothesis] = (),
    observations: Sequence[StatObservation] = (),
    underexplored: Sequence[DomainGap] = (),
    cost_model_version: str = "cm-binance-usdm-v1",
    cost_bps_per_unit_turnover: float = 2.0,
    context_id: str | None = None,
) -> ResearchContext:
    """Seal everything the LLM hypothesis generator may cite (including lessons).

    Lessons and prior outcomes are first-class so meta-learning feeds the next cycle without
    leaking validation-plane holdout detail.
    """
    return ResearchContext(
        id=context_id or new_id("ctx"),
        objective_id=objective.id,
        objective=objective.statement,
        research_cutoff=research_cutoff,
        instruments=tuple(str(i) for i in objective.instruments),
        cost_model_version=cost_model_version,
        cost_bps_per_unit_turnover=cost_bps_per_unit_turnover,
        evidence=tuple(evidence_excerpt(e) for e in evidence),
        prior_hypotheses=tuple(prior_from_hypothesis(h) for h in prior_hypotheses),
        lessons=tuple(lesson_note(x) for x in lessons),
        failures=tuple(failure_note(f) for f in failures),
        datasets=tuple(datasets),
        observations=tuple(observations),
        underexplored=tuple(underexplored),
    )
