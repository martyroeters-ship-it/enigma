"""Deterministic hypothesis generation from evidence, market measurements and prior lessons.

Method `template:mechanism-library-v1`: score each mechanism in the library against retrieved
evidence (keyword support and contradiction), measured dataset statistics, and the institution's
prior lessons; pick the best-supported untested variant; fill a mechanism-first, falsifiable
Hypothesis citing the evidence it used. An optional model-based refiner can rewrite prose fields
but cannot change parameters, trial budgets, or falsification criteria.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from enigma.core.hashing import sha256_hex
from enigma.research.mechanisms import MECHANISMS, Mechanism
from enigma.schemas.common import AgentRole, Instrument, UtcDatetime
from enigma.schemas.cycle import MarketAnalysis, NextExperimentProposal
from enigma.schemas.knowledge import EvidenceItem, FailureCondition, Lesson
from enigma.schemas.research import Hypothesis, ResearchObjective

METHOD = "template:mechanism-library-v1"
METHODOLOGY_VERSION = "qm-0.1"
FALSIFICATION = (
    "Holdout annualized net Sharpe < 0.5 or holdout net PnL <= 0 after removing the 5 best days "
    "(protocol 0.1.0-slice G03)",
    "Deflated Sharpe probability < 0.95 given every registered trial in the family (G07)",
    "Holm-adjusted p-value of mean holdout net return > 0.05 (G06)",
    "Net Sharpe <= 0 with all trading costs doubled (G12)",
)


@dataclass(frozen=True)
class MechanismScore:
    mechanism_id: str
    support: float
    contradiction: float
    data_support: float
    lesson_penalty: float
    supporting_evidence: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]
    notes: tuple[str, ...] = field(default=())

    @property
    def total(self) -> float:
        return self.support - 0.5 * self.contradiction + self.data_support - self.lesson_penalty


def _hits(text: str, keywords: Sequence[str]) -> int:
    lowered = text.lower()
    return sum(1 for k in keywords if k in lowered)


def score_mechanisms(
    evidence: Sequence[EvidenceItem],
    analysis: MarketAnalysis | None,
    lessons: Sequence[Lesson],
    failures: Sequence[FailureCondition],
    tested_mechanisms: dict[str, int],
) -> list[MechanismScore]:
    scores: list[MechanismScore] = []
    for mech in MECHANISMS.values():
        sup: list[str] = []
        con: list[str] = []
        for ev in evidence:
            text = f"{ev.excerpt} {ev.claim}"
            if _hits(text, mech.keywords) >= 2:
                sup.append(ev.id)
            if _hits(text, mech.contra_keywords) >= 1 and _hits(text, mech.keywords) >= 1:
                con.append(ev.id)
        data_support, notes = _data_support(mech, analysis)
        failed_lessons = [
            les for les in lessons if mech.id in les.topics and les.outcome == "failure"
        ]
        penalty = 1.5 * len(failed_lessons) + 0.75 * tested_mechanisms.get(mech.id, 0)
        scores.append(
            MechanismScore(
                mechanism_id=mech.id,
                support=float(len(sup)),
                contradiction=float(len(con)),
                data_support=data_support,
                lesson_penalty=penalty,
                supporting_evidence=tuple(sup),
                contradicting_evidence=tuple(con),
                notes=notes,
            )
        )
    return sorted(scores, key=lambda s: (-s.total, s.mechanism_id))


def _data_support(
    mech: Mechanism, analysis: MarketAnalysis | None
) -> tuple[float, tuple[str, ...]]:
    if analysis is None or not analysis.per_symbol:
        return 0.0, ("no market analysis available",)
    stats = list(analysis.per_symbol.values())

    def avg(key: str) -> float:
        values = [s[key] for s in stats if key in s]
        return sum(values) / len(values) if values else 0.0

    notes: list[str] = []
    score = 0.0
    if mech.id in {"funding_carry", "funding_crowding_reversal"}:
        persistence = avg("funding_autocorr_lag1")
        positive_share = avg("funding_positive_share")
        if mech.id == "funding_carry" and positive_share > 0.6:
            score += 1.0
            notes.append(f"funding positive in {positive_share:.0%} of bars")
        if persistence > 0.5:
            score += 0.5 if mech.id == "funding_carry" else 0.25
            notes.append(f"funding lag-1 autocorrelation {persistence:.2f}")
        if mech.id == "funding_crowding_reversal" and avg("funding_extreme_share") > 0.03:
            score += 0.5
            notes.append("frequent funding extremes (|z|>2)")
    if mech.id == "basis_reversion":
        half_life = avg("basis_half_life_bars")
        if 0 < half_life < 30:
            score += 1.0
            notes.append(f"basis half-life {half_life:.1f} bars")
    if mech.id == "funding_momentum" and avg("return_autocorr_lag1") > 0.02:
        score += 0.5
        notes.append("positive return autocorrelation")
    return score, tuple(notes)


def build_hypothesis(
    objective: ResearchObjective,
    mechanism: Mechanism,
    score: MechanismScore,
    instruments: Sequence[Instrument],
    cycle_id: str,
    now: UtcDatetime,
    dataset_version_ids: Sequence[str] = (),
    lesson_ids: Sequence[str] = (),
    parent: NextExperimentProposal | None = None,
    cost_model_version: str = "cm-binance-usdm-v1",
) -> Hypothesis:
    grid = dict(mechanism.param_grid)
    if parent is not None:
        for key, values in parent.hypothesis_delta.get("param_grid", {}).items():
            grid[key] = tuple(values)
    n_trials = 1
    for values in grid.values():
        n_trials *= len(values)
    trial_budget = max(1, min(objective.max_trials, n_trials))
    family_id = f"fam_{mechanism.id}"
    hyp_id = "hyp_" + sha256_hex(f"{cycle_id}|{mechanism.id}|{sorted(grid.items())}")[:26].upper()
    uncertainty = [
        f"evidence support={score.support:.0f}, contradiction={score.contradiction:.0f}, "
        f"data support={score.data_support:.2f}, lesson penalty={score.lesson_penalty:.2f}",
        *score.notes,
        "Mechanism plausibility from literature is L0 evidence; it does not imply an edge.",
    ]
    if parent is not None:
        uncertainty.append(f"Refines proposal {parent.id}: {parent.rationale}")
    return Hypothesis(
        id=hyp_id,
        objective_id=objective.id,
        family_id=family_id,
        title=mechanism.name,
        mechanism=mechanism.mechanism,
        prediction=mechanism.prediction,
        formal_statement=mechanism.formal_statement,
        falsification_criteria=FALSIFICATION,
        instruments=tuple(instruments),
        horizon=mechanism.horizon,
        features=mechanism.features,
        search_space={k: tuple(v) for k, v in sorted(grid.items())},
        trial_budget=trial_budget,
        cost_model_version=cost_model_version,
        dataset_version_ids=tuple(dataset_version_ids),
        evidence_ids=score.supporting_evidence[:25],
        contradictory_evidence_ids=score.contradicting_evidence[:25],
        lesson_ids=tuple(lesson_ids),
        parent_hypothesis_id=parent.parent_hypothesis_id if parent else None,
        uncertainty_notes=" | ".join(uncertainty),
        generation_method=METHOD,
        methodology_version=METHODOLOGY_VERSION,
        created_by=AgentRole.QUANT_RESEARCH,
        created_at=now,
    )
