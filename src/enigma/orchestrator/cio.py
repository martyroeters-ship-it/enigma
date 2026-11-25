"""CIO research prioritization: rank candidate research directions and allocate budget.

The CIO chooses *what to research next*; it has no tools that touch validation, risk or
security policy, so a high priority can never override a gate. Every ranking is fully
explained by its score components, which are recorded with the CIO's decision.

Scoring (all components in [0, 1]):

- **information gain**: posterior variance of the family's success probability under a
  Beta(1 + passes, 1 + fails) prior, normalised by its maximum (1/12 for Beta(1, 1)). Families
  tested many times with consistent outcomes have little left to learn.
- **novelty**: 1 - the highest token-set Jaccard similarity to any prior hypothesis text.
- **data readiness**: share of required datasets already available; missing ones need an
  acquisition proposal first.
- **agent reliability**: mean task success rate of the roles the candidate needs.
- **cost**: expected spend relative to the remaining budget (higher is worse).
- **guidance penalty**: learned constraints from the meta-learning agent (higher is worse).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from enigma.schemas.common import AgentRole

_TOKEN = re.compile(r"[a-z0-9]+")
_BETA_UNIFORM_VARIANCE = 1.0 / 12.0


@dataclass(frozen=True)
class FamilyHistory:
    family: str
    cycles: int = 0
    trials: int = 0
    passes: int = 0
    fails: int = 0
    aborted: int = 0


@dataclass(frozen=True)
class Candidate:
    id: str
    kind: str
    family: str
    summary: str
    required_datasets: tuple[str, ...] = ()
    roles: tuple[AgentRole, ...] = ()
    expected_cost: float = 1.0
    proposal_id: str | None = None
    refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class PriorityWeights:
    information_gain: float = 0.35
    novelty: float = 0.25
    data_readiness: float = 0.15
    agent_reliability: float = 0.10
    cost: float = 0.15
    guidance: float = 0.30


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    score: float
    components: dict[str, float]
    reasons: tuple[str, ...] = ()
    viable: bool = True


@dataclass(frozen=True)
class PrioritizationInputs:
    history: Mapping[str, FamilyHistory]
    prior_hypothesis_texts: Sequence[str]
    available_datasets: frozenset[str]
    agent_success_rates: Mapping[AgentRole, float]
    remaining_budget: float
    guidance_penalty: Callable[[Candidate], tuple[float, tuple[str, ...]]] | None = None
    holdout_available: bool = True


@dataclass(frozen=True)
class Agenda:
    ranked: tuple[ScoredCandidate, ...]
    weights: PriorityWeights
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def top(self) -> ScoredCandidate | None:
        return self.ranked[0] if self.ranked else None


def information_gain(history: FamilyHistory | None) -> float:
    passes = history.passes if history else 0
    fails = history.fails if history else 0
    a, b = 1.0 + passes, 1.0 + fails
    variance = a * b / ((a + b) ** 2 * (a + b + 1.0))
    return min(1.0, variance / _BETA_UNIFORM_VARIANCE)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in _TOKEN.findall(text.lower()) if len(t) > 2)


def novelty(summary: str, prior_texts: Sequence[str]) -> float:
    mine = _tokens(summary)
    if not mine:
        return 0.0
    best = 0.0
    for text in prior_texts:
        other = _tokens(text)
        if other:
            best = max(best, len(mine & other) / len(mine | other))
    return 1.0 - best


class ResearchPrioritizer:
    def __init__(self, weights: PriorityWeights | None = None) -> None:
        self.weights = weights or PriorityWeights()

    def score(self, candidate: Candidate, inputs: PrioritizationInputs) -> ScoredCandidate:
        w = self.weights
        history = inputs.history.get(candidate.family)
        required = candidate.required_datasets
        ready = (
            sum(1 for d in required if d in inputs.available_datasets) / len(required)
            if required
            else 1.0
        )
        rates = [inputs.agent_success_rates.get(r, 1.0) for r in candidate.roles]
        reliability = sum(rates) / len(rates) if rates else 1.0
        budget = max(inputs.remaining_budget, 0.0)
        # Fraction of the remaining budget this candidate would consume. Exactly affordable
        # (cost == remaining) stays viable; only an overspend is refused.
        cost = 1.0 if budget <= 0 else min(1.0, candidate.expected_cost / budget)
        unaffordable = candidate.expected_cost > budget
        penalty: float = 0.0
        reasons: tuple[str, ...] = ()
        if inputs.guidance_penalty is not None:
            penalty, reasons = inputs.guidance_penalty(candidate)
            penalty = min(1.0, max(0.0, penalty))
        components = {
            "information_gain": information_gain(history),
            "novelty": novelty(candidate.summary, inputs.prior_hypothesis_texts),
            "data_readiness": ready,
            "agent_reliability": reliability,
            "cost": cost,
            "guidance_penalty": penalty,
        }
        score = (
            w.information_gain * components["information_gain"]
            + w.novelty * components["novelty"]
            + w.data_readiness * components["data_readiness"]
            + w.agent_reliability * components["agent_reliability"]
            - w.cost * components["cost"]
            - w.guidance * components["guidance_penalty"]
        )
        notes = list(reasons)
        viable = True
        if not inputs.holdout_available and candidate.kind != "technology":
            viable = False
            notes.append("no holdout evaluation budget available: final validation impossible")
        if unaffordable:
            viable = False
            notes.append("expected cost exceeds the remaining budget")
        return ScoredCandidate(candidate, round(score, 6), components, tuple(notes), viable)

    def rank(self, candidates: Sequence[Candidate], inputs: PrioritizationInputs) -> Agenda:
        scored = [self.score(c, inputs) for c in candidates]
        scored.sort(key=lambda s: (not s.viable, -s.score, s.candidate.id))
        notes: list[str] = []
        if not inputs.holdout_available:
            notes.append("holdout evaluation budget exhausted; only technology work is useful")
        return Agenda(tuple(scored), self.weights, tuple(notes))


def allocate(
    agenda: Agenda, capacity: int, technology_share: float = 0.2
) -> tuple[ScoredCandidate, ...]:
    """Pick up to `capacity` items, reserving a share for technology improvement work."""
    if capacity <= 0:
        return ()
    viable = [s for s in agenda.ranked if s.viable]
    tech = [s for s in viable if s.candidate.kind == "technology"]
    research = [s for s in viable if s.candidate.kind != "technology"]
    tech_slots = min(len(tech), math.floor(capacity * technology_share)) if research else len(tech)
    chosen = tech[:tech_slots] + research[: capacity - tech_slots]
    if len(chosen) < capacity:
        chosen += [s for s in viable if s not in chosen][: capacity - len(chosen)]
    chosen.sort(key=lambda s: (-s.score, s.candidate.id))
    return tuple(chosen[:capacity])
