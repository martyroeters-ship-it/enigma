from __future__ import annotations

from enigma.orchestrator.cio import (
    Candidate,
    FamilyHistory,
    PrioritizationInputs,
    ResearchPrioritizer,
    allocate,
    information_gain,
    novelty,
)
from enigma.schemas.common import AgentRole


def _inputs(**overrides: object) -> PrioritizationInputs:
    base: dict[str, object] = {
        "history": {
            "basis_reversion": FamilyHistory("basis_reversion", cycles=3, passes=0, fails=3),
        },
        "prior_hypothesis_texts": ["Perp-spot basis mean reversion pulls the perp back to spot"],
        "available_datasets": frozenset({"binance_perp_funding_8h"}),
        "agent_success_rates": {AgentRole.QUANT_RESEARCH: 1.0},
        "remaining_budget": 10.0,
    }
    base.update(overrides)
    return PrioritizationInputs(**base)  # type: ignore[arg-type]


def test_information_gain_shrinks_with_repeated_consistent_outcomes() -> None:
    assert information_gain(None) == 1.0
    assert information_gain(FamilyHistory("f", fails=3)) < information_gain(
        FamilyHistory("f", fails=1)
    )


def test_novelty_penalises_near_duplicates() -> None:
    prior = ["perp spot basis mean reversion"]
    assert novelty("perp spot basis mean reversion", prior) == 0.0
    assert novelty("open interest liquidation cascade momentum", prior) == 1.0


def test_untested_novel_direction_outranks_repeatedly_failed_family() -> None:
    repeat = Candidate("c1", "next_experiment", "basis_reversion", "basis mean reversion perp spot")
    fresh = Candidate(
        "c2", "gap", "oi_liquidation", "open interest build-up precedes liquidation cascades"
    )
    agenda = ResearchPrioritizer().rank([repeat, fresh], _inputs())
    assert agenda.top is not None and agenda.top.candidate.id == "c2"
    assert set(agenda.top.components) >= {"information_gain", "novelty", "cost"}


def test_missing_data_and_guidance_penalty_lower_the_score() -> None:
    needs_data = Candidate(
        "c3", "gap", "trades", "trade imbalance", required_datasets=("aggtrades",)
    )
    ready = Candidate("c4", "gap", "trades2", "trade imbalance two")
    agenda = ResearchPrioritizer().rank([needs_data, ready], _inputs())
    assert agenda.ranked[0].candidate.id == "c4"

    def penalty(c: Candidate) -> tuple[float, tuple[str, ...]]:
        return (1.0, ("family failed G12 repeatedly",)) if c.family == "trades2" else (0.0, ())

    agenda = ResearchPrioritizer().rank([needs_data, ready], _inputs(guidance_penalty=penalty))
    assert agenda.ranked[0].candidate.id == "c3"
    penalised = next(s for s in agenda.ranked if s.candidate.id == "c4")
    assert "family failed G12 repeatedly" in penalised.reasons


def test_exhausted_holdout_deprioritises_research_but_not_technology() -> None:
    research = Candidate("r", "gap", "x", "new research direction")
    tech = Candidate("t", "technology", "retrieval", "bm25 retrieval prototype")
    agenda = ResearchPrioritizer().rank([research, tech], _inputs(holdout_available=False))
    assert agenda.top is not None and agenda.top.candidate.id == "t"
    assert agenda.notes


def test_allocation_reserves_a_technology_share() -> None:
    cands = [Candidate(f"r{i}", "gap", f"f{i}", f"direction number {i}") for i in range(5)]
    cands.append(Candidate("t", "technology", "tech", "prototype benchmark"))
    agenda = ResearchPrioritizer().rank(cands, _inputs())
    chosen = allocate(agenda, capacity=5, technology_share=0.2)
    assert len(chosen) == 5
    assert any(s.candidate.kind == "technology" for s in chosen)
    assert allocate(agenda, capacity=0) == ()


def test_unaffordable_candidate_is_not_viable() -> None:
    pricey = Candidate("p", "gap", "f", "expensive direction", expected_cost=100.0)
    agenda = ResearchPrioritizer().rank([pricey], _inputs())
    assert not agenda.ranked[0].viable
    assert allocate(agenda, capacity=1) == ()


def test_exactly_affordable_candidate_remains_viable() -> None:
    one_slot = Candidate("c", "next_experiment", "f", "one more cycle", expected_cost=1.0)
    agenda = ResearchPrioritizer().rank([one_slot], _inputs(remaining_budget=1.0))
    assert agenda.ranked[0].viable
    assert allocate(agenda, capacity=1)[0].candidate.id == "c"
