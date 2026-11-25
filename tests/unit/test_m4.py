from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from enigma.core.hashing import sha256_of
from enigma.gateway.benchmark import ModelSummary
from enigma.learning.adoption import may_roll_out, measured_lift
from enigma.orchestrator.tech import selection_from_benchmark
from enigma.research.narrative_study import association, scramble
from enigma.schemas.common import AgentRole, ClaimLevel
from enigma.schemas.governance import ImprovementProposal, ProposalStatus, ProposalTarget
from enigma.schemas.narrative import NarrativeEvent
from enigma.schemas.prediction import PredictionMarketSnapshot

START = datetime(2026, 11, 1, tzinfo=UTC)
NOW = datetime(2026, 11, 25, tzinfo=UTC)


def _event(index: int, positive: bool) -> NarrativeEvent:
    stamp = START + timedelta(hours=index)
    title = "positive funding" if positive else "negative funding"
    return NarrativeEvent(
        id=f"nre_{index:02d}",
        source_url=f"https://example.test/{index}",
        title=title,
        excerpt=title,
        event_time=stamp,
        available_at=stamp,
        content_hash=sha256_of({"title": title, "index": index}),
    )


def test_shuffled_narrative_times_weaken_the_association() -> None:
    events = tuple(_event(index, index < 8) for index in range(16))
    bars = [START + timedelta(hours=index) for index in range(16)]
    # Sign of events known at each bar: early bars see only positive titles.
    signed = []
    for bar in bars:
        known = [event for event in events if event.available_at <= bar]
        signed.append(sum(1.0 if event.title.startswith("positive") else -1.0 for event in known))
    aligned = association(events, bars, signed)
    scrambled = association(scramble(events, seed=1), bars, signed)
    assert aligned > 0.9
    assert abs(scrambled) < abs(aligned) - 0.2


def test_prediction_snapshot_is_an_l0_prior() -> None:
    snap = PredictionMarketSnapshot(
        id="pms_fixture",
        market_id="macro-1",
        question="Will the fixture resolve yes?",
        probability=0.42,
        volume_usd=10.0,
        available_at=NOW,
        source="fixture",
    )
    assert snap.claim_level is ClaimLevel.L0_UNVERIFIED
    assert "Not a validation result" in snap.research_note()
    with pytest.raises(ValidationError):
        PredictionMarketSnapshot.model_validate(
            {
                "id": "pms_bad",
                "market_id": "macro-1",
                "question": "order",
                "probability": 0.5,
                "volume_usd": 1,
                "available_at": NOW.isoformat(),
                "source": "fixture",
                "order_side": "buy",
            }
        )


def test_benchmark_selection_does_not_adopt() -> None:
    report = {
        "task_set_hash": "sha256:abc",
        "summary": [
            ModelSummary(
                model_alias="fixture-local",
                provider="fixture",
                tasks_scored=4,
                mean_score=0.4,
                total_cost_usd=0.0,
                mean_latency_ms=30.0,
            ).model_dump(),
            ModelSummary(
                model_alias="fixture-frontier",
                provider="fixture",
                tasks_scored=4,
                mean_score=0.7,
                total_cost_usd=0.02,
                mean_latency_ms=200.0,
            ).model_dump(),
        ],
    }
    choice = selection_from_benchmark(report)
    assert choice["selected_model"] == "fixture-frontier"
    assert choice["adoption_allowed"] is False


def test_rollout_requires_positive_lift_and_citations() -> None:
    assert measured_lift(0.2, 0.5) == pytest.approx(0.3)
    assert may_roll_out(0.0, postmortem_ids=("pm_1",), benchmark_id="bench_1") is False
    assert may_roll_out(0.3, postmortem_ids=(), benchmark_id="bench_1") is False
    assert may_roll_out(0.3, postmortem_ids=("pm_1",), benchmark_id=None) is False
    common = dict(
        id="imp_m4",
        target=ProposalTarget.WORKFLOW,
        target_ref="narrative",
        change_summary="keep the aligned feature",
        rationale="the scramble killed the association",
        success_metric="holdout_correlation",
        proposed_by=AgentRole.META_LEARNING,
        created_at=NOW,
        postmortem_ids=("pm_1",),
        benchmark_id="bench_1",
    )
    with pytest.raises(ValidationError, match="positive measured lift"):
        ImprovementProposal(**common, status=ProposalStatus.ROLLED_OUT, measured_lift=0.0)
    rolled = ImprovementProposal(**common, status=ProposalStatus.ROLLED_OUT, measured_lift=0.3)
    assert rolled.measured_lift == pytest.approx(0.3)
