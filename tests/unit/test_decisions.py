from __future__ import annotations

from pathlib import Path

import pytest

from enigma.core.errors import ProviderUnavailableError
from enigma.decisions.evaluate import Adoption, ProviderScore, adoption_decision, evaluate
from enigma.decisions.providers import (
    DeterministicDecisionProvider,
    LlmDecisionProvider,
    TypeSafeDecisionProvider,
)
from enigma.decisions.tasks import frozen_items, label_set_hash
from enigma.gateway.benchmark import ModelSummary, choose_model

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = ROOT / "registries" / "models" / "jev_m3_acceptance.yaml"


def test_frozen_hash_matches_the_pre_registered_file() -> None:
    items = frozen_items()
    assert len(items) >= 200
    text = THRESHOLDS.read_text(encoding="utf-8")
    assert label_set_hash(items) in text


def test_measured_providers_are_ranked_and_unmeasured_jev_is_rejected() -> None:
    def complete(text: str) -> str:
        word = text.split(maxsplit=1)[0]
        index = int(text.rsplit(" ", 1)[-1])
        return "no" if word == "yes" and index % 3 == 0 else word

    report = evaluate(
        (
            DeterministicDecisionProvider(),
            LlmDecisionProvider(complete, latency_ms=20.0, cost_usd=0.01),
            TypeSafeDecisionProvider(api_key=None),
        )
    )
    by_id = {row.provider_id: row for row in report.scores}
    assert by_id["deterministic"].macro_f1 == 1.0
    assert by_id["llm"].measured is True
    assert (by_id["llm"].macro_f1 or 0) < 1.0
    assert by_id["typesafe"].measured is False
    assert report.selected == "deterministic"
    decision = adoption_decision(report, THRESHOLDS)
    assert decision.adopt is False
    assert decision.reason == "typesafe_not_measured"


def test_typesafe_refuses_an_unofficial_host_and_a_missing_key() -> None:
    with pytest.raises(ValueError, match=r"api\.typesafe\.ai"):
        TypeSafeDecisionProvider(api_key="present", host="jevai.net")
    item = frozen_items()[0]
    with pytest.raises(ProviderUnavailableError, match="TYPESAFE_API_KEY"):
        TypeSafeDecisionProvider(api_key=None).decide(item)


def test_adoption_gate_can_pass_only_on_a_measured_score() -> None:
    items = frozen_items()
    digest = label_set_hash(items)
    report = evaluate((DeterministicDecisionProvider(),), items)
    passed = adoption_decision(
        report.__class__(
            label_set_hash=digest,
            n_items=len(items),
            scores=(
                *report.scores,
                ProviderScore(
                    provider_id="typesafe",
                    measured=True,
                    macro_f1=1.0,
                    latency_ms=10.0,
                    cost_usd=0.0,
                    errors=0,
                ),
            ),
            selected="typesafe",
        ),
        THRESHOLDS,
    )
    assert passed == Adoption(True, "thresholds_met")


def test_model_choice_uses_quality_then_latency_then_cost() -> None:
    local = ModelSummary(
        model_alias="fixture-local",
        provider="fixture",
        tasks_scored=4,
        mean_score=0.8,
        total_cost_usd=0.0,
        mean_latency_ms=40.0,
    )
    frontier = ModelSummary(
        model_alias="fixture-frontier",
        provider="fixture",
        tasks_scored=4,
        mean_score=0.9,
        total_cost_usd=0.02,
        mean_latency_ms=400.0,
    )
    assert choose_model((local, frontier)).model_alias == "fixture-frontier"  # type: ignore[union-attr]
    tied = frontier.model_copy(update={"mean_score": 0.8, "mean_latency_ms": 10.0})
    assert choose_model((local, tied)).model_alias == "fixture-frontier"  # type: ignore[union-attr]
    cheaper = tied.model_copy(update={"mean_latency_ms": 40.0, "total_cost_usd": 0.0})
    dearer = local.model_copy(update={"total_cost_usd": 0.05})
    assert choose_model((dearer, cheaper)).model_alias == "fixture-frontier"  # type: ignore[union-attr]
    assert choose_model(()) is None
    unscored = local.model_copy(update={"mean_score": None, "tasks_scored": 0})
    assert choose_model((unscored,)) is None
