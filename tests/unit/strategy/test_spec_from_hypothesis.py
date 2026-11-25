"""StrategySpec derivation from a cycle Hypothesis for LLM codegen."""

from __future__ import annotations

from datetime import UTC, datetime

from enigma.schemas.common import AgentRole, Instrument
from enigma.schemas.governance import ExecutionMode
from enigma.schemas.research import Hypothesis
from enigma.strategy.spec import spec_from_hypothesis

NOW = datetime(2026, 3, 1, tzinfo=UTC)
BTC = Instrument(symbol="BTCUSDT", base="BTC", quote="USDT", kind="perpetual", venue="binance_usdm")


def test_spec_from_hypothesis_builds_valid_spec() -> None:
    hyp = Hypothesis(
        id="hyp_llm_1",
        objective_id="obj_1",
        family_id="fam_funding_carry",
        title="Funding Carry Edge",
        mechanism="crowded funding unwinds",
        prediction="high funding predicts negative returns",
        formal_statement="signal = -sign(funding_rate)",
        falsification_criteria=("holdout net sharpe < 0",),
        instruments=(BTC,),
        horizon="8h",
        features=("funding_rate", "close"),
        search_space={"lookback": (4, 8)},
        trial_budget=2,
        cost_model_version="cm-binance-usdm-v1",
        evidence_ids=(),
        generation_method="llm:test",
        methodology_version="qm-llm-0.1",
        created_by=AgentRole.QUANT_RESEARCH,
        created_at=NOW,
    )
    spec = spec_from_hypothesis(
        hyp,
        available_columns=("funding_rate", "close", "ts"),
        created_by=AgentRole.STRATEGY_ENGINEERING,
        created_at=NOW,
    )
    assert spec.hypothesis_id == hyp.id
    assert spec.execution_mode is ExecutionMode.LLM
    assert "funding_rate" in spec.required_columns
    assert spec.parameters
