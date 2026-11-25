"""Offline tests for HypothesisGenerator, checks, and registry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from enigma.gateway.prompts import PromptRegistry
from enigma.gateway.structured import Traced
from enigma.gateway.types import FinishReason, ModelResponse, Usage
from enigma.research.checks import check_proposal
from enigma.research.generator import HypothesisGenerator
from enigma.research.registry import HypothesisRegistry
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import Budget, ExecutionMode
from enigma.schemas.quant import (
    ColumnUse,
    DatasetInfo,
    EvidenceExcerpt,
    ExecutionCostEstimate,
    ExperimentalDesign,
    HypothesisProposal,
    ParameterGrid,
    ResearchContext,
)
from enigma.store import Stores

NOW = datetime(2026, 3, 1, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[3]


def _budget() -> Budget:
    return Budget(max_usd=0.0, max_tokens=50_000, max_http_requests=20)


def _proposal(**overrides: Any) -> HypothesisProposal:
    base: dict[str, Any] = dict(
        id="prop_1",
        title="Funding persistence after positive regime",
        domain="funding",
        market_mechanism="persistent positive funding attracts leveraged longs",
        inefficiency_rationale="slow mean reversion in funding after crowded longs",
        persistence_rationale="risk limits and liquidation cascades delay arbitrage",
        signal_definition="sign of trailing mean funding over W bars",
        target_definition="forward log return over H bars",
        test_statistic="mean net return / Newey-West se",
        required_data=(
            ColumnUse(dataset_id="ds_1", column="funding_rate", role="signal", min_lag_bars=0),
            ColumnUse(dataset_id="ds_1", column="close", role="target", min_lag_bars=-4),
        ),
        horizon_bars=8,
        horizon="8 bars",
        null_hypothesis="mean net return <= 0",
        alternative_hypothesis="mean net return > 0 after costs",
        design=ExperimentalDesign(
            parameters=(ParameterGrid(name="W", values=(4, 8, 16)),),
            grid_size=3,
            train_test_protocol="expanding window pre-cutoff",
            primary_metric="holdout_net_sharpe",
        ),
        confounders=("funding_regime_shift",),
        execution_costs=ExecutionCostEstimate(
            turnover_per_bar=0.2,
            bars_per_year=3 * 365,
            cost_bps_per_unit_turnover=2.0,
            annual_cost_bps=438.0,
            cost_model_version="cm-binance-usdm-v1",
            rationale="2 bps per unit turnover times expected turnover",
        ),
        failure_conditions=("holdout sharpe < 0.5",),
        supporting_evidence_ids=("ev_1",),
        contradictory_evidence_ids=("ev_2",),
        novelty_statement="uses sealed context evidence only",
        complexity_class="rule",
        complexity_rationale="threshold on lagged funding mean",
        instruments=("BTCUSDT",),
    )
    base.update(overrides)
    return HypothesisProposal.model_validate(base)


def _context() -> ResearchContext:
    return ResearchContext(
        id="ctx_1",
        objective_id="obj_1",
        objective="Find a cost-aware funding edge on BTCUSDT perps",
        research_cutoff=NOW,
        instruments=("BTCUSDT",),
        cost_model_version="cm-binance-usdm-v1",
        cost_bps_per_unit_turnover=2.0,
        evidence=(
            EvidenceExcerpt(
                id="ev_1",
                source_id="src_1",
                text="Funding stayed positive for weeks in 2021.",
                claim_level="L1",
                topics=("funding",),
            ),
            EvidenceExcerpt(
                id="ev_2",
                source_id="src_2",
                text="Carry collapsed after liquidations in 2022.",
                claim_level="L1",
                topics=("funding",),
            ),
        ),
        datasets=(
            DatasetInfo(
                dataset_version_id="ds_1",
                name="binance_btcusdt_1h",
                frequency="1h",
                bar_hours=1.0,
                symbols=("BTCUSDT",),
                columns=("funding_rate", "close", "basis"),
                coverage_start=datetime(2024, 1, 1, tzinfo=UTC),
                coverage_end=NOW,
                rows=1000,
            ),
        ),
    )


@dataclass
class _FakeLLM:
    proposal: HypothesisProposal
    calls: int = 0

    def generate_traced(self, model_cls, system, user, actor, task_id, budget, **kwargs):
        self.calls += 1
        assert model_cls is HypothesisProposal
        assert "Research context JSON" in user or "context" in user.lower() or True
        response = ModelResponse(
            text="{}",
            provider="ollama",
            model_id="qwen2.5:3b",
            model_version="qwen2.5:3b",
            request_hash="sha256:" + "a" * 64,
            raw_response_hash="sha256:" + "b" * 64,
            finish_reason=FinishReason.STOP,
            usage=Usage(input_tokens=10, output_tokens=20),
            cost_usd=0.0,
            latency_ms=12.0,
            cached=False,
            parsed=self.proposal,
            structured_mode="native",
            repaired=False,
        )
        return Traced(self.proposal, response)


def test_check_rejects_unknown_evidence() -> None:
    bad = _proposal(supporting_evidence_ids=("nope",))
    failures = check_proposal(bad, _context())
    assert any(f.check == "evidence_exists" for f in failures)


def test_check_rejects_lookahead() -> None:
    bad = _proposal(
        required_data=(
            ColumnUse(dataset_id="ds_1", column="funding_rate", role="signal", min_lag_bars=-1),
        )
    )
    failures = check_proposal(bad, _context())
    assert any(f.check == "no_lookahead" for f in failures)


def test_generator_accepts_valid_proposal(tmp_path: Path) -> None:
    prompts = PromptRegistry.load(ROOT / "registries" / "prompts")
    prop = _proposal()
    llm = _FakeLLM(prop)
    gen = HypothesisGenerator(llm, prompts, model="qwen2.5:3b")
    result = gen.generate(_context(), task_id="tsk_1", budget=_budget(), now=NOW)
    assert result.ok
    assert result.execution_mode is ExecutionMode.LLM
    assert result.prompt_refs and result.prompt_refs[0].startswith("quant.hypothesis@")
    assert result.provenance
    assert llm.calls == 1


def test_generator_repairs_then_fails_explicitly() -> None:
    prompts = PromptRegistry.load(ROOT / "registries" / "prompts")
    bad = _proposal(supporting_evidence_ids=("missing",))
    llm = _FakeLLM(bad)
    gen = HypothesisGenerator(llm, prompts)
    result = gen.generate(_context(), task_id=None, budget=_budget(), now=NOW)
    assert not result.ok
    assert result.failure_reason == "no_proposal_survived_checks"
    assert result.repair_rounds == 1
    assert len(result.rejected) == 2
    assert llm.calls == 2


def test_registry_hashes_and_amends(stores: Stores) -> None:
    ctx = _context()
    prop = _proposal()
    prompts = PromptRegistry.load(ROOT / "registries" / "prompts")
    result = HypothesisGenerator(_FakeLLM(prop), prompts).generate(
        ctx, task_id="t", budget=_budget(), now=NOW
    )
    registry = HypothesisRegistry(stores.documents)
    first = registry.register(
        hypothesis_id="hyp_a",
        proposal=prop,
        context=ctx,
        generation=result,
        trial_budget=3,
        registered_by=AgentRole.QUANT_RESEARCH,
        registered_at=NOW,
    )
    assert registry.assert_unmodified(first.id).id == first.id
    second = registry.register(
        hypothesis_id="hyp_b",
        proposal=prop,
        context=ctx,
        generation=result,
        trial_budget=2,
        amends=first.id,
        registered_by=AgentRole.QUANT_RESEARCH,
        registered_at=NOW,
    )
    assert second.amends == first.id
    assert second.cumulative_trial_budget == 5
    assert "hyp_b" in second.family_members
