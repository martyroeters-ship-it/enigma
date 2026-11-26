"""Chained live path: HypothesisGenerator → StrategySpec → StrategyCodeGenerator."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from enigma.core.clock import SystemClock
from enigma.gateway import ModelGateway, ModelRegistry, StructuredLLM
from enigma.gateway.prompts import PromptRegistry
from enigma.research.bridge import hypothesis_from_proposal
from enigma.research.generator import HypothesisGenerator
from enigma.schemas.common import AgentRole, Instrument
from enigma.schemas.governance import Budget, ExecutionMode
from enigma.schemas.quant import DatasetInfo, EvidenceExcerpt, ResearchContext
from enigma.schemas.research import CostModel
from enigma.store import Stores
from enigma.strategy.codegen import StrategyCodeGenerator
from enigma.strategy.spec import spec_from_hypothesis

pytestmark = pytest.mark.network

NOW = datetime(2026, 3, 1, tzinfo=UTC)
BTC = Instrument(symbol="BTCUSDT", base="BTC", quote="USDT", kind="perpetual", venue="binance_usdm")


def _ollama_7b() -> bool:
    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0)
        return r.status_code == 200 and any(
            m.get("name") == "qwen2.5:7b-instruct" for m in r.json().get("models", [])
        )
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_7b(), reason="local Ollama qwen2.5:7b-instruct not available")
def test_hypothesis_then_codegen_ollama(tmp_path: Path) -> None:
    stores = Stores.open(tmp_path / "c.sqlite3")
    gateway = ModelGateway(
        ModelRegistry.load(Path("registries/models")),
        stores.costs,
        tmp_path / "cache.sqlite3",
        SystemClock(),
    )
    llm = StructuredLLM(gateway)
    prompts = PromptRegistry.load(Path("registries/prompts"))
    budget = Budget(
        max_usd=0,
        max_tokens=200_000,
        max_http_requests=40,
        max_wall_seconds=1800,
        max_cpu_seconds=1800,
    )
    context = ResearchContext(
        id="ctx_chain",
        objective_id="obj_chain",
        objective="Find a cost-aware funding edge on BTCUSDT perps",
        research_cutoff=NOW,
        instruments=("BTCUSDT",),
        cost_model_version="cm-binance-usdm-v1",
        cost_bps_per_unit_turnover=2.0,
        evidence=(
            EvidenceExcerpt(
                id="ev_1",
                source_id="src_1",
                text="Funding stayed positive for weeks in the 2021 bull market.",
                claim_level="L1",
                topics=("funding",),
            ),
            EvidenceExcerpt(
                id="ev_2",
                source_id="src_2",
                text="Carry collapsed after cascading liquidations in 2022.",
                claim_level="L1",
                topics=("funding",),
            ),
            EvidenceExcerpt(
                id="ev_3",
                source_id="src_3",
                text="Ignoring costs makes funding carry look profitable in-sample only.",
                claim_level="L1",
                topics=("funding", "costs"),
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
    hyp_result = HypothesisGenerator(
        llm, prompts, model="local-qwen", max_output_tokens=1800
    ).generate(
        context,
        actor=AgentRole.QUANT_RESEARCH,
        task_id="tsk_chain_hyp",
        budget=budget,
        now=NOW,
    )
    assert hyp_result.ok and hyp_result.execution_mode is ExecutionMode.LLM
    proposal = hyp_result.proposals[0]
    hyp = hypothesis_from_proposal(
        proposal,
        objective_id="obj_chain",
        instruments=(BTC,),
        generation=hyp_result,
        dataset_version_ids=("ds_1",),
        created_by=AgentRole.QUANT_RESEARCH,
        created_at=NOW,
        cycle_id="cyc_chain",
    )
    spec = spec_from_hypothesis(
        hyp,
        available_columns=("funding_rate", "close", "basis"),
        created_by=AgentRole.STRATEGY_ENGINEERING,
        created_at=NOW,
    )
    cost = CostModel(
        version="cm-binance-usdm-v1",
        taker_fee_bps=4.0,
        maker_fee_bps=2.0,
        half_spread_bps=1.0,
        impact_bps_per_million=0.5,
        capacity_notional_usd=1_000_000.0,
    )
    codegen = StrategyCodeGenerator(
        llm, prompts, model="local-qwen", max_output_tokens=1600, max_attempts=3
    )
    record = codegen.generate(
        spec,
        cost_model=cost,
        actor=AgentRole.STRATEGY_ENGINEERING,
        task_id="tsk_chain_code",
        budget=budget,
        now=NOW,
    )
    assert record.ok, (record.failure_reason, record.attempts)
    assert record.execution_mode is ExecutionMode.LLM
    assert codegen.last_package is not None
