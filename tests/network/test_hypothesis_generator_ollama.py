"""Live HypothesisGenerator against local Ollama (M2 quant path)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from enigma.core.clock import SystemClock
from enigma.gateway import ModelGateway, ModelRegistry, StructuredLLM
from enigma.gateway.prompts import PromptRegistry
from enigma.research.generator import HypothesisGenerator
from enigma.research.registry import HypothesisRegistry
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import Budget, ExecutionMode
from enigma.schemas.quant import (
    DatasetInfo,
    EvidenceExcerpt,
    ResearchContext,
)
from enigma.store import Stores

pytestmark = pytest.mark.network

NOW = datetime(2026, 3, 1, tzinfo=UTC)


def _ollama_7b() -> bool:
    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0)
        return r.status_code == 200 and any(
            m.get("name") == "qwen2.5:7b-instruct" for m in r.json().get("models", [])
        )
    except Exception:
        return False


def _context() -> ResearchContext:
    return ResearchContext(
        id="ctx_net",
        objective_id="obj_net",
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


@pytest.mark.skipif(not _ollama_7b(), reason="local Ollama qwen2.5:7b-instruct not available")
def test_hypothesis_generator_ollama(tmp_path: Path) -> None:
    stores = Stores.open(tmp_path / "c.sqlite3")
    gateway = ModelGateway(
        ModelRegistry.load(Path("registries/models")),
        stores.costs,
        tmp_path / "cache.sqlite3",
        SystemClock(),
    )
    gen = HypothesisGenerator(
        StructuredLLM(gateway),
        PromptRegistry.load(Path("registries/prompts")),
        model="local-qwen",
        max_output_tokens=2200,
    )
    result = gen.generate(
        _context(),
        actor=AgentRole.QUANT_RESEARCH,
        task_id="tsk_hyp_ollama",
        budget=Budget(
            max_usd=0,
            max_tokens=80_000,
            max_http_requests=20,
            max_wall_seconds=900,
            max_cpu_seconds=900,
        ),
        now=NOW,
    )
    assert result.execution_mode is ExecutionMode.LLM
    assert result.prompt_refs and result.prompt_refs[0].startswith("quant.hypothesis@")
    assert result.provenance and result.provenance[0].provider == "ollama"
    assert result.provenance[0].cost_usd == 0.0
    assert result.ok, result.failure_reason or result.rejected
    registry = HypothesisRegistry(stores.documents)
    registered = registry.register(
        hypothesis_id="hyp_net_1",
        proposal=result.proposals[0],
        context=_context(),
        generation=result,
        trial_budget=3,
        registered_by=AgentRole.QUANT_RESEARCH,
        registered_at=NOW,
    )
    assert registry.assert_unmodified(registered.id).content_hash == registered.content_hash
