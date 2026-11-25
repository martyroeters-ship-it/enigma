"""Live StrategyCodeGenerator against local Ollama (sandbox + truncation)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pytest

from enigma.core.clock import SystemClock
from enigma.core.ids import new_id
from enigma.gateway import ModelGateway, ModelRegistry, StructuredLLM
from enigma.gateway.prompts import PromptRegistry
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import Budget, ExecutionMode
from enigma.schemas.quant import BaselineSpec, FeatureDef, ParameterGrid, StrategySpec
from enigma.schemas.research import CostModel
from enigma.store import Stores
from enigma.strategy import (
    SandboxJob,
    SandboxRunner,
    sample_checkpoints,
    truncation_invariance_check,
)
from enigma.strategy.codegen import StrategyCodeGenerator

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


def _spec() -> StrategySpec:
    features = (
        FeatureDef(
            name="funding_signal",
            formula="funding_rate",
            inputs=("funding_rate",),
            lookback_bars=1,
            min_lag_bars=0,
        ),
    )
    baseline = BaselineSpec(
        name="sign_funding",
        signal_math="sign(funding_rate)",
        features=features,
        parameters=(),
        position_rule="clip(sign(funding_rate), -1, 1)",
    )
    data = {
        "id": new_id("sspec"),
        "hypothesis_spec_id": "rhs_net",
        "hypothesis_id": "hyp_net",
        "name": "funding_sign_v1",
        "signal_math": "sign of trailing mean funding over W bars",
        "features": [f.model_dump(mode="json") for f in features],
        "parameters": [ParameterGrid(name="W", values=(4, 8)).model_dump(mode="json")],
        "position_rule": "clip(signal, -1, 1)",
        "position_bounds": (-1.0, 1.0),
        "rebalance_bars": 1,
        "holding_bars": 8,
        "complexity_class": "rule",
        "complexity_justification": "lagged funding mean",
        "baseline": baseline.model_dump(mode="json"),
        "required_columns": ["funding_rate", "close"],
        "label_horizon_bars": 8,
        "execution_mode": ExecutionMode.LLM,
        "prompt_ref": "",
        "provenance": [],
        "created_by": "strategy_engineering",
        "created_at": NOW.isoformat(),
        "content_hash": "sha256:" + ("0" * 64),
    }
    return StrategySpec.model_validate({**data, "content_hash": StrategySpec.hash_fields(data)})


@pytest.mark.skipif(not _ollama_7b(), reason="local Ollama qwen2.5:7b-instruct not available")
def test_strategy_codegen_ollama_sandbox(tmp_path: Path) -> None:
    stores = Stores.open(tmp_path / "c.sqlite3")
    gateway = ModelGateway(
        ModelRegistry.load(Path("registries/models")),
        stores.costs,
        tmp_path / "cache.sqlite3",
        SystemClock(),
    )
    codegen = StrategyCodeGenerator(
        StructuredLLM(gateway),
        PromptRegistry.load(Path("registries/prompts")),
        model="local-qwen",
        max_output_tokens=1600,
        max_attempts=3,
    )
    cost = CostModel(
        version="cm-binance-usdm-v1",
        taker_fee_bps=4.0,
        maker_fee_bps=2.0,
        half_spread_bps=1.0,
        impact_bps_per_million=0.5,
        capacity_notional_usd=1_000_000.0,
    )
    record = codegen.generate(
        _spec(),
        cost_model=cost,
        actor=AgentRole.STRATEGY_ENGINEERING,
        task_id="tsk_codegen_net",
        budget=Budget(
            max_usd=0,
            max_tokens=100_000,
            max_http_requests=30,
            max_wall_seconds=1200,
            max_cpu_seconds=1200,
        ),
        now=NOW,
    )
    assert record.execution_mode is ExecutionMode.LLM
    assert record.ok, (record.failure_reason, record.attempts)
    pkg = codegen.last_package
    assert pkg is not None
    n = 128
    rng = np.random.default_rng(0)
    funding = rng.normal(0, 0.0001, size=n)
    close = 100 + np.cumsum(rng.normal(0, 0.5, size=n))
    data = {
        "s0": {
            "funding_rate": funding.astype(float),
            "close": close.astype(float),
            "ts": np.arange(n, dtype=np.int64) * 3_600_000_000_000,
        }
    }
    runner = SandboxRunner(audit=stores.audit)
    result = runner.run(pkg, data, [SandboxJob("j", "s0", dict(pkg.params))])
    assert result.ok, result.error
    trunc = truncation_invariance_check(
        pkg, data, {"s0": sample_checkpoints(n, 4, seed=1)}, runner=runner
    )
    assert trunc.ok
