from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from gateway_fixtures import OLLAMA_BASE, OPENAI_URL, GatewayFactory, openai_reply
from pydantic import BaseModel

from enigma.core.errors import ProviderUnavailableError
from enigma.gateway import ModelResponse, StructuredLLM
from enigma.gateway.benchmark import (
    NO_PROVIDERS,
    TASK_SET_HASH,
    TASKS,
    BenchmarkRunner,
    ResultStatus,
    main,
    score,
)
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import Budget

BUDGET = Budget(max_usd=5.0, max_tokens=1_000_000)


class Thesis(BaseModel):
    claim: str
    horizon_days: int


def _ollama_down() -> None:
    respx.get(f"{OLLAMA_BASE}/api/version").mock(side_effect=httpx.ConnectError("refused"))


# ------------------------------------------------------------------ StructuredLLM


@respx.mock
def test_structured_llm_unavailable_without_providers(make_gateway: GatewayFactory) -> None:
    _ollama_down()
    llm = StructuredLLM(make_gateway(env={}))
    assert not llm.is_available()
    with pytest.raises(ProviderUnavailableError, match="OPENAI_API_KEY"):
        llm.generate(Thesis, "sys", "user", AgentRole.INTELLIGENCE, "t", BUDGET)
    assert llm.last_response is None


@respx.mock
def test_structured_llm_returns_typed_instance(make_gateway: GatewayFactory) -> None:
    route = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(
            200, json=openai_reply('{"claim": "funding predicts returns", "horizon_days": 1}')
        )
    )
    llm = StructuredLLM(make_gateway())
    out = llm.generate(
        Thesis, "sys", "abstract...", AgentRole.CIO, "tsk_1", BUDGET, model="o-plain"
    )
    assert out == Thesis(claim="funding predicts returns", horizon_days=1)
    assert llm.last_response is not None and llm.last_response.provider == "openai"
    body = json.loads(route.calls.last.request.content)
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["response_format"]["json_schema"]["name"] == "Thesis"


# ------------------------------------------------------------------ benchmark


def test_task_set_is_versioned_and_stable() -> None:
    assert TASK_SET_HASH.startswith("sha256:")
    assert len({t.id for t in TASKS}) == len(TASKS) == 5
    assert {t.kind.value for t in TASKS} == {
        "structured_extraction",
        "schema_adherence",
        "tool_call",
        "arithmetic_reasoning",
    }


@respx.mock
def test_benchmark_reports_no_providers(make_gateway: GatewayFactory, tmp_path: Path) -> None:
    _ollama_down()
    gw = make_gateway(env={})
    report = BenchmarkRunner(gw, BUDGET).run()
    assert report.status == NO_PROVIDERS
    assert report.results == () and report.summary == ()
    assert "OPENAI_API_KEY" in report.missing_credentials["openai"]
    table = report.to_table()
    assert "no providers available" in table
    assert "ollama: not reachable" in table
    out = tmp_path / "bench.json"
    assert main(["--out", str(out)], gateway_factory=lambda: gw) == 2
    assert json.loads(out.read_text())["status"] == NO_PROVIDERS


def _fake_response(**kw: Any) -> ModelResponse:
    base: dict[str, Any] = {
        "text": "",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "provider": "p",
        "model_id": "m",
        "model_version": "m-1",
        "latency_ms": 1.0,
        "cost_usd": 0.0,
        "finish_reason": "stop",
        "raw_response_hash": "sha256:x",
        "request_hash": "sha256:y",
    }
    base.update(kw)
    return ModelResponse.model_validate(base)


def test_scoring_is_deterministic() -> None:
    extraction, schema_task, tool_task, arith, _ = TASKS
    good = extraction.response_schema
    assert isinstance(good, type)
    parsed = good.model_validate(
        {
            "statement": "High funding predicts negative 24h BTC returns.",
            "asset": "BTC",
            "predictor": "8-hour Funding rate",
            "direction": "negative",
            "horizon_hours": 24,
            "falsification_test": "No negative mean return after costs out of sample.",
        }
    )
    assert all(score(extraction, _fake_response(parsed=parsed)).values())
    assert score(extraction, None) == {
        "schema_valid": False,
        "asset": False,
        "direction": False,
        "horizon_hours": False,
        "predictor": False,
    }
    partial = score(
        schema_task,
        _fake_response(
            parsed={
                "symbol": "ETHUSDT",
                "venue": "binance",
                "kind": "spot",
                "max_leverage": 3,
                "tags": ["paper", "research"],
            }
        ),
    )
    assert sum(partial.values()) == 5 and not partial["kind"]
    tool_ok = _fake_response(
        tool_calls=[
            {
                "id": "c",
                "name": "get_ohlcv",
                "arguments": {"symbol": "ETHUSDT", "interval": "1d", "start_date": "2024-03-01"},
                "raw_arguments": "{}",
                "arguments_valid": True,
            }
        ]
    )
    assert all(score(tool_task, tool_ok).values())
    assert not any(score(tool_task, _fake_response()).values())
    arith_schema = arith.response_schema
    assert isinstance(arith_schema, type)
    assert score(arith, _fake_response(parsed=arith_schema(answer=285000.0)))["answer"]
    assert not score(arith, _fake_response(parsed=arith_schema(answer=285001.0)))["answer"]


@respx.mock
def test_benchmark_runs_available_models_through_gateway(
    make_gateway: GatewayFactory, tmp_path: Path
) -> None:
    _ollama_down()
    replies = [
        openai_reply(
            json.dumps(
                {
                    "statement": "High funding precedes negative BTC returns over 24 hours.",
                    "asset": "BTC",
                    "predictor": "funding rate",
                    "direction": "negative",
                    "horizon_hours": 24,
                    "falsification_test": "Out-of-sample mean return after costs is >= 0.",
                }
            )
        ),
        openai_reply(
            json.dumps(
                {
                    "symbol": "ETHUSDT",
                    "venue": "binance",
                    "kind": "perpetual",
                    "max_leverage": 3,
                    "tags": ["research", "paper"],
                }
            )
        ),
        openai_reply(
            None,
            finish="tool_calls",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "get_ohlcv",
                        "arguments": json.dumps(
                            {"symbol": "ETHUSDT", "interval": "1d", "start_date": "2024-03-01"}
                        ),
                    },
                }
            ],
        ),
        openai_reply('{"answer": 285000}'),
        openai_reply('{"answer": 1039.5}'),
    ]
    respx.post(OPENAI_URL).mock(side_effect=[httpx.Response(200, json=r) for r in replies])
    gw = make_gateway(
        env={"OPENAI_API_KEY": "sk-test-0123456789", "ENIGMA_MAX_DAILY_API_SPEND_USD": "1"}
    )
    report = BenchmarkRunner(gw, BUDGET).run(models=["o-plain"])
    assert report.status == "completed"
    assert [r.status for r in report.results] == [ResultStatus.SCORED] * 5
    by_task = {r.task_id: r for r in report.results}
    assert by_task["hypothesis_extraction_v1"].score == 1.0
    assert by_task["schema_adherence_v1"].score == pytest.approx(5 / 6)
    assert by_task["tool_call_v1"].score == 1.0
    assert by_task["arithmetic_allocation_v1"].score == 1.0
    (summary,) = report.summary
    assert summary.model_alias == "o-plain" and summary.tasks_scored == 5
    assert summary.mean_score == pytest.approx((4 + 5 / 6) / 5)
    assert summary.total_cost_usd > 0
    assert "| o-plain | openai | 5 |" in report.to_table()
    out = tmp_path / "r.json"
    report.write(out)
    assert json.loads(out.read_text())["task_set_hash"] == TASK_SET_HASH


@respx.mock
def test_benchmark_records_budget_skips_not_scores(make_gateway: GatewayFactory) -> None:
    _ollama_down()
    openai = respx.post(OPENAI_URL)
    gw = make_gateway(env={"OPENAI_API_KEY": "sk-test-0123456789"})  # daily cap defaults to 0
    report = BenchmarkRunner(gw, BUDGET).run(models=["o-plain"])
    assert {r.status for r in report.results} == {ResultStatus.SKIPPED_BUDGET}
    assert all(r.score is None for r in report.results)
    assert report.summary[0].mean_score is None
    assert not openai.called
