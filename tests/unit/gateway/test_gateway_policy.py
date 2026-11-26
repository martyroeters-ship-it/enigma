from __future__ import annotations

import copy
import json
from datetime import timedelta
from typing import Any

import httpx
import pytest
import respx
from gateway_fixtures import (
    ALL_KEYS,
    ANTHROPIC_URL,
    FAKE_ANTHROPIC_KEY,
    FAKE_OPENAI_KEY,
    GEMINI_BASE,
    MODELS_DOC,
    NOW,
    OLLAMA_BASE,
    OPENAI_URL,
    PRICING_DOC,
    ROUTING_DOC,
    GatewayFactory,
    SleepRecorder,
    anthropic_reply,
    openai_reply,
)
from pydantic import BaseModel, ConfigDict, Field

from enigma.core.errors import BudgetExceededError, ProviderUnavailableError
from enigma.gateway import (
    FailureKind,
    GatewayUnavailableError,
    Message,
    MessageRole,
    ModelRequest,
    RequestMetadata,
    StructuredOutputError,
    ToolSpec,
)
from enigma.gateway.credentials import redact
from enigma.gateway.registry import ModelRegistry
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import Budget, CostRecord
from enigma.store.ledger import CostLedger

BUDGET = Budget(max_usd=5.0, max_tokens=1_000_000)
GEMINI_URL = f"{GEMINI_BASE}/models/gemini-test:generateContent"


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    confidence: float = Field(ge=0, le=1)


def _req(**kw: Any) -> ModelRequest:
    base: dict[str, Any] = {"messages": (Message(role=MessageRole.USER, content="hi"),)}
    base.update(kw)
    return ModelRequest(**base)


def _rows(ledger: CostLedger) -> list[dict[str, Any]]:
    return [dict(r) for r in ledger.db.query("SELECT * FROM costs ORDER BY rowid")]


def _ollama_up() -> None:
    respx.get(f"{OLLAMA_BASE}/api/version").mock(return_value=httpx.Response(200, json={}))


def _ollama_down() -> None:
    respx.get(f"{OLLAMA_BASE}/api/version").mock(side_effect=httpx.ConnectError("refused"))


# ------------------------------------------------------------------ routing and fallback


def test_route_resolution_layers(make_gateway: GatewayFactory) -> None:
    reg = make_gateway().registry
    task = reg.route(AgentRole.INTELLIGENCE, "intelligence.extract")
    assert task.candidates == ("gem", "o-plain")
    assert task.max_cost_per_call_usd == 0.5
    assert task.max_output_tokens == 1000  # inherited from defaults
    assert task.source == "intelligence:intelligence.extract"
    role = reg.route(AgentRole.INTELLIGENCE, "unknown.kind")
    assert role.candidates == ("claude", "o-plain", "local")
    qr = reg.route(AgentRole.QUANT_RESEARCH, None)
    assert qr.candidates == ("o-plain", "claude", "local")
    assert qr.max_output_tokens == 2000
    assert reg.route(AgentRole.CIO, None).source == "defaults"


@respx.mock
def test_fallback_order_missing_key_then_provider_error_then_success(
    make_gateway: GatewayFactory, ledger: CostLedger, sleeps: SleepRecorder
) -> None:
    gem = respx.post(GEMINI_URL)
    openai = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(400, json={"error": "bad request"})
    )
    env = {k: v for k, v in ALL_KEYS.items() if k != "GEMINI_API_KEY"}
    _ollama_down()
    gw = make_gateway(env=env)
    with pytest.raises(GatewayUnavailableError) as info:
        gw.complete(
            _req(metadata=RequestMetadata(task_kind="intelligence.extract")),
            AgentRole.INTELLIGENCE,
            "t",
            BUDGET,
        )
    kinds = [(f.candidate, f.kind) for f in info.value.failures]
    assert kinds == [
        ("gem", FailureKind.MISSING_CREDENTIALS),
        ("o-plain", FailureKind.PROVIDER_ERROR),
    ]
    assert not gem.called
    assert openai.call_count == 1  # 400 is not retried
    assert sleeps.calls == []
    rows = _rows(ledger)
    assert len(rows) == 1 and rows[0]["usd"] == 0 and rows[0]["http_requests"] == 1


@respx.mock
def test_fallback_to_next_candidate_after_retries_exhausted(
    make_gateway: GatewayFactory, ledger: CostLedger, sleeps: SleepRecorder
) -> None:
    openai = respx.post(OPENAI_URL).mock(return_value=httpx.Response(503, text="overloaded"))
    anth = respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=anthropic_reply([{"type": "text", "text": "ok"}]))
    )
    gw = make_gateway(max_retries=2)
    resp = gw.complete(_req(), AgentRole.CIO, "t", BUDGET)
    assert resp.provider == "anthropic" and resp.text == "ok"
    assert openai.call_count == 3
    assert anth.call_count == 1
    assert len(sleeps.calls) == 2
    rows = _rows(ledger)
    assert [(r["provider"], r["http_requests"]) for r in rows] == [
        ("openai", 3),
        ("anthropic", 1),
    ]


@respx.mock
def test_retry_backoff_grows_with_jitter_and_honours_retry_after(
    make_gateway: GatewayFactory, sleeps: SleepRecorder
) -> None:
    respx.post(OPENAI_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "7"}),
            httpx.TimeoutException("slow"),
            httpx.Response(500),
            httpx.Response(200, json=openai_reply("done")),
        ]
    )
    gw = make_gateway(max_retries=3, backoff_base_s=1.0, backoff_max_s=30.0)
    resp = gw.complete(_req(model="o-plain"), AgentRole.CIO, "t", BUDGET)
    assert resp.text == "done"
    assert resp.attempts == 4
    first, second, third = sleeps.calls
    assert first == pytest.approx(7.0)  # Retry-After dominates the 0.5-1.0s backoff
    assert 1.0 <= second <= 2.0
    assert 2.0 <= third <= 4.0


@respx.mock
def test_pinned_model_does_not_fall_back(make_gateway: GatewayFactory) -> None:
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(401, json={}))
    anth = respx.post(ANTHROPIC_URL)
    with pytest.raises(ProviderUnavailableError):
        make_gateway().complete(_req(model="o-plain"), AgentRole.CIO, "t", BUDGET)
    assert not anth.called


@respx.mock
def test_candidate_without_tool_support_is_skipped(make_gateway: GatewayFactory) -> None:
    tool = ToolSpec(name="f", description="d")
    with pytest.raises(GatewayUnavailableError) as info:
        make_gateway().complete(
            _req(model="compat-notools", tools=(tool,)), AgentRole.CIO, "t", BUDGET
        )
    assert info.value.failures[0].kind is FailureKind.CAPABILITY


# ------------------------------------------------------------------ credentials


@respx.mock
def test_no_credentials_fails_closed_and_reports_env_vars(
    make_gateway: GatewayFactory, ledger: CostLedger
) -> None:
    _ollama_down()
    gw = make_gateway(env={})
    assert gw.available_providers() == []
    assert gw.missing_credentials() == {
        "anthropic": ("ANTHROPIC_API_KEY",),
        "compat": ("COMPAT_API_KEY",),
        "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "localvllm": ("VLLM_BASE_URL",),
        "openai": ("OPENAI_API_KEY",),
    }
    ollama = next(s for s in gw.credential_report() if s.provider == "ollama")
    assert not ollama.available and "not reachable" in ollama.detail
    with pytest.raises(ProviderUnavailableError) as info:
        gw.complete(_req(), AgentRole.CIO, "t", BUDGET)
    assert "OPENAI_API_KEY" in str(info.value)
    assert "ANTHROPIC_API_KEY" in str(info.value)
    assert _rows(ledger) == []


@respx.mock
def test_available_providers_with_keys_and_local(make_gateway: GatewayFactory) -> None:
    _ollama_up()
    gw = make_gateway(
        env={"GOOGLE_API_KEY": "AIza-alt-key-123456789", "OLLAMA_HOST": "ollama.test:11434"}
    )
    assert gw.available_providers() == ["google", "ollama"]
    assert {m.alias for m in gw.available_models()} == {"gem", "local"}


@respx.mock
def test_keys_redacted_from_errors(make_gateway: GatewayFactory) -> None:
    respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(
            401,
            text=(
                f"Incorrect API key provided: {FAKE_OPENAI_KEY}. "
                f"Authorization: Bearer {FAKE_OPENAI_KEY}"
            ),
        )
    )
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(403, text=f'{{"x-api-key": "{FAKE_ANTHROPIC_KEY}"}}')
    )
    _ollama_down()
    with pytest.raises(ProviderUnavailableError) as info:
        make_gateway().complete(_req(), AgentRole.CIO, "t", BUDGET)
    message = str(info.value)
    assert FAKE_OPENAI_KEY not in message
    assert FAKE_ANTHROPIC_KEY not in message
    assert "[REDACTED]" in message
    assert info.value.__cause__ is None


def test_redact_patterns() -> None:
    text = (
        "key sk-proj-abcdefghijklmnop gsk_abcdefghijkl AIzaSyA1234567890abcdefghijk "
        "Authorization: Bearer abc.def.ghijklmnop url?key=SECRETVALUE&x=1 custom-secret-xyz"
    )
    out = redact(text, ["custom-secret-xyz"])
    for leaked in (
        "sk-proj-abcdefghijklmnop",
        "gsk_abcdefghijkl",
        "AIzaSyA1234567890abcdefghijk",
        "abc.def.ghijklmnop",
        "SECRETVALUE",
        "custom-secret-xyz",
    ):
        assert leaked not in out
    assert "&x=1" in out


# ------------------------------------------------------------------ budgets


@respx.mock
def test_budget_refusal_happens_before_any_http(make_gateway: GatewayFactory) -> None:
    openai = respx.post(OPENAI_URL)
    with pytest.raises(BudgetExceededError, match="remaining budget"):
        make_gateway().complete(
            _req(model="o-plain"), AgentRole.CIO, "t", Budget(max_usd=0.0001, max_tokens=10**6)
        )
    assert not openai.called


@respx.mock
def test_token_budget_and_http_budget(make_gateway: GatewayFactory) -> None:
    openai = respx.post(OPENAI_URL)
    gw = make_gateway()
    with pytest.raises(BudgetExceededError, match="token budget"):
        gw.complete(_req(model="o-plain"), AgentRole.CIO, "t", Budget(max_usd=5, max_tokens=50))
    with pytest.raises(BudgetExceededError, match="HTTP request budget"):
        gw.complete(
            _req(model="o-plain"),
            AgentRole.CIO,
            "t",
            Budget(max_usd=5, max_tokens=10**6, max_http_requests=0),
        )
    assert not openai.called


@respx.mock
def test_budget_counts_prior_spend_for_task(
    make_gateway: GatewayFactory, ledger: CostLedger
) -> None:
    ledger.record(
        CostRecord(
            id="cst_prior",
            actor=AgentRole.CIO,
            task_id="t",
            category="llm",
            provider="openai",
            usd=0.999,
            created_at=NOW,
        )
    )
    with pytest.raises(BudgetExceededError):
        make_gateway().complete(
            _req(model="o-plain"), AgentRole.CIO, "t", Budget(max_usd=1.0, max_tokens=10**6)
        )


@respx.mock
def test_route_max_cost_per_call(make_gateway: GatewayFactory) -> None:
    routing = copy.deepcopy(ROUTING_DOC)
    routing["defaults"]["max_cost_per_call_usd"] = 0.001
    registry = ModelRegistry.from_documents(MODELS_DOC, PRICING_DOC, routing)
    openai = respx.post(OPENAI_URL)
    with pytest.raises(BudgetExceededError, match="route max per call"):
        make_gateway(registry=registry).complete(
            _req(model="o-plain"), AgentRole.CIO, "t", Budget(max_usd=100, max_tokens=10**7)
        )
    assert not openai.called


@respx.mock
def test_daily_cap_default_zero_blocks_paid_but_allows_local(
    make_gateway: GatewayFactory, ledger: CostLedger
) -> None:
    openai = respx.post(OPENAI_URL)
    _ollama_up()
    respx.post(f"{OLLAMA_BASE}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "llama-test",
                "message": {"role": "assistant", "content": "local answer"},
                "done_reason": "stop",
                "prompt_eval_count": 5,
                "eval_count": 3,
            },
        )
    )
    env = {k: v for k, v in ALL_KEYS.items() if k != "ENIGMA_MAX_DAILY_API_SPEND_USD"}
    gw = make_gateway(env=env)
    resp = gw.complete(_req(), AgentRole.CIO, "t", BUDGET)
    assert resp.provider == "ollama" and resp.text == "local answer"
    assert not openai.called
    with pytest.raises(BudgetExceededError, match="daily API ceiling"):
        gw.complete(_req(model="o-plain"), AgentRole.CIO, "t", BUDGET)
    assert _rows(ledger)[0]["usd"] == 0.0


@respx.mock
def test_daily_cap_counts_all_actors_today_only(
    make_gateway: GatewayFactory, ledger: CostLedger
) -> None:
    for i, (actor, when) in enumerate(
        [
            (AgentRole.INTELLIGENCE, NOW - timedelta(hours=1)),
            (AgentRole.QUANT_RESEARCH, NOW - timedelta(days=1)),
        ]
    ):
        ledger.record(
            CostRecord(
                id=f"cst_{i}",
                actor=actor,
                task_id=None,
                category="llm",
                provider="openai",
                usd=0.9,
                created_at=when,
            )
        )
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=openai_reply()))
    env = {**ALL_KEYS, "ENIGMA_MAX_DAILY_API_SPEND_USD": "0.905"}
    with pytest.raises(BudgetExceededError, match="daily API ceiling"):
        make_gateway(env=env).complete(_req(model="o-plain"), AgentRole.CIO, "t", BUDGET)
    env["ENIGMA_MAX_DAILY_API_SPEND_USD"] = "1.0"
    assert make_gateway(env=env).complete(_req(model="o-plain"), AgentRole.CIO, "t", BUDGET)


def test_invalid_daily_cap_fails_closed(make_gateway: GatewayFactory) -> None:
    gw = make_gateway(env={**ALL_KEYS, "ENIGMA_MAX_DAILY_API_SPEND_USD": "lots"})
    with pytest.raises(BudgetExceededError):
        gw.daily_cap_usd()


@respx.mock
def test_unpriced_paid_model_is_refused(make_gateway: GatewayFactory) -> None:
    openai = respx.post(OPENAI_URL)
    with pytest.raises(BudgetExceededError, match="no pricing entry"):
        make_gateway().complete(_req(model="openai/gpt-unlisted"), AgentRole.CIO, "t", BUDGET)
    assert not openai.called


# ------------------------------------------------------------------ ledger and caching


@respx.mock
def test_cost_record_written_with_provider_model_version(
    make_gateway: GatewayFactory, ledger: CostLedger
) -> None:
    respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(
            200,
            json=openai_reply(
                "x", model="gpt-plain-2026-02-02", prompt_tokens=1000, completion_tokens=500
            ),
        )
    )
    resp = make_gateway().complete(_req(model="o-plain"), AgentRole.QUANT_RESEARCH, "tsk_9", BUDGET)
    expected_usd = (1000 * 2.0 + 500 * 8.0) / 1_000_000
    assert resp.cost_usd == pytest.approx(expected_usd)
    assert resp.cost_is_estimate
    assert resp.latency_ms == pytest.approx(250.0)
    assert resp.registry_version and resp.registry_version.startswith("sha256:")
    (row,) = _rows(ledger)
    assert row["category"] == "llm"
    assert row["actor"] == "quant_research"
    assert row["task_id"] == "tsk_9"
    assert row["provider"] == "openai"
    assert row["model"] == "gpt-plain-2026-02-02"
    assert row["input_tokens"] == 1000 and row["output_tokens"] == 500
    assert row["usd"] == pytest.approx(expected_usd)
    assert row["cached"] == 0


@respx.mock
def test_cache_hit_for_temperature_zero(make_gateway: GatewayFactory, ledger: CostLedger) -> None:
    route = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, json=openai_reply('{"label":"up","confidence":0.5}'))
    )
    gw = make_gateway()
    first = gw.complete(_req(model="o-plain", response_schema=Verdict), AgentRole.CIO, "t", BUDGET)
    second = gw.complete(_req(model="o-plain", response_schema=Verdict), AgentRole.CIO, "t", BUDGET)
    assert route.call_count == 1
    assert not first.cached and second.cached
    assert second.cost_usd == 0.0
    assert second.parsed == first.parsed == Verdict(label="up", confidence=0.5)
    assert second.raw_response_hash == first.raw_response_hash
    rows = _rows(ledger)
    assert [r["cached"] for r in rows] == [0, 1]
    assert rows[1]["usd"] == 0 and rows[1]["input_tokens"] == 0


@respx.mock
def test_no_cache_when_temperature_positive_unless_forced(make_gateway: GatewayFactory) -> None:
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=openai_reply()))
    gw = make_gateway()
    for _ in range(2):
        gw.complete(_req(model="o-plain", temperature=0.7), AgentRole.CIO, "t", BUDGET)
    assert route.call_count == 2
    for _ in range(2):
        gw.complete(
            _req(model="o-plain", temperature=0.7, cacheable=True), AgentRole.CIO, "t", BUDGET
        )
    assert route.call_count == 3
    for _ in range(2):
        gw.complete(_req(model="o-plain", cacheable=False), AgentRole.CIO, "t", BUDGET)
    assert route.call_count == 5


@respx.mock
def test_cache_serves_even_when_key_later_removed(make_gateway: GatewayFactory) -> None:
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=openai_reply("cached")))
    make_gateway().complete(_req(model="o-plain"), AgentRole.CIO, "t", BUDGET)
    resp = make_gateway(env={}).complete(_req(model="o-plain"), AgentRole.CIO, "t", BUDGET)
    assert resp.cached and resp.text == "cached"


# ------------------------------------------------------------------ structured output


@respx.mock
def test_structured_output_repair_once(make_gateway: GatewayFactory, ledger: CostLedger) -> None:
    route = respx.post(OPENAI_URL).mock(
        side_effect=[
            httpx.Response(200, json=openai_reply('{"label":"up","confidence":7}')),
            httpx.Response(200, json=openai_reply('{"label":"up","confidence":0.7}')),
        ]
    )
    resp = make_gateway().complete(
        _req(model="o-plain", response_schema=Verdict), AgentRole.CIO, "t", BUDGET
    )
    assert resp.repaired and resp.parsed == Verdict(label="up", confidence=0.7)
    assert resp.usage.input_tokens == 200
    repair_msgs = json.loads(route.calls[1].request.content)["messages"]
    assert repair_msgs[-2] == {"role": "assistant", "content": '{"label":"up","confidence":7}'}
    assert "confidence" in repair_msgs[-1]["content"]
    assert len(_rows(ledger)) == 2


@respx.mock
def test_structured_output_failure_after_repair(make_gateway: GatewayFactory) -> None:
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=openai_reply("nope")))
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=anthropic_reply([{"type": "text", "text": "x"}]))
    )
    with pytest.raises(StructuredOutputError, match="not valid JSON"):
        make_gateway().complete(
            _req(model="o-plain", response_schema=Verdict), AgentRole.CIO, "t", BUDGET
        )


@respx.mock
def test_dict_schema_validation_and_truncation(make_gateway: GatewayFactory) -> None:
    schema = {
        "type": "object",
        "required": ["n"],
        "properties": {"n": {"type": "integer", "minimum": 1}},
        "additionalProperties": False,
    }
    respx.post(OPENAI_URL).mock(
        side_effect=[
            httpx.Response(200, json=openai_reply('{"n": 1', finish="length")),
            httpx.Response(200, json=openai_reply('{"n": 2}')),
        ]
    )
    resp = make_gateway().complete(
        _req(model="o-plain", response_schema=schema), AgentRole.CIO, "t", BUDGET
    )
    assert resp.parsed == {"n": 2} and resp.repaired


def test_api_key_never_in_request_hash_or_cache(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    request = _req(model="o-plain")
    assert FAKE_OPENAI_KEY not in json.dumps(request.canonical())
    assert gw.cache is not None
