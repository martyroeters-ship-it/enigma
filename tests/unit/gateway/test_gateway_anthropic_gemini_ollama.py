from __future__ import annotations

import json
from typing import Any

import httpx
import respx
from gateway_fixtures import (
    ANTHROPIC_URL,
    FAKE_ANTHROPIC_KEY,
    FAKE_GEMINI_KEY,
    GEMINI_BASE,
    OLLAMA_BASE,
    GatewayFactory,
    anthropic_reply,
)
from pydantic import BaseModel, ConfigDict

from enigma.gateway import (
    FinishReason,
    Message,
    MessageRole,
    ModelRequest,
    NamedToolChoice,
    ReasoningEffort,
    ToolCall,
    ToolSpec,
)
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import Budget

BUDGET = Budget(max_usd=5.0, max_tokens=1_000_000)
GEMINI_URL = f"{GEMINI_BASE}/models/gemini-test:generateContent"


class Signal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset: str
    note: str | None = None


TOOL = ToolSpec(
    name="get_price",
    description="Get a price",
    parameters={
        "type": "object",
        "additionalProperties": False,
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"],
    },
)
PRIOR_CALL = ToolCall(
    id="toolu_1",
    name="get_price",
    arguments={"symbol": "BTC"},
    raw_arguments='{"symbol": "BTC"}',
    arguments_valid=True,
)


def _req(**kw: Any) -> ModelRequest:
    base: dict[str, Any] = {
        "messages": (
            Message(role=MessageRole.SYSTEM, content="be precise"),
            Message(role=MessageRole.USER, content="hi"),
        )
    }
    base.update(kw)
    return ModelRequest(**base)


def _body(route: respx.Route) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(route.calls.last.request.content)
    return payload


# ------------------------------------------------------------------ Anthropic


@respx.mock
def test_anthropic_structured_output_via_forced_tool(make_gateway: GatewayFactory) -> None:
    route = respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            200,
            json=anthropic_reply(
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_x",
                        "name": "emit_structured_output",
                        "input": {"asset": "ETH"},
                    }
                ],
                stop_reason="tool_use",
            ),
        )
    )
    resp = make_gateway().complete(
        _req(model="claude", response_schema=Signal, reasoning_effort=ReasoningEffort.HIGH),
        AgentRole.CIO,
        "t",
        BUDGET,
    )
    req = route.calls.last.request
    body = _body(route)
    assert req.headers["x-api-key"] == FAKE_ANTHROPIC_KEY
    assert req.headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "be precise"
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert body["tool_choice"] == {"type": "tool", "name": "emit_structured_output"}
    assert body["tools"][0]["input_schema"]["properties"]["asset"] == {
        "title": "Asset",
        "type": "string",
    }
    assert "thinking" not in body
    assert any("thinking incompatible" in p for p in resp.ignored_parameters)
    assert resp.parsed == Signal(asset="ETH")
    assert resp.finish_reason is FinishReason.STOP
    assert resp.structured_mode == "tool"
    assert resp.usage.input_tokens == 140
    assert resp.usage.cached_input_tokens == 20
    assert resp.model_version == "claude-test-20260101"


@respx.mock
def test_anthropic_thinking_budget_and_tool_turns(make_gateway: GatewayFactory) -> None:
    route = respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            200,
            json=anthropic_reply(
                [
                    {"type": "thinking", "thinking": "...", "signature": "s"},
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "get_price",
                        "input": {"symbol": "ETH"},
                    },
                ],
                stop_reason="tool_use",
            ),
        )
    )
    resp = make_gateway().complete(
        _req(
            model="claude",
            tools=(TOOL,),
            reasoning_effort=ReasoningEffort.MEDIUM,
            max_output_tokens=800,
            messages=(
                Message(role=MessageRole.USER, content="price?"),
                Message(role=MessageRole.ASSISTANT, content="", tool_calls=(PRIOR_CALL,)),
                Message(role=MessageRole.TOOL, tool_call_id="toolu_1", content="100"),
                Message(role=MessageRole.USER, content="and ETH?"),
            ),
        ),
        AgentRole.CIO,
        "t",
        BUDGET,
    )
    body = _body(route)
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert body["max_tokens"] == 800 + 4096
    assert "temperature" not in body
    assert body["tool_choice"] == {"type": "auto"}
    assert body["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "get_price", "input": {"symbol": "BTC"}}
        ],
    }
    # tool_result and the following user text merge into one user turn
    assert body["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "100"},
        {"type": "text", "text": "and ETH?"},
    ]
    assert resp.text == "Checking."
    assert resp.tool_calls[0].arguments == {"symbol": "ETH"}
    assert resp.finish_reason is FinishReason.TOOL_CALLS
    assert resp.reasoning_effort_applied == "thinking.budget_tokens=4096"


@respx.mock
def test_anthropic_max_tokens_maps_to_length(make_gateway: GatewayFactory) -> None:
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            200, json=anthropic_reply([{"type": "text", "text": "cut"}], stop_reason="max_tokens")
        )
    )
    resp = make_gateway().complete(_req(model="claude"), AgentRole.CIO, "t", BUDGET)
    assert resp.finish_reason is FinishReason.LENGTH
    assert resp.raw_finish_reason == "max_tokens"


# ------------------------------------------------------------------ Gemini


def _gemini_reply(parts: list[dict[str, Any]], finish: str = "STOP") -> dict[str, Any]:
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}],
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 10,
            "thoughtsTokenCount": 30,
        },
        "modelVersion": "gemini-test-001",
        "responseId": "r1",
    }


@respx.mock
def test_gemini_response_schema_and_thinking(make_gateway: GatewayFactory) -> None:
    route = respx.post(GEMINI_URL).mock(
        return_value=httpx.Response(
            200,
            json=_gemini_reply(
                [{"text": "thinking...", "thought": True}, {"text": '{"asset":"BTC"}'}]
            ),
        )
    )
    resp = make_gateway().complete(
        _req(model="gem", response_schema=Signal, reasoning_effort=ReasoningEffort.LOW),
        AgentRole.CIO,
        "t",
        BUDGET,
    )
    req = route.calls.last.request
    body = _body(route)
    assert req.headers["x-goog-api-key"] == FAKE_GEMINI_KEY
    assert "key=" not in str(req.url)
    gen = body["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    schema = gen["responseSchema"]
    assert schema["type"] == "OBJECT"
    assert schema["properties"]["asset"]["type"] == "STRING"
    assert schema["properties"]["note"] == {"type": "STRING", "nullable": True}
    assert "additionalProperties" not in schema and "title" not in schema
    assert gen["thinkingConfig"] == {"thinkingBudget": 1024}
    assert gen["maxOutputTokens"] == 1000 + 1024
    assert body["systemInstruction"] == {"parts": [{"text": "be precise"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert resp.parsed == Signal(asset="BTC")
    assert resp.text == '{"asset":"BTC"}'
    assert resp.usage.output_tokens == 40
    assert resp.usage.reasoning_tokens == 30
    assert resp.model_version == "gemini-test-001"


@respx.mock
def test_gemini_function_declarations_and_calls(make_gateway: GatewayFactory) -> None:
    route = respx.post(GEMINI_URL).mock(
        return_value=httpx.Response(
            200,
            json=_gemini_reply(
                [{"functionCall": {"name": "get_price", "args": {"symbol": "ETH"}}}]
            ),
        )
    )
    resp = make_gateway().complete(
        _req(
            model="gem",
            tools=(TOOL,),
            tool_choice=NamedToolChoice(name="get_price"),
            messages=(
                Message(role=MessageRole.USER, content="price?"),
                Message(role=MessageRole.ASSISTANT, tool_calls=(PRIOR_CALL,)),
                Message(role=MessageRole.TOOL, tool_call_id="toolu_1", content='{"usd": 1}'),
            ),
        ),
        AgentRole.CIO,
        "t",
        BUDGET,
    )
    body = _body(route)
    decl = body["tools"][0]["functionDeclarations"][0]
    assert decl["name"] == "get_price"
    assert decl["parameters"]["properties"]["symbol"]["type"] == "STRING"
    assert body["toolConfig"] == {
        "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_price"]}
    }
    assert body["contents"][1] == {
        "role": "model",
        "parts": [{"functionCall": {"name": "get_price", "args": {"symbol": "BTC"}}}],
    }
    assert body["contents"][2] == {
        "role": "user",
        "parts": [{"functionResponse": {"name": "get_price", "response": {"usd": 1}}}],
    }
    assert resp.finish_reason is FinishReason.TOOL_CALLS
    assert resp.tool_calls[0].arguments == {"symbol": "ETH"}


@respx.mock
def test_gemini_blocked_prompt(make_gateway: GatewayFactory) -> None:
    respx.post(GEMINI_URL).mock(
        return_value=httpx.Response(
            200, json={"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {}}
        )
    )
    resp = make_gateway().complete(_req(model="gem"), AgentRole.CIO, "t", BUDGET)
    assert resp.finish_reason is FinishReason.CONTENT_FILTER
    assert resp.text == ""


# ------------------------------------------------------------------ Ollama


@respx.mock
def test_ollama_format_think_and_tool_calls(make_gateway: GatewayFactory) -> None:
    respx.get(f"{OLLAMA_BASE}/api/version").mock(
        return_value=httpx.Response(200, json={"version": "0.12.0"})
    )
    route = respx.post(f"{OLLAMA_BASE}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "llama-test",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "get_price", "arguments": {"symbol": "ETH"}}}
                    ],
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 33,
                "eval_count": 7,
            },
        )
    )
    gw = make_gateway(env={})
    resp = gw.complete(
        _req(model="local", tools=(TOOL,), reasoning_effort=ReasoningEffort.NONE),
        AgentRole.CIO,
        "t",
        Budget(max_tokens=100_000),
    )
    body = _body(route)
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"] == {"num_predict": 1000, "temperature": 0.0}
    assert body["tools"][0]["function"]["name"] == "get_price"
    assert resp.tool_calls[0].arguments == {"symbol": "ETH"}
    assert resp.finish_reason is FinishReason.TOOL_CALLS
    assert resp.usage.input_tokens == 33 and resp.usage.output_tokens == 7
    assert resp.cost_usd == 0.0


@respx.mock
def test_ollama_native_schema_format(make_gateway: GatewayFactory) -> None:
    respx.get(f"{OLLAMA_BASE}/api/version").mock(return_value=httpx.Response(200, json={}))
    route = respx.post(f"{OLLAMA_BASE}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "llama-test",
                "message": {"role": "assistant", "content": '{"asset": "SOL"}'},
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
        )
    )
    resp = make_gateway(env={}).complete(
        _req(model="local", response_schema=Signal, reasoning_effort=ReasoningEffort.HIGH),
        AgentRole.CIO,
        "t",
        Budget(max_tokens=100_000),
    )
    body = _body(route)
    assert body["format"]["properties"]["asset"]["type"] == "string"
    assert body["think"] is True
    assert resp.parsed == Signal(asset="SOL")
