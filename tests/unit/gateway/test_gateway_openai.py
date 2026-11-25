from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from gateway_fixtures import (
    COMPAT_URL,
    FAKE_OPENAI_KEY,
    OPENAI_URL,
    GatewayFactory,
    openai_reply,
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


class Pick(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str
    score: int


TOOL = ToolSpec(
    name="get_price",
    description="Get a price",
    parameters={
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"],
    },
)


def _req(**kw: Any) -> ModelRequest:
    base: dict[str, Any] = {
        "messages": (
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(role=MessageRole.USER, content="hi"),
        )
    }
    base.update(kw)
    return ModelRequest(**base)


def _body(route: respx.Route) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(route.calls.last.request.content)
    return payload


@respx.mock
def test_openai_structured_reasoning_request_shape(make_gateway: GatewayFactory) -> None:
    route = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(
            200,
            json=openai_reply(
                '{"symbol":"BTC","score":3}',
                model="gpt-reason-2026-01-01",
                completion_tokens=80,
                reasoning_tokens=60,
            ),
        )
    )
    gw = make_gateway()
    resp = gw.complete(
        _req(model="o-reason", response_schema=Pick, reasoning_effort=ReasoningEffort.NONE),
        AgentRole.CIO,
        "tsk_1",
        BUDGET,
    )
    body = _body(route)
    request = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {FAKE_OPENAI_KEY}"
    assert body["model"] == "gpt-reason"
    assert body["reasoning_effort"] == "minimal"
    assert "temperature" not in body
    assert body["max_completion_tokens"] == 1000
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "Pick"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"]["required"] == ["symbol", "score"]
    assert resp.parsed == Pick(symbol="BTC", score=3)
    assert resp.structured_mode == "native"
    assert resp.model_version == "gpt-reason-2026-01-01"
    assert resp.usage.reasoning_tokens == 60
    assert resp.reasoning_effort_applied == "minimal"
    assert "temperature" in resp.ignored_parameters
    assert "reasoning_effort=none->minimal" in resp.ignored_parameters


@respx.mock
def test_openai_reasoning_budget_added_to_completion_cap(make_gateway: GatewayFactory) -> None:
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=openai_reply()))
    make_gateway().complete(
        _req(model="o-reason", reasoning_effort=ReasoningEffort.HIGH, max_output_tokens=500),
        AgentRole.CIO,
        "t",
        BUDGET,
    )
    body = _body(route)
    assert body["reasoning_effort"] == "high"
    assert body["max_completion_tokens"] == 500 + 16384


@respx.mock
def test_openai_tools_and_tool_call_parsing(make_gateway: GatewayFactory) -> None:
    route = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(
            200,
            json=openai_reply(
                None,
                finish="tool_calls",
                tool_calls=[
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "get_price", "arguments": '{"symbol":"ETH"}'},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "get_price", "arguments": '{"symbol":'},
                    },
                    {
                        "id": "call_c",
                        "type": "function",
                        "function": {"name": "rm_rf", "arguments": "{}"},
                    },
                ],
            ),
        )
    )
    resp = make_gateway().complete(
        _req(model="o-plain", tools=(TOOL,), tool_choice=NamedToolChoice(name="get_price")),
        AgentRole.CIO,
        "t",
        BUDGET,
    )
    body = _body(route)
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_price",
                "description": "Get a price",
                "parameters": TOOL.parameters,
            },
        }
    ]
    assert body["tool_choice"] == {"type": "function", "function": {"name": "get_price"}}
    assert body["temperature"] == 0.0
    assert resp.finish_reason is FinishReason.TOOL_CALLS
    ok, bad_json, unknown = resp.tool_calls
    assert ok.arguments == {"symbol": "ETH"} and ok.arguments_valid
    assert not bad_json.arguments_valid and "not valid JSON" in (bad_json.error or "")
    assert not unknown.arguments_valid and "unknown tool" in (unknown.error or "")


@respx.mock
def test_openai_tool_result_turns_round_trip(make_gateway: GatewayFactory) -> None:
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=openai_reply()))
    call = ToolCall(
        id="call_a",
        name="get_price",
        arguments={"symbol": "ETH"},
        raw_arguments='{"symbol":"ETH"}',
        arguments_valid=True,
    )
    make_gateway().complete(
        _req(
            model="o-plain",
            tools=(TOOL,),
            messages=(
                Message(role=MessageRole.USER, content="price?"),
                Message(role=MessageRole.ASSISTANT, tool_calls=(call,)),
                Message(role=MessageRole.TOOL, tool_call_id="call_a", content='{"p": 1}'),
            ),
        ),
        AgentRole.CIO,
        "t",
        BUDGET,
    )
    msgs = _body(route)["messages"]
    assert msgs[1]["tool_calls"][0]["function"] == {
        "name": "get_price",
        "arguments": '{"symbol":"ETH"}',
    }
    assert msgs[2] == {"role": "tool", "tool_call_id": "call_a", "content": '{"p": 1}'}


@respx.mock
def test_openai_refusal_maps_to_content_filter(make_gateway: GatewayFactory) -> None:
    reply = openai_reply(None)
    reply["choices"][0]["message"]["refusal"] = "I can't help with that"
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=reply))
    resp = make_gateway().complete(_req(model="o-plain"), AgentRole.CIO, "t", BUDGET)
    assert resp.finish_reason is FinishReason.CONTENT_FILTER


@respx.mock
def test_compat_json_mode_and_max_tokens_param(make_gateway: GatewayFactory) -> None:
    route = respx.post(COMPAT_URL).mock(
        return_value=httpx.Response(
            200, json=openai_reply('```json\n{"symbol":"SOL","score":1}\n```', model="compat-model")
        )
    )
    resp = make_gateway().complete(
        _req(model="compat", response_schema=Pick), AgentRole.CIO, "t", BUDGET
    )
    body = _body(route)
    assert body["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in body["messages"][0]["content"]
    assert body["max_tokens"] == 1000
    assert "max_completion_tokens" not in body
    assert resp.structured_mode == "json_mode"
    assert resp.parsed == Pick(symbol="SOL", score=1)


@respx.mock
def test_compat_local_server_without_key_sends_no_auth(make_gateway: GatewayFactory) -> None:
    route = respx.post("http://gpu-box:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=openai_reply(model="served-model"))
    )
    gw = make_gateway(env={"VLLM_BASE_URL": "http://gpu-box:8000/v1"})
    resp = gw.complete(_req(model="vllm"), AgentRole.CIO, "t", Budget(max_tokens=10_000))
    assert "authorization" not in route.calls.last.request.headers
    assert resp.cost_usd == 0.0
    assert resp.provider == "localvllm"


@pytest.mark.parametrize(
    ("status", "retryable"), [(400, False), (401, False), (429, True), (503, True)]
)
@respx.mock
def test_openai_error_mapping(make_gateway: GatewayFactory, status: int, retryable: bool) -> None:
    from enigma.gateway.errors import ProviderCallError  # noqa: PLC0415
    from enigma.gateway.providers import OpenAIAdapter, ProviderCall  # noqa: PLC0415

    respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(
            status, json={"error": {"message": f"bad key {FAKE_OPENAI_KEY}"}}
        )
    )
    gw = make_gateway()
    spec = gw.registry.models["o-plain"]
    call = ProviderCall(
        request=_req(),
        model=spec,
        provider=gw.registry.providers["openai"],
        base_url="https://api.openai.test/v1",
        api_key=FAKE_OPENAI_KEY,
        output_token_cap=10,
        reasoning_effort=ReasoningEffort.NONE,
        json_schema=None,
        secrets=(FAKE_OPENAI_KEY,),
    )
    with httpx.Client() as client, pytest.raises(ProviderCallError) as info:
        OpenAIAdapter(client).invoke(call)
    assert info.value.status_code == status
    assert info.value.retryable is retryable
    assert FAKE_OPENAI_KEY not in str(info.value)
    assert "[REDACTED]" in str(info.value)
