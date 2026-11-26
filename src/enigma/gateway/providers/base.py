"""Shared adapter machinery: HTTP posting, error mapping, redaction, tool-call normalization."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from enigma.core.hashing import sha256_of
from enigma.gateway.credentials import redact
from enigma.gateway.errors import ProviderCallError
from enigma.gateway.registry import ModelSpec, ProviderSpec
from enigma.gateway.types import (
    FinishReason,
    MessageRole,
    ModelRequest,
    ReasoningEffort,
    ToolCall,
    Usage,
)

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_ERROR_BODY_CHARS = 500

REASONING_BUDGET_TOKENS: dict[ReasoningEffort, int] = {
    ReasoningEffort.NONE: 0,
    ReasoningEffort.LOW: 1024,
    ReasoningEffort.MEDIUM: 4096,
    ReasoningEffort.HIGH: 16384,
}

STRUCTURED_INSTRUCTION = (
    "Respond with a single JSON object only (no prose, no code fences) that validates "
    "against this JSON Schema:\n{schema}"
)


@dataclass(frozen=True)
class ProviderCall:
    request: ModelRequest
    model: ModelSpec
    provider: ProviderSpec
    base_url: str
    api_key: str | None
    output_token_cap: int
    reasoning_effort: ReasoningEffort
    json_schema: dict[str, Any] | None
    secrets: tuple[str, ...] = ()
    # "auto" uses the provider's native schema when it has one. "json" asks for
    # a JSON object plus the schema in the prompt, used when native grammar
    # returns an empty body.
    structured_format: str = "auto"

    @property
    def offered_tools(self) -> frozenset[str]:
        return frozenset(t.name for t in self.request.tools)


@dataclass
class PreparedRequest:
    url: str
    headers: dict[str, str]
    payload: dict[str, Any]
    ignored: tuple[str, ...] = ()
    reasoning_applied: str | None = None
    structured_mode: str | None = None


@dataclass
class ProviderResult:
    text: str
    tool_calls: tuple[ToolCall, ...]
    usage: Usage
    model_version: str
    finish_reason: FinishReason
    raw_finish_reason: str | None
    raw_hash: str
    reasoning_applied: str | None = None
    ignored: tuple[str, ...] = ()
    structured_mode: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh")


def map_effort(effort: ReasoningEffort, supported: tuple[str, ...]) -> str | None:
    """Pick the provider-native effort closest to the requested one (None if unsupported)."""
    if not supported:
        return None
    if effort.value in supported:
        return effort.value
    want = EFFORT_ORDER.index(effort.value)
    return min(
        supported, key=lambda s: abs(EFFORT_ORDER.index(s) - want) if s in EFFORT_ORDER else 99
    )


def billable_output_cap(call: ProviderCall) -> int:
    """Hard ceiling on billed output tokens (visible + reasoning) sent to the provider."""
    cap = call.output_token_cap
    if call.model.capabilities.reasoning and call.reasoning_effort is not ReasoningEffort.NONE:
        cap += REASONING_BUDGET_TOKENS[call.reasoning_effort]
    return min(cap, call.model.max_output_tokens)


def schema_instruction(schema: dict[str, Any]) -> str:
    return STRUCTURED_INSTRUCTION.format(schema=json.dumps(schema, sort_keys=True))


def normalize_tool_call(
    call_id: str, name: str, raw: str | dict[str, Any] | None, offered: frozenset[str]
) -> ToolCall:
    raw_text = raw if isinstance(raw, str) else json.dumps(raw or {}, sort_keys=True)
    error: str | None = None
    args: dict[str, Any] = {}
    if isinstance(raw, dict):
        args = raw
    else:
        try:
            parsed = json.loads(raw_text) if raw_text.strip() else {}
        except json.JSONDecodeError as exc:
            parsed = None
            error = f"arguments are not valid JSON: {exc.msg}"
        if isinstance(parsed, dict):
            args = parsed
        elif error is None:
            error = "arguments are not a JSON object"
    if error is None and name not in offered:
        error = f"model called unknown tool {name!r}"
    return ToolCall(
        id=call_id or "call_0",
        name=name or "unknown",
        arguments=args,
        raw_arguments=raw_text,
        arguments_valid=error is None,
        error=error,
    )


def conversation_tool_names(request: ModelRequest) -> dict[str, str]:
    """Map tool_call_id -> tool name from prior assistant turns."""
    names: dict[str, str] = {}
    for message in request.messages:
        if message.role is MessageRole.ASSISTANT:
            for tc in message.tool_calls:
                names[tc.id] = tc.name
    return names


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


class ProviderAdapter(ABC):
    """Translates a `ProviderCall` to one HTTP request and parses the reply."""

    def __init__(self, client: httpx.Client) -> None:
        self.client = client

    @abstractmethod
    def build(self, call: ProviderCall) -> PreparedRequest:
        """Translate the call to the provider's wire format."""

    @abstractmethod
    def parse(
        self, call: ProviderCall, prepared: PreparedRequest, data: dict[str, Any]
    ) -> ProviderResult:
        """Convert the provider's JSON reply to a normalized result."""

    def invoke(self, call: ProviderCall) -> ProviderResult:
        prepared = self.build(call)
        data = self.post(call, prepared.url, prepared.headers, prepared.payload)
        try:
            return self.parse(call, prepared, data)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderCallError(
                redact(f"{call.provider.name}: unexpected response shape: {exc!r}", call.secrets),
                provider=call.provider.name,
            ) from None

    def post(
        self, call: ProviderCall, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        name = call.provider.name
        all_headers = {
            "content-type": "application/json",
            **call.provider.extra_headers,
            **headers,
        }
        try:
            response = self.client.post(
                url, headers=all_headers, json=payload, timeout=call.provider.timeout_s
            )
        except httpx.TimeoutException as exc:
            raise ProviderCallError(
                redact(f"{name}: timeout ({type(exc).__name__})", call.secrets),
                provider=name,
                retryable=True,
            ) from None
        except httpx.TransportError as exc:
            raise ProviderCallError(
                redact(f"{name}: transport error: {exc}", call.secrets),
                provider=name,
                retryable=True,
            ) from None
        if response.status_code >= 400:
            body = response.text[:_ERROR_BODY_CHARS]
            raise ProviderCallError(
                redact(f"{name}: HTTP {response.status_code}: {body}", call.secrets),
                provider=name,
                status_code=response.status_code,
                retryable=response.status_code in RETRYABLE_STATUS,
                retry_after_s=_retry_after(response),
            )
        try:
            data = response.json()
        except ValueError:
            raise ProviderCallError(
                redact(f"{name}: response is not JSON", call.secrets), provider=name
            ) from None
        if not isinstance(data, dict):
            raise ProviderCallError(f"{name}: response JSON is not an object", provider=name)
        return data


def raw_hash(data: dict[str, Any]) -> str:
    return sha256_of(data)
