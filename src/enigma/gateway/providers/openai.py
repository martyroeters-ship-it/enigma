"""OpenAI Chat Completions and OpenAI-compatible endpoints (OpenRouter, Groq, Together,
DeepSeek, Mistral, vLLM, LM Studio)."""

from __future__ import annotations

from typing import Any

from enigma.gateway.jsonschema import is_strict_compatible
from enigma.gateway.providers.base import (
    PreparedRequest,
    ProviderAdapter,
    ProviderCall,
    ProviderResult,
    billable_output_cap,
    map_effort,
    normalize_tool_call,
    raw_hash,
    schema_instruction,
)
from enigma.gateway.registry import ReasoningStyle, StructuredMode
from enigma.gateway.types import (
    FinishReason,
    MessageRole,
    NamedToolChoice,
    ReasoningEffort,
    Usage,
)

_FINISH = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}


def openai_messages(call: ProviderCall, extra_system: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if extra_system:
        out.append({"role": "system", "content": extra_system})
    for m in call.request.messages:
        if m.role is MessageRole.TOOL:
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        elif m.role is MessageRole.ASSISTANT and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.raw_arguments},
                        }
                        for tc in m.tool_calls
                    ],
                }
            )
        else:
            out.append({"role": m.role.value, "content": m.content})
    return out


class OpenAIAdapter(ProviderAdapter):
    """OpenAI `/chat/completions` with json_schema response_format, tools, reasoning_effort."""

    def build(self, call: ProviderCall) -> PreparedRequest:  # noqa: PLR0912
        caps = call.model.capabilities
        req = call.request
        ignored: list[str] = []
        extra_system: str | None = None
        payload: dict[str, Any] = {"model": call.model.model_id}
        structured_mode: str | None = None

        if call.json_schema is not None:
            if caps.structured_output is StructuredMode.JSON_SCHEMA and not req.tools:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": req.resolved_schema_name(),
                        "schema": call.json_schema,
                        "strict": is_strict_compatible(call.json_schema),
                    },
                }
                structured_mode = "native"
            elif caps.structured_output is StructuredMode.JSON_MODE and not req.tools:
                payload["response_format"] = {"type": "json_object"}
                extra_system = schema_instruction(call.json_schema)
                structured_mode = "json_mode"
            else:
                extra_system = schema_instruction(call.json_schema)
                structured_mode = "prompt"
        payload["messages"] = openai_messages(call, extra_system)

        if req.temperature is not None:
            if caps.temperature:
                payload["temperature"] = req.temperature
            else:
                ignored.append("temperature")

        reasoning_applied: str | None = None
        if caps.reasoning is ReasoningStyle.OPENAI_EFFORT:
            reasoning_applied = map_effort(call.reasoning_effort, caps.reasoning_efforts)
            payload["reasoning_effort"] = reasoning_applied
        elif caps.reasoning is ReasoningStyle.OPENROUTER:
            if call.reasoning_effort is ReasoningEffort.NONE:
                payload["reasoning"] = {"enabled": False}
                reasoning_applied = "disabled"
            else:
                reasoning_applied = map_effort(call.reasoning_effort, caps.reasoning_efforts)
                payload["reasoning"] = {"effort": reasoning_applied}
        elif call.reasoning_effort is not ReasoningEffort.NONE:
            ignored.append("reasoning_effort")
        if reasoning_applied is not None and reasoning_applied != call.reasoning_effort.value:
            ignored.append(f"reasoning_effort={call.reasoning_effort.value}->{reasoning_applied}")

        payload[call.provider.max_tokens_param] = billable_output_cap(call)

        if req.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in req.tools
            ]
            if isinstance(req.tool_choice, NamedToolChoice):
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": req.tool_choice.name},
                }
            else:
                payload["tool_choice"] = req.tool_choice

        headers: dict[str, str] = {}
        if call.api_key:
            headers["authorization"] = f"Bearer {call.api_key}"
        return PreparedRequest(
            url=f"{call.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            payload=payload,
            ignored=tuple(ignored),
            reasoning_applied=reasoning_applied,
            structured_mode=structured_mode,
        )

    def parse(
        self, call: ProviderCall, prepared: PreparedRequest, data: dict[str, Any]
    ) -> ProviderResult:
        choice = data["choices"][0]
        message = choice.get("message") or {}
        tool_calls = tuple(
            normalize_tool_call(
                tc.get("id", f"call_{i}"),
                tc["function"]["name"],
                tc["function"].get("arguments"),
                call.offered_tools,
            )
            for i, tc in enumerate(message.get("tool_calls") or [])
        )
        raw_finish = choice.get("finish_reason")
        finish = _FINISH.get(str(raw_finish), FinishReason.OTHER)
        extra: dict[str, Any] = {}
        if message.get("refusal"):
            finish = FinishReason.CONTENT_FILTER
            extra["refusal"] = message["refusal"]
        if data.get("system_fingerprint"):
            extra["system_fingerprint"] = data["system_fingerprint"]
        usage = data.get("usage") or {}
        out_details = usage.get("completion_tokens_details") or {}
        in_details = usage.get("prompt_tokens_details") or {}
        return ProviderResult(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                reasoning_tokens=int(out_details.get("reasoning_tokens") or 0),
                cached_input_tokens=int(in_details.get("cached_tokens") or 0),
            ),
            model_version=str(data.get("model") or call.model.model_id),
            finish_reason=finish,
            raw_finish_reason=None if raw_finish is None else str(raw_finish),
            raw_hash=raw_hash(data),
            reasoning_applied=prepared.reasoning_applied,
            ignored=prepared.ignored,
            structured_mode=prepared.structured_mode,
            extra=extra,
        )


class OpenAICompatibleAdapter(OpenAIAdapter):
    """Generic OpenAI-compatible server. Capabilities come from the model registry entry;
    the API key is optional (local servers) and the max-tokens field name is configurable."""
