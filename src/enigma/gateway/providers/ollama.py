"""Ollama local server (`/api/chat`, no key): `format` JSON schema, tools, `think`."""

from __future__ import annotations

from typing import Any

from enigma.gateway.providers.base import (
    PreparedRequest,
    ProviderAdapter,
    ProviderCall,
    ProviderResult,
    billable_output_cap,
    conversation_tool_names,
    map_effort,
    normalize_tool_call,
    raw_hash,
    schema_instruction,
)
from enigma.gateway.registry import ReasoningStyle, StructuredMode
from enigma.gateway.types import FinishReason, MessageRole, NamedToolChoice, ReasoningEffort, Usage

_FINISH = {"stop": FinishReason.STOP, "length": FinishReason.LENGTH}


class OllamaAdapter(ProviderAdapter):
    def build(self, call: ProviderCall) -> PreparedRequest:  # noqa: PLR0912
        req = call.request
        caps = call.model.capabilities
        ignored: list[str] = []
        names = conversation_tool_names(req)
        messages: list[dict[str, Any]] = []
        structured_mode: str | None = None
        payload: dict[str, Any] = {"model": call.model.model_id, "stream": False}
        if call.json_schema is not None:
            if caps.structured_output is StructuredMode.JSON_SCHEMA and not req.tools:
                payload["format"] = call.json_schema
                structured_mode = "native"
            else:
                messages.append({"role": "system", "content": schema_instruction(call.json_schema)})
                structured_mode = "prompt"
        for m in req.messages:
            if m.role is MessageRole.TOOL:
                messages.append(
                    {
                        "role": "tool",
                        "content": m.content,
                        "tool_name": m.name or names.get(m.tool_call_id or "", "tool"),
                    }
                )
            elif m.role is MessageRole.ASSISTANT and m.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": m.content,
                        "tool_calls": [
                            {"function": {"name": tc.name, "arguments": tc.arguments}}
                            for tc in m.tool_calls
                        ],
                    }
                )
            else:
                messages.append({"role": m.role.value, "content": m.content})
        payload["messages"] = messages
        options: dict[str, Any] = {"num_predict": billable_output_cap(call)}
        if req.temperature is not None:
            options["temperature"] = req.temperature
        payload["options"] = options

        reasoning_applied: str | None = None
        if caps.reasoning is ReasoningStyle.OLLAMA_THINK:
            if call.reasoning_effort is ReasoningEffort.NONE:
                payload["think"] = False
                reasoning_applied = "think=false"
            elif caps.reasoning_efforts:
                level = map_effort(call.reasoning_effort, caps.reasoning_efforts)
                payload["think"] = level
                reasoning_applied = f"think={level}"
            else:
                payload["think"] = True
                reasoning_applied = "think=true"
        elif call.reasoning_effort is not ReasoningEffort.NONE:
            ignored.append("reasoning_effort")

        if req.tools:
            if req.tool_choice == "none":
                ignored.append("tools (tool_choice=none)")
            else:
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
                if req.tool_choice == "required" or isinstance(req.tool_choice, NamedToolChoice):
                    ignored.append("tool_choice (Ollama supports auto only)")
        return PreparedRequest(
            url=f"{call.base_url.rstrip('/')}/api/chat",
            headers={},
            payload=payload,
            ignored=tuple(ignored),
            reasoning_applied=reasoning_applied,
            structured_mode=structured_mode,
        )

    def parse(
        self, call: ProviderCall, prepared: PreparedRequest, data: dict[str, Any]
    ) -> ProviderResult:
        message = data.get("message") or {}
        tool_calls = tuple(
            normalize_tool_call(
                tc.get("id") or f"call_{i}",
                (tc.get("function") or {}).get("name", ""),
                (tc.get("function") or {}).get("arguments") or {},
                call.offered_tools,
            )
            for i, tc in enumerate(message.get("tool_calls") or [])
        )
        raw_finish = data.get("done_reason")
        finish = _FINISH.get(str(raw_finish), FinishReason.OTHER)
        if tool_calls:
            finish = FinishReason.TOOL_CALLS
        return ProviderResult(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=int(data.get("prompt_eval_count") or 0),
                output_tokens=int(data.get("eval_count") or 0),
            ),
            model_version=str(data.get("model") or call.model.model_id),
            finish_reason=finish,
            raw_finish_reason=None if raw_finish is None else str(raw_finish),
            raw_hash=raw_hash(data),
            reasoning_applied=prepared.reasoning_applied,
            ignored=prepared.ignored,
            structured_mode=prepared.structured_mode,
        )
