"""Anthropic Messages API (`/v1/messages`): tools, extended-thinking budget, tool-forced JSON."""

from __future__ import annotations

import json
from typing import Any

from enigma.gateway.providers.base import (
    REASONING_BUDGET_TOKENS,
    PreparedRequest,
    ProviderAdapter,
    ProviderCall,
    ProviderResult,
    normalize_tool_call,
    raw_hash,
    schema_instruction,
)
from enigma.gateway.registry import ReasoningStyle
from enigma.gateway.types import (
    FinishReason,
    MessageRole,
    NamedToolChoice,
    ReasoningEffort,
    ToolCall,
    Usage,
)

ANTHROPIC_VERSION = "2023-06-01"
STRUCTURED_TOOL = "emit_structured_output"
_MIN_THINKING_BUDGET = 1024

_FINISH = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL_CALLS,
    "refusal": FinishReason.CONTENT_FILTER,
}


def _messages(call: ProviderCall) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []

    def push(role: str, blocks: list[dict[str, Any]]) -> None:
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for m in call.request.messages:
        if m.role is MessageRole.SYSTEM:
            system_parts.append(m.content)
        elif m.role is MessageRole.TOOL:
            push(
                "user",
                [{"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}],
            )
        elif m.role is MessageRole.ASSISTANT:
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            blocks.extend(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                for tc in m.tool_calls
            )
            push("assistant", blocks)
        else:
            push("user", [{"type": "text", "text": m.content}])
    return "\n\n".join(system_parts), out


class AnthropicAdapter(ProviderAdapter):
    def build(self, call: ProviderCall) -> PreparedRequest:  # noqa: PLR0912, PLR0915
        req = call.request
        caps = call.model.capabilities
        ignored: list[str] = []
        system, messages = _messages(call)
        payload: dict[str, Any] = {"model": call.model.model_id, "messages": messages}
        tools: list[dict[str, Any]] = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in req.tools
        ]
        tool_choice: dict[str, Any] | None = None
        if req.tools:
            if isinstance(req.tool_choice, NamedToolChoice):
                tool_choice = {"type": "tool", "name": req.tool_choice.name}
            else:
                tool_choice = {"type": {"required": "any"}.get(req.tool_choice, req.tool_choice)}

        structured_mode: str | None = None
        if call.json_schema is not None:
            if not req.tools:
                tools = [
                    {
                        "name": STRUCTURED_TOOL,
                        "description": "Return the final answer as structured data.",
                        "input_schema": call.json_schema,
                    }
                ]
                tool_choice = {"type": "tool", "name": STRUCTURED_TOOL}
                structured_mode = "tool"
            else:
                instruction = schema_instruction(call.json_schema)
                system = f"{system}\n\n{instruction}" if system else instruction
                structured_mode = "prompt"

        thinking_budget = 0
        wants_thinking = call.reasoning_effort is not ReasoningEffort.NONE
        forced_tool = tool_choice is not None and tool_choice["type"] in ("any", "tool")
        if wants_thinking:
            if caps.reasoning is not ReasoningStyle.ANTHROPIC_BUDGET:
                ignored.append("reasoning_effort")
            elif forced_tool:
                ignored.append("reasoning_effort (thinking incompatible with forced tool use)")
            else:
                thinking_budget = max(
                    _MIN_THINKING_BUDGET, REASONING_BUDGET_TOKENS[call.reasoning_effort]
                )
        max_tokens = min(call.output_token_cap + thinking_budget, call.model.max_output_tokens)
        if thinking_budget and thinking_budget >= max_tokens:
            thinking_budget = 0
            ignored.append("reasoning_effort (budget exceeds model output limit)")
            max_tokens = min(call.output_token_cap, call.model.max_output_tokens)
        payload["max_tokens"] = max_tokens
        reasoning_applied: str | None = None
        if thinking_budget:
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            reasoning_applied = f"thinking.budget_tokens={thinking_budget}"
            if req.temperature is not None and req.temperature != 1.0:
                ignored.append("temperature (extended thinking requires default)")
        elif req.temperature is not None:
            if caps.temperature:
                payload["temperature"] = min(req.temperature, 1.0)
                if req.temperature > 1.0:
                    ignored.append("temperature>1 (clamped to 1.0)")
            else:
                ignored.append("temperature")
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        headers = {"anthropic-version": ANTHROPIC_VERSION}
        if call.api_key:
            headers["x-api-key"] = call.api_key
        return PreparedRequest(
            url=f"{call.base_url.rstrip('/')}/v1/messages",
            headers=headers,
            payload=payload,
            ignored=tuple(ignored),
            reasoning_applied=reasoning_applied,
            structured_mode=structured_mode,
        )

    def parse(
        self, call: ProviderCall, prepared: PreparedRequest, data: dict[str, Any]
    ) -> ProviderResult:
        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        structured_text: str | None = None
        for block in data.get("content") or []:
            kind = block.get("type")
            if kind == "text":
                texts.append(block.get("text", ""))
            elif kind == "tool_use":
                if prepared.structured_mode == "tool" and block.get("name") == STRUCTURED_TOOL:
                    structured_text = json.dumps(block.get("input") or {}, sort_keys=True)
                else:
                    tool_calls.append(
                        normalize_tool_call(
                            block.get("id", f"call_{len(tool_calls)}"),
                            block.get("name", ""),
                            block.get("input") or {},
                            call.offered_tools,
                        )
                    )
        raw_finish = data.get("stop_reason")
        finish = _FINISH.get(str(raw_finish), FinishReason.OTHER)
        if structured_text is not None and finish is FinishReason.TOOL_CALLS:
            finish = FinishReason.STOP
        usage = data.get("usage") or {}
        return ProviderResult(
            text=structured_text if structured_text is not None else "".join(texts),
            tool_calls=tuple(tool_calls),
            usage=Usage(
                input_tokens=int(usage.get("input_tokens") or 0)
                + int(usage.get("cache_creation_input_tokens") or 0)
                + int(usage.get("cache_read_input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                cached_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
            ),
            model_version=str(data.get("model") or call.model.model_id),
            finish_reason=finish,
            raw_finish_reason=None if raw_finish is None else str(raw_finish),
            raw_hash=raw_hash(data),
            reasoning_applied=prepared.reasoning_applied,
            ignored=prepared.ignored,
            structured_mode=prepared.structured_mode,
        )
