"""Google Gemini `models/{id}:generateContent`: responseSchema, functionDeclarations, thinking."""

from __future__ import annotations

import json
from typing import Any

from enigma.gateway.jsonschema import inline_refs, to_gemini_schema
from enigma.gateway.providers.base import (
    REASONING_BUDGET_TOKENS,
    PreparedRequest,
    ProviderAdapter,
    ProviderCall,
    ProviderResult,
    billable_output_cap,
    conversation_tool_names,
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
    ToolCall,
    Usage,
)

_FINISH = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.CONTENT_FILTER,
    "RECITATION": FinishReason.CONTENT_FILTER,
    "BLOCKLIST": FinishReason.CONTENT_FILTER,
    "PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
    "SPII": FinishReason.CONTENT_FILTER,
}


def _contents(call: ProviderCall) -> tuple[list[str], list[dict[str, Any]]]:
    system: list[str] = []
    out: list[dict[str, Any]] = []
    names = conversation_tool_names(call.request)

    def push(role: str, parts: list[dict[str, Any]]) -> None:
        if out and out[-1]["role"] == role:
            out[-1]["parts"].extend(parts)
        else:
            out.append({"role": role, "parts": parts})

    for m in call.request.messages:
        if m.role is MessageRole.SYSTEM:
            system.append(m.content)
        elif m.role is MessageRole.TOOL:
            name = m.name or names.get(m.tool_call_id or "", "tool")
            try:
                response: Any = json.loads(m.content)
            except json.JSONDecodeError:
                response = m.content
            if not isinstance(response, dict):
                response = {"result": response}
            push("user", [{"functionResponse": {"name": name, "response": response}}])
        elif m.role is MessageRole.ASSISTANT:
            parts: list[dict[str, Any]] = []
            if m.content:
                parts.append({"text": m.content})
            parts.extend(
                {"functionCall": {"name": tc.name, "args": tc.arguments}} for tc in m.tool_calls
            )
            push("model", parts)
        else:
            push("user", [{"text": m.content}])
    return system, out


class GeminiAdapter(ProviderAdapter):
    def build(self, call: ProviderCall) -> PreparedRequest:  # noqa: PLR0912
        req = call.request
        caps = call.model.capabilities
        ignored: list[str] = []
        system, contents = _contents(call)
        gen: dict[str, Any] = {"maxOutputTokens": billable_output_cap(call)}
        structured_mode: str | None = None
        if call.json_schema is not None:
            if caps.structured_output is StructuredMode.JSON_SCHEMA and not req.tools:
                gen["responseMimeType"] = "application/json"
                gen["responseSchema"] = to_gemini_schema(call.json_schema)
                structured_mode = "native"
            else:
                system.append(schema_instruction(inline_refs(call.json_schema)))
                structured_mode = "prompt"
        if req.temperature is not None:
            if caps.temperature:
                gen["temperature"] = req.temperature
            else:
                ignored.append("temperature")
        reasoning_applied: str | None = None
        if caps.reasoning is ReasoningStyle.GEMINI_BUDGET:
            budget = REASONING_BUDGET_TOKENS[call.reasoning_effort]
            if budget == 0 and "none" not in caps.reasoning_efforts:
                budget = REASONING_BUDGET_TOKENS[ReasoningEffort.LOW]
                ignored.append("reasoning_effort=none (model cannot disable thinking)")
            gen["thinkingConfig"] = {"thinkingBudget": budget}
            reasoning_applied = f"thinkingBudget={budget}"
        elif call.reasoning_effort is not ReasoningEffort.NONE:
            ignored.append("reasoning_effort")

        payload: dict[str, Any] = {"contents": contents, "generationConfig": gen}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
        if req.tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": to_gemini_schema(t.parameters),
                        }
                        for t in req.tools
                    ]
                }
            ]
            fc: dict[str, Any]
            if isinstance(req.tool_choice, NamedToolChoice):
                fc = {"mode": "ANY", "allowedFunctionNames": [req.tool_choice.name]}
            else:
                fc = {"mode": {"auto": "AUTO", "none": "NONE", "required": "ANY"}[req.tool_choice]}
            payload["toolConfig"] = {"functionCallingConfig": fc}
        headers = {"x-goog-api-key": call.api_key} if call.api_key else {}
        return PreparedRequest(
            url=f"{call.base_url.rstrip('/')}/models/{call.model.model_id}:generateContent",
            headers=headers,
            payload=payload,
            ignored=tuple(ignored),
            reasoning_applied=reasoning_applied,
            structured_mode=structured_mode,
        )

    def parse(
        self, call: ProviderCall, prepared: PreparedRequest, data: dict[str, Any]
    ) -> ProviderResult:
        candidates = data.get("candidates") or []
        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_finish: str | None = None
        if candidates:
            cand = candidates[0]
            raw_finish = cand.get("finishReason")
            for part in (cand.get("content") or {}).get("parts") or []:
                if part.get("thought"):
                    continue
                if "text" in part:
                    texts.append(part["text"])
                elif "functionCall" in part:
                    fn = part["functionCall"]
                    tool_calls.append(
                        normalize_tool_call(
                            fn.get("id") or f"call_{len(tool_calls)}",
                            fn.get("name", ""),
                            fn.get("args") or {},
                            call.offered_tools,
                        )
                    )
        else:
            raw_finish = (data.get("promptFeedback") or {}).get("blockReason") or "NO_CANDIDATES"
        finish = _FINISH.get(str(raw_finish), FinishReason.OTHER)
        if tool_calls and finish is FinishReason.STOP:
            finish = FinishReason.TOOL_CALLS
        um = data.get("usageMetadata") or {}
        thoughts = int(um.get("thoughtsTokenCount") or 0)
        return ProviderResult(
            text="".join(texts),
            tool_calls=tuple(tool_calls),
            usage=Usage(
                input_tokens=int(um.get("promptTokenCount") or 0),
                output_tokens=int(um.get("candidatesTokenCount") or 0) + thoughts,
                reasoning_tokens=thoughts,
                cached_input_tokens=int(um.get("cachedContentTokenCount") or 0),
            ),
            model_version=str(data.get("modelVersion") or call.model.model_id),
            finish_reason=finish,
            raw_finish_reason=raw_finish,
            raw_hash=raw_hash(data),
            reasoning_applied=prepared.reasoning_applied,
            ignored=prepared.ignored,
            structured_mode=prepared.structured_mode,
        )
