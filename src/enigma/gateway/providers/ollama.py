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
# Ollama's server default. This host logged default_num_ctx=4096 and then
# "truncating input prompt" on hypothesis calls whose prompts were longer.
_MIN_NUM_CTX = 4096


def compact_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$ref`` and drop title noise so Ollama's grammar stays small.

    Property names, types, required fields and descriptions are kept. The native
    schema test still sees ``properties.<field>.type``.
    """
    defs = schema.get("$defs") or schema.get("definitions") or {}
    annotations = {"title", "$defs", "definitions", "$schema"}

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            name = str(node["$ref"]).rsplit("/", 1)[-1]
            target = defs.get(name, {})
            extra = {k: v for k, v in node.items() if k != "$ref"}
            resolved = resolve(target)
            if isinstance(resolved, dict):
                return resolve({**resolved, **extra})
            return resolved
        # `properties` keys are field names. A field named title must survive;
        # only the schema annotation of that name is dropped.
        properties = node.get("properties")
        kept: dict[str, Any] = {}
        for key, value in node.items():
            if key in annotations:
                continue
            if key == "properties" and isinstance(properties, dict):
                kept[key] = {name: resolve(spec) for name, spec in properties.items()}
            else:
                kept[key] = resolve(value)
        return kept

    resolved = resolve(schema)
    return resolved if isinstance(resolved, dict) else {}


def _structured_messages(call: ProviderCall) -> tuple[list[dict[str, Any]], str | None]:
    if call.json_schema is None:
        return [], None
    schema = compact_json_schema(call.json_schema)
    caps = call.model.capabilities
    if call.structured_format == "json" and not call.request.tools:
        return [{"role": "system", "content": schema_instruction(schema)}], "json"
    if caps.structured_output is StructuredMode.JSON_SCHEMA and not call.request.tools:
        return [], "native"
    return [{"role": "system", "content": schema_instruction(schema)}], "prompt"


def ollama_num_ctx(call: ProviderCall) -> int:
    """Context length to request so the prompt is not truncated.

    The value is a power of two (or the registry window when that is smaller
    than the next power of two) so similar calls reuse one loaded runner.
    It never exceeds ``model.context_window``.
    """
    from enigma.gateway.gateway import estimate_input_tokens  # noqa: PLC0415

    window = call.model.context_window
    needed = estimate_input_tokens(call.request) + billable_output_cap(call)
    if needed >= window:
        return window
    ctx = window if window < _MIN_NUM_CTX else _MIN_NUM_CTX
    while ctx < needed and ctx * 2 <= window:
        ctx *= 2
    if ctx < needed:
        return window
    return ctx


class OllamaAdapter(ProviderAdapter):
    def build(self, call: ProviderCall) -> PreparedRequest:  # noqa: PLR0912
        req = call.request
        caps = call.model.capabilities
        ignored: list[str] = []
        names = conversation_tool_names(req)
        messages, structured_mode = _structured_messages(call)
        payload: dict[str, Any] = {"model": call.model.model_id, "stream": False}
        if call.json_schema is not None and structured_mode == "json":
            payload["format"] = "json"
        elif call.json_schema is not None and structured_mode == "native":
            payload["format"] = compact_json_schema(call.json_schema)
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
        options: dict[str, Any] = {
            "num_predict": billable_output_cap(call),
            "num_ctx": ollama_num_ctx(call),
        }
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
        text = message.get("content") or ""
        if not str(text).strip() and not tool_calls:
            thinking = message.get("thinking") or ""
            if isinstance(thinking, str) and thinking.lstrip().startswith(("{", "[")):
                text = thinking
        extra: dict[str, Any] = {}
        total_ns = data.get("total_duration")
        if isinstance(total_ns, (int, float)) and not isinstance(total_ns, bool) and total_ns > 0:
            extra["provider_seconds"] = float(total_ns) / 1_000_000_000
        return ProviderResult(
            text=text,
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
            extra=extra,
        )
