"""Ollama must be asked for a context that fits the prompt.

The server default is 4096. A longer prompt is truncated, and the hypothesis
repair round starts from a prefix of the request.
"""

from __future__ import annotations

import httpx

from enigma.gateway.gateway import estimate_input_tokens
from enigma.gateway.providers.base import ProviderCall, billable_output_cap
from enigma.gateway.providers.ollama import OllamaAdapter, ollama_num_ctx
from enigma.gateway.registry import ModelSpec, ProviderSpec
from enigma.gateway.types import Message, MessageRole, ModelRequest, ReasoningEffort


def _call(content: str, *, cap: int = 256, window: int = 32768) -> ProviderCall:
    return ProviderCall(
        request=ModelRequest(
            messages=(Message(role=MessageRole.USER, content=content),),
            model="local-qwen",
            temperature=0.0,
        ),
        model=ModelSpec(
            alias="local-qwen",
            provider="ollama",
            model_id="qwen2.5:7b-instruct",
            context_window=window,
            max_output_tokens=8192,
        ),
        provider=ProviderSpec(
            name="ollama",
            kind="ollama",
            base_url="http://127.0.0.1:11434",
            key_required=False,
            local=True,
        ),
        base_url="http://127.0.0.1:11434",
        api_key=None,
        output_token_cap=cap,
        reasoning_effort=ReasoningEffort.NONE,
        json_schema=None,
    )


def test_short_prompt_stays_on_the_server_default() -> None:
    assert ollama_num_ctx(_call("hi")) == 4096


def test_long_prompt_grows_within_the_registry_window() -> None:
    call = _call("funding crowding " * 8000, cap=1024)
    ctx = ollama_num_ctx(call)
    needed = estimate_input_tokens(call.request) + billable_output_cap(call)
    assert needed > 4096
    assert ctx >= min(needed, call.model.context_window)
    assert ctx <= call.model.context_window
    assert ctx > 4096
    payload = OllamaAdapter(httpx.Client()).build(call).payload
    assert payload["options"]["num_ctx"] == ctx
    assert payload["options"]["num_predict"] == 1024


def test_num_ctx_never_exceeds_a_small_window() -> None:
    call = _call("x " * 20000, window=2048, cap=128)
    assert ollama_num_ctx(call) == 2048
