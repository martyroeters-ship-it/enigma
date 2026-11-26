"""Provider adapters over raw HTTP (httpx). No vendor SDKs."""

from __future__ import annotations

import httpx

from enigma.gateway.providers.anthropic import AnthropicAdapter
from enigma.gateway.providers.base import (
    PreparedRequest,
    ProviderAdapter,
    ProviderCall,
    ProviderResult,
    billable_output_cap,
)
from enigma.gateway.providers.gemini import GeminiAdapter
from enigma.gateway.providers.ollama import OllamaAdapter
from enigma.gateway.providers.openai import OpenAIAdapter, OpenAICompatibleAdapter
from enigma.gateway.registry import ProviderKind

ADAPTERS: dict[ProviderKind, type[ProviderAdapter]] = {
    ProviderKind.OPENAI: OpenAIAdapter,
    ProviderKind.OPENAI_COMPATIBLE: OpenAICompatibleAdapter,
    ProviderKind.ANTHROPIC: AnthropicAdapter,
    ProviderKind.GEMINI: GeminiAdapter,
    ProviderKind.OLLAMA: OllamaAdapter,
}


def make_adapter(kind: ProviderKind, client: httpx.Client) -> ProviderAdapter:
    return ADAPTERS[kind](client)


__all__ = [
    "ADAPTERS",
    "AnthropicAdapter",
    "GeminiAdapter",
    "OllamaAdapter",
    "OpenAIAdapter",
    "OpenAICompatibleAdapter",
    "PreparedRequest",
    "ProviderAdapter",
    "ProviderCall",
    "ProviderResult",
    "billable_output_cap",
    "make_adapter",
]
