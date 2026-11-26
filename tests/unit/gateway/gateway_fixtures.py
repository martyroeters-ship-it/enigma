"""Fixtures for gateway unit tests.

HTTP is intercepted with respx: these tests exercise adapter wire formats and gateway policy,
they are not evidence that any live provider works (see tests/network for that).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from enigma.core.clock import FrozenClock
from enigma.gateway.gateway import ModelGateway
from enigma.gateway.registry import ModelRegistry
from enigma.store.db import Database
from enigma.store.ledger import CostLedger

NOW = datetime(2026, 11, 25, 10, 0, tzinfo=UTC)
OPENAI_URL = "https://api.openai.test/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.test/v1/messages"
GEMINI_BASE = "https://gemini.test/v1beta"
COMPAT_URL = "https://compat.test/v1/chat/completions"
OLLAMA_BASE = "http://ollama.test:11434"
FAKE_OPENAI_KEY = "sk-test-SECRET-openai-0123456789"
FAKE_ANTHROPIC_KEY = "sk-ant-test-SECRET-0123456789"
FAKE_GEMINI_KEY = "AIzaTEST-SECRET-gemini-0123456789abcd"
FAKE_COMPAT_KEY = "compat-SECRET-0123456789"

MODELS_DOC: dict[str, Any] = {
    "providers": {
        "openai": {
            "kind": "openai",
            "base_url": "https://api.openai.test/v1",
            "api_key_env": ["OPENAI_API_KEY"],
            "max_tokens_param": "max_completion_tokens",
        },
        "anthropic": {
            "kind": "anthropic",
            "base_url": "https://api.anthropic.test",
            "api_key_env": ["ANTHROPIC_API_KEY"],
        },
        "google": {
            "kind": "gemini",
            "base_url": GEMINI_BASE,
            "api_key_env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        },
        "compat": {
            "kind": "openai_compatible",
            "base_url": "https://compat.test/v1",
            "api_key_env": ["COMPAT_API_KEY"],
        },
        "localvllm": {
            "kind": "openai_compatible",
            "base_url_env": "VLLM_BASE_URL",
            "api_key_env": ["VLLM_API_KEY"],
            "key_required": False,
            "local": True,
        },
        "ollama": {
            "kind": "ollama",
            "base_url": OLLAMA_BASE,
            "base_url_env": "OLLAMA_HOST",
            "key_required": False,
            "local": True,
            "probe_path": "/api/version",
        },
    },
    "models": {
        "o-reason": {
            "provider": "openai",
            "model_id": "gpt-reason",
            "context_window": 100000,
            "max_output_tokens": 50000,
            "capabilities": {
                "structured_output": "json_schema",
                "tools": True,
                "reasoning": "openai_effort",
                "reasoning_efforts": ["minimal", "low", "medium", "high"],
                "temperature": False,
            },
        },
        "o-plain": {
            "provider": "openai",
            "model_id": "gpt-plain",
            "context_window": 100000,
            "max_output_tokens": 8000,
            "capabilities": {"structured_output": "json_schema", "tools": True},
        },
        "claude": {
            "provider": "anthropic",
            "model_id": "claude-test",
            "context_window": 200000,
            "max_output_tokens": 64000,
            "capabilities": {
                "structured_output": "json_schema",
                "tools": True,
                "reasoning": "anthropic_budget",
            },
        },
        "gem": {
            "provider": "google",
            "model_id": "gemini-test",
            "context_window": 1000000,
            "max_output_tokens": 65536,
            "capabilities": {
                "structured_output": "json_schema",
                "tools": True,
                "reasoning": "gemini_budget",
                "reasoning_efforts": ["none", "low", "medium", "high"],
            },
        },
        "compat": {
            "provider": "compat",
            "model_id": "compat-model",
            "context_window": 32000,
            "max_output_tokens": 4000,
            "capabilities": {"structured_output": "json_mode", "tools": True},
        },
        "compat-notools": {
            "provider": "compat",
            "model_id": "compat-small",
            "context_window": 32000,
            "max_output_tokens": 4000,
            "capabilities": {"structured_output": "prompt", "tools": False},
        },
        "vllm": {
            "provider": "localvllm",
            "model_id": "served-model",
            "context_window": 32000,
            "max_output_tokens": 4000,
            "capabilities": {"structured_output": "json_schema", "tools": True},
        },
        "local": {
            "provider": "ollama",
            "model_id": "llama-test",
            "context_window": 32000,
            "max_output_tokens": 4000,
            "capabilities": {
                "structured_output": "json_schema",
                "tools": True,
                "reasoning": "ollama_think",
            },
        },
    },
}

PRICING_DOC: dict[str, Any] = {
    "as_of": "2026-01-01",
    "estimate": True,
    "prices": {
        "openai/gpt-reason": {"input": 1.0, "output": 4.0, "source": "https://example.test"},
        "openai/gpt-plain": {
            "input": 2.0,
            "output": 8.0,
            "cached_input": 0.5,
            "source": "https://example.test",
        },
        "anthropic/claude-test": {"input": 3.0, "output": 15.0, "source": "https://example.test"},
        "google/gemini-test": {"input": 0.3, "output": 2.5, "source": "https://example.test"},
        "compat/compat-model": {"input": 0.5, "output": 0.5, "source": "https://example.test"},
    },
}

ROUTING_DOC: dict[str, Any] = {
    "defaults": {
        "candidates": ["o-plain", "claude", "local"],
        "reasoning_effort": "none",
        "max_cost_per_call_usd": 1.0,
        "max_output_tokens": 1000,
    },
    "roles": {
        "intelligence": {
            "default": {"candidates": ["claude", "o-plain", "local"]},
            "tasks": {
                "intelligence.extract": {
                    "candidates": ["gem", "o-plain"],
                    "reasoning_effort": "low",
                    "max_cost_per_call_usd": 0.5,
                }
            },
        },
        "quant_research": {"default": {"reasoning_effort": "medium", "max_output_tokens": 2000}},
    },
}

ALL_KEYS = {
    "OPENAI_API_KEY": FAKE_OPENAI_KEY,
    "ANTHROPIC_API_KEY": FAKE_ANTHROPIC_KEY,
    "GEMINI_API_KEY": FAKE_GEMINI_KEY,
    "COMPAT_API_KEY": FAKE_COMPAT_KEY,
    "ENIGMA_MAX_DAILY_API_SPEND_USD": "10",
}


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry.from_documents(MODELS_DOC, PRICING_DOC, ROUTING_DOC)


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[CostLedger]:
    db = Database(tmp_path / "control.sqlite3")
    yield CostLedger(db)
    db.close()


class SleepRecorder:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture
def sleeps() -> SleepRecorder:
    return SleepRecorder()


GatewayFactory = Callable[..., ModelGateway]


@pytest.fixture
def make_gateway(
    registry: ModelRegistry, ledger: CostLedger, tmp_path: Path, sleeps: SleepRecorder
) -> Iterator[GatewayFactory]:
    created: list[ModelGateway] = []

    def factory(env: dict[str, str] | None = None, **kwargs: Any) -> ModelGateway:
        ticks = iter(range(0, 10_000_000, 250))
        options: dict[str, Any] = {
            "environ": dict(ALL_KEYS if env is None else env),
            "sleep": sleeps,
            "rng": random.Random(7),  # noqa: S311 - deterministic jitter
            "monotonic": lambda: next(ticks) / 1000.0,
        }
        options.update(kwargs)
        gw = ModelGateway(
            options.pop("registry", registry),
            ledger,
            options.pop("cache_db_path", tmp_path / "cache.sqlite3"),
            FrozenClock(NOW),
            **options,
        )
        created.append(gw)
        return gw

    yield factory
    for gw in created:
        gw.close()


def openai_reply(
    content: str | None = "hello",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    finish: str = "stop",
    model: str = "gpt-plain-2026-01-01",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    reasoning_tokens: int = 0,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model,
        "system_fingerprint": "fp_test",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }


def anthropic_reply(
    content: list[dict[str, Any]],
    *,
    stop_reason: str = "end_turn",
    model: str = "claude-test-20260101",
) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 120, "output_tokens": 30, "cache_read_input_tokens": 20},
    }
