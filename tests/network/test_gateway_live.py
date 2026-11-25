"""Live provider calls. Each test runs only when its credential env var is set (and, for paid
providers, ENIGMA_MAX_DAILY_API_SPEND_USD > 0); otherwise it is skipped, never faked."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from enigma.core.clock import SystemClock
from enigma.gateway import ModelGateway, ModelRegistry, StructuredLLM
from enigma.gateway.gateway import DAILY_CAP_ENV
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import Budget
from enigma.store.db import Database
from enigma.store.ledger import CostLedger

pytestmark = pytest.mark.network

REGISTRY_DIR = Path(__file__).resolve().parents[2] / "registries" / "models"
BUDGET = Budget(max_usd=0.05, max_tokens=20_000, max_http_requests=10)


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset: str
    hours: int


PAID = [
    ("gpt-5-mini", "OPENAI_API_KEY"),
    ("claude-haiku", "ANTHROPIC_API_KEY"),
    ("gemini-flash", "GEMINI_API_KEY"),
    ("openrouter-sonnet", "OPENROUTER_API_KEY"),
    ("deepseek-chat", "DEEPSEEK_API_KEY"),
    ("groq-llama-70b", "GROQ_API_KEY"),
    ("mistral-large", "MISTRAL_API_KEY"),
    ("together-llama-70b", "TOGETHER_API_KEY"),
]


@pytest.fixture
def gateway(tmp_path: Path) -> Iterator[ModelGateway]:
    db = Database(tmp_path / "control.sqlite3")
    gw = ModelGateway(
        ModelRegistry.load(REGISTRY_DIR), CostLedger(db), tmp_path / "cache.sqlite3", SystemClock()
    )
    yield gw
    gw.close()
    db.close()


def _ask(gw: ModelGateway, alias: str) -> Answer:
    return StructuredLLM(gw).generate(
        Answer,
        "Extract fields exactly as stated.",
        "Hypothesis: BTC returns are negative over the next 24 hours after high funding.",
        AgentRole.TECHNOLOGY_INNOVATION,
        "live_test",
        BUDGET,
        model=alias,
        max_output_tokens=200,
    )


@pytest.mark.parametrize(("alias", "env_var"), PAID)
def test_live_structured_call(gateway: ModelGateway, alias: str, env_var: str) -> None:
    if alias == "gemini-flash" and not (
        os.environ.get(env_var) or os.environ.get("GOOGLE_API_KEY")
    ):
        pytest.skip(f"{env_var}/GOOGLE_API_KEY not set")
    elif alias != "gemini-flash" and not os.environ.get(env_var):
        pytest.skip(f"{env_var} not set")
    if float(os.environ.get(DAILY_CAP_ENV, "0") or 0) <= 0:
        pytest.skip(f"{DAILY_CAP_ENV} is 0; paid calls are disabled")
    out = _ask(gateway, alias)
    assert out.asset.upper() == "BTC"
    assert out.hours == 24


def test_live_ollama(gateway: ModelGateway) -> None:
    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    base = base if "://" in base else f"http://{base}"
    try:
        tags = httpx.get(f"{base}/api/tags", timeout=2.0).json()
    except httpx.HTTPError:
        pytest.skip(f"Ollama not reachable at {base}")
    model_id = gateway.registry.models["local-qwen"].model_id
    if model_id not in {m.get("name") for m in tags.get("models", [])}:
        pytest.skip(f"Ollama model {model_id} not pulled")
    out = _ask(gateway, "local-qwen")
    assert out.asset.upper() == "BTC"
