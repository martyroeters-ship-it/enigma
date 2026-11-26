"""Real local-Ollama structured generation through the ModelGateway (M2)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel, Field

from enigma.core.clock import SystemClock
from enigma.gateway import ModelGateway, ModelRegistry, StructuredLLM, provenance_of
from enigma.gateway.prompts import PromptRegistry
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import Budget
from enigma.store import Stores

pytestmark = pytest.mark.network


class Claim(BaseModel):
    claim: str = Field(min_length=8)
    null_hypothesis: str = Field(min_length=5)


def _ollama_up() -> bool:
    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0)
        return r.status_code == 200 and any(
            "qwen2.5" in m.get("name", "") for m in r.json().get("models", [])
        )
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="local Ollama with qwen2.5 not available")
def test_structured_llm_via_local_ollama(tmp_path: Path) -> None:
    stores = Stores.open(tmp_path / "c.sqlite3")
    gateway = ModelGateway(
        ModelRegistry.load(Path("registries/models")),
        stores.costs,
        tmp_path / "cache.sqlite3",
        SystemClock(),
    )
    assert "ollama" in gateway.available_providers()
    prompt = PromptRegistry.load(Path("registries/prompts")).get("smoke.structured")
    system, user = prompt.render(topic="ETHUSDT funding rates")
    traced = StructuredLLM(gateway).generate_traced(
        Claim,
        system,
        user,
        AgentRole.QUANT_RESEARCH,
        "tsk_ollama_smoke",
        Budget(max_usd=0, max_tokens=4000, max_wall_seconds=600, max_cpu_seconds=600),
        model="local-qwen",
        max_output_tokens=120,
    )
    assert traced.value.claim
    assert traced.response.provider == "ollama"
    assert traced.response.cost_usd == 0.0
    prov = provenance_of(traced.response, prompt.ref)
    assert prov.prompt_ref and prov.prompt_ref.startswith("smoke.structured@")
    assert prov.output_tokens > 0
