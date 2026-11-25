from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cycle_rig import Rig
from fastapi.testclient import TestClient

from enigma.api.app import create_app
from enigma.api.readmodels import HEARTBEAT_PREFIX, ReadModels
from enigma.core.clock import FrozenClock
from enigma.core.config import Settings
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import AgentDecision, ExecutionMode, ModelProvenance


@pytest.fixture
def completed(tmp_path: Path) -> tuple[Rig, str]:
    rig = Rig(tmp_path / "c.sqlite3")
    cycle = rig.submit()
    assert rig.run(cycle) == "COMPLETED"
    return rig, cycle


def test_empty_dashboard_renders(tmp_path: Path, stores) -> None:  # type: ignore[no-untyped-def]
    client = TestClient(create_app(Settings.from_env(), stores))
    for path in (
        "/",
        "/agents",
        "/knowledge",
        "/strategies",
        "/learning",
        "/messages",
        "/models",
        "/audit",
    ):
        r = client.get(path)
        assert r.status_code == 200, path
    assert "No research objective has been submitted" in client.get("/").text
    assert client.get("/cycles/cyc_missing").status_code == 404


def test_dashboard_shows_recorded_cycle(completed: tuple[Rig, str]) -> None:
    rig, cycle = completed
    client = TestClient(create_app(Settings.from_env(), rig.stores))
    home = client.get("/").text
    assert cycle in home and "FAIL" in home
    page = client.get(f"/cycles/{cycle}").text
    for text in (
        "NEXT_EXPERIMENT_PROPOSED",
        "Independent validation",
        "Lessons stored",
        "Proposed next experiment",
        "strategy.implement",
    ):
        assert text in page, text
    for path in ("/agents", "/knowledge", "/strategies", "/learning", "/messages", "/audit"):
        assert client.get(path).status_code == 200, path
    detail = client.get(f"/api/cycles/{cycle}").json()
    assert detail["status"] == "completed"
    assert len(detail["events"]) == 13
    agents = client.get("/api/agents").json()
    assert len(agents) == 12
    assert client.get("/api/health").json()["audit_chain_ok"] is True


def test_worker_without_recent_heartbeat_is_shown_stale(stores) -> None:  # type: ignore[no-untyped-def]
    stores.flags.set(HEARTBEAT_PREFIX + "host:1", "running", "worker")
    stores.flags.set(HEARTBEAT_PREFIX + "host:2", "stopped", "worker")
    later = FrozenClock(datetime.now(UTC) + timedelta(minutes=5))
    status = {w["worker"]: w["status"] for w in ReadModels(stores, later).health()["workers"]}
    assert status == {"host:1": "stale (no heartbeat)", "host:2": "stopped"}
    fresh = {w["worker"]: w["status"] for w in ReadModels(stores).health()["workers"]}
    assert fresh["host:1"] == "running"


def test_model_usage_distinguishes_execution_modes(stores) -> None:  # type: ignore[no-untyped-def]
    now = datetime.now(UTC)
    stores.documents.put(
        AgentDecision(
            id="dec_llm_1",
            agent=AgentRole.QUANT_RESEARCH,
            task_id="tsk_1",
            decision_type="hypothesis",
            outcome="proposed",
            reason_code="llm",
            reasoning_summary="model wrote a hypothesis",
            instructions_version="1",
            created_at=now,
            execution_mode=ExecutionMode.LLM,
            model_calls=(
                ModelProvenance(
                    provider="ollama",
                    model_id="qwen2.5:3b",
                    model_version="digest",
                    request_hash="sha256:" + "a" * 64,
                    raw_response_hash="sha256:" + "b" * 64,
                    prompt_ref="quant.hypothesis@1#abc",
                    input_tokens=10,
                    output_tokens=20,
                    cost_usd=0.0,
                    latency_ms=12.0,
                ),
            ),
        ),
        recorded_by="test",
    )
    stores.documents.put(
        AgentDecision(
            id="dec_det_1",
            agent=AgentRole.CIO,
            task_id=None,
            decision_type="research_prioritization",
            outcome="none",
            reason_code="max_expected_value",
            reasoning_summary="ranked",
            instructions_version="1",
            created_at=now,
            execution_mode=ExecutionMode.DETERMINISTIC,
        ),
        recorded_by="test",
    )
    usage = ReadModels(stores).model_usage()
    assert usage["decisions_by_execution_mode"]["llm"] == 1
    assert usage["decisions_by_execution_mode"]["deterministic"] == 1
    assert usage["total_model_calls"] == 1
    assert usage["total_tokens"] == 30
    client = TestClient(create_app(Settings.from_env(), stores))
    assert client.get("/models").status_code == 200
    assert client.get("/api/models").json()["total_model_calls"] == 1
