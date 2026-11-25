from __future__ import annotations

import os
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cycle_rig import Rig
from fastapi.testclient import TestClient

from enigma.api.app import create_app
from enigma.api.readmodels import HEARTBEAT_PREFIX, ReadModels
from enigma.core.clock import FrozenClock
from enigma.core.config import Settings


@pytest.fixture
def completed(tmp_path: Path) -> tuple[Rig, str]:
    rig = Rig(tmp_path / "c.sqlite3")
    cycle = rig.submit()
    assert rig.run(cycle) == "COMPLETED"
    return rig, cycle


def test_empty_dashboard_renders(tmp_path: Path, stores) -> None:  # type: ignore[no-untyped-def]
    client = TestClient(create_app(Settings.from_env(), stores))
    for path in ("/", "/agents", "/knowledge", "/strategies", "/learning", "/messages", "/audit"):
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


def test_dead_local_worker_is_not_called_stale(stores) -> None:  # type: ignore[no-untyped-def]
    dead = f"{socket.gethostname()}:999999"
    live = f"{socket.gethostname()}:{os.getpid()}"
    stores.flags.set(HEARTBEAT_PREFIX + dead, "running: quant.hypothesize", "worker")
    stores.flags.set(HEARTBEAT_PREFIX + live, "running", "worker")
    later = FrozenClock(datetime.now(UTC) + timedelta(minutes=5))
    status = {w["worker"]: w["status"] for w in ReadModels(stores, later).health()["workers"]}
    assert status[dead] == "stopped (process exited)"
    assert status[live] == "stale (no heartbeat)"
