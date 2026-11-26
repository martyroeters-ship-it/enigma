from __future__ import annotations

import os
import socket
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cycle_rig import Rig

from enigma.api.readmodels import ReadModels
from enigma.core.config import Settings
from enigma.core.ids import new_id
from enigma.fsm import RESEARCH_CYCLE
from enigma.orchestrator.daemon import Daemon, Schedule, local_worker_alive
from enigma.orchestrator.institution import Institution
from enigma.orchestrator.tech import TechnologyInnovationAgent
from enigma.schemas.common import AgentRole
from enigma.schemas.cycle import NextExperimentProposal, ProposalState
from enigma.schemas.governance import Task, TaskStatus


def _daemon(rig: Rig, **overrides: Any) -> Daemon:
    rig.worker.agents[AgentRole.TECHNOLOGY_INNOVATION] = TechnologyInnovationAgent()
    rig.worker.roles = tuple(rig.worker.agents)
    inst = Institution(
        Settings.from_env(),
        rig.stores,
        rig.bus,
        rig.fsm,
        rig.contracts,
        rig.driver.policy,
        rig.driver,
        rig.worker,
        rig.fakes.services(),
        rig.driver.clock,
    )
    raw = {
        "poll_interval_seconds": 0.0,
        "heartbeat_seconds": 0.0,
        "auto_next_experiment": True,
        "max_cycles_per_day": 2,
        "max_concurrent_cycles": 1,
        "tech_scan_interval_hours": 0.0,
        "tech_scan_topics": (),
    }
    raw.update(overrides)
    return Daemon(inst, Schedule(**raw))


def test_daemon_chains_next_experiment_up_to_daily_cap(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    first = rig.submit()
    daemon = _daemon(rig)
    daemon.run(max_iterations=400)
    cycles = rig.stores.checkpoints.list("research_cycle")
    assert len(cycles) == 2  # parent + one chained child; cap of 2 per day stops further chaining
    child = next(c for c in cycles if c["workflow_id"] != first)
    assert rig.fsm.state(RESEARCH_CYCLE, child["workflow_id"]) == "COMPLETED"
    parent_proposal = rig.stores.documents.list(
        NextExperimentProposal, where={"parent_cycle_id": first}
    )[0]
    assert parent_proposal.state is ProposalState.SCHEDULED
    assert parent_proposal.child_cycle_id == child["workflow_id"]
    assert child["state"]["proposal_id"] == parent_proposal.id
    beats = [k for k in rig.stores.db.query("SELECT name FROM flags WHERE name LIKE 'heartbeat:%'")]
    assert beats


def test_tech_scan_is_scheduled_once_per_window(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    daemon = _daemon(rig, tech_scan_interval_hours=24.0, tech_scan_topics=("backtesting",))
    daemon.run(max_iterations=5)
    scans = [t for t in rig.stores.queue.list(limit=50) if t.kind == "tech.scan"]
    assert len(scans) == 1
    assert scans[0].status is TaskStatus.SUCCEEDED
    assert scans[0].outputs["source_ids"]


def test_kill_switch_stops_daemon_work(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    rig.submit()
    rig.stores.flags.set("kill_switch", "on", "human")
    daemon = _daemon(rig)
    assert daemon.step() is False
    assert rig.stores.queue.counts().get("pending", 0) == 0


def test_local_worker_liveness() -> None:
    assert local_worker_alive(f"{socket.gethostname()}:{os.getpid()}")
    assert not local_worker_alive(f"{socket.gethostname()}:999999")
    assert local_worker_alive("other-host:1")


def test_heartbeat_pulses_and_lease_renews_while_run_once_blocks(tmp_path: Path) -> None:
    """A model call blocks inside one step. The one-minute stale label must not trip,
    and the task lease must move out so another worker cannot claim the same task."""
    rig = Rig(tmp_path / "c.sqlite3")
    daemon = _daemon(rig, heartbeat_seconds=0.05)
    now = datetime.now(UTC)
    task = Task(
        id=new_id("tsk"),
        kind="quant.hypothesize",
        assigned_to=AgentRole.QUANT_RESEARCH,
        created_by=AgentRole.CIO,
        correlation_id="cyc_pulse",
        created_at=now,
        updated_at=now,
    )
    rig.stores.queue.enqueue(task)
    claimed = rig.stores.queue.claim(daemon.worker_name, lease=timedelta(seconds=1), now=now)
    assert claimed is not None
    started = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        started.set()
        assert release.wait(timeout=5)

    daemon.inst.worker.run_once = blocked  # type: ignore[method-assign]
    worker = threading.Thread(target=daemon.step)
    worker.start()
    assert started.wait(timeout=2)
    fresh = False
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        row = rig.stores.db.query_one(
            "SELECT value, updated_at FROM flags WHERE name = ?",
            (f"heartbeat:{daemon.worker_name}",),
        )
        lease = rig.stores.db.query_one(
            "SELECT lease_expires_at FROM task_queue WHERE id = ?",
            (task.id,),
        )
        if row is not None and lease is not None and str(row["value"]).startswith("running"):
            seen = datetime.fromisoformat(str(row["updated_at"]))
            expires = datetime.fromisoformat(str(lease["lease_expires_at"]))
            recent = datetime.now(UTC) - seen < timedelta(seconds=2)
            extended = expires > datetime.now(UTC) + timedelta(minutes=5)
            if recent and extended:
                fresh = True
                break
        time.sleep(0.05)
    status = ReadModels(rig.stores).health()["workers"][0]["status"]
    release.set()
    worker.join(timeout=5)
    assert fresh, "heartbeat was not refreshed while run_once was blocked"
    assert status.startswith("running")
    assert "stale" not in status
    assert not worker.is_alive()
    assert rig.stores.flags.get(f"heartbeat:{daemon.worker_name}") == "idle"
