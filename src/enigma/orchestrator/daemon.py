"""Long-running autonomous loop: drive cycles, execute tasks, chain experiments, run schedules."""

from __future__ import annotations

import os
import signal
import socket
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import FrameType
from typing import Any

import yaml

from enigma.agents.base import TaskRequest, delegate
from enigma.api.readmodels import HEARTBEAT_PREFIX, ReadModels
from enigma.core.logging import get_logger
from enigma.fsm import RESEARCH_CYCLE
from enigma.orchestrator.institution import Institution
from enigma.schemas.common import AgentRole
from enigma.schemas.cycle import NextExperimentProposal, ProposalState
from enigma.schemas.research import ResearchObjective

log = get_logger(__name__)
CIO = AgentRole.CIO


@dataclass(frozen=True)
class Schedule:
    poll_interval_seconds: float
    heartbeat_seconds: float
    auto_next_experiment: bool
    max_cycles_per_day: int
    max_concurrent_cycles: int
    tech_scan_interval_hours: float
    tech_scan_topics: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> Schedule:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            poll_interval_seconds=float(raw["poll_interval_seconds"]),
            heartbeat_seconds=float(raw["heartbeat_seconds"]),
            auto_next_experiment=bool(raw["auto_next_experiment"]),
            max_cycles_per_day=int(raw["max_cycles_per_day"]),
            max_concurrent_cycles=int(raw["max_concurrent_cycles"]),
            tech_scan_interval_hours=float(raw["tech_scan_interval_hours"]),
            tech_scan_topics=tuple(raw.get("tech_scan_topics", [])),
        )


def local_worker_alive(worker_id: str) -> bool:
    """Worker ids are `host:pid`. Workers on other hosts are presumed alive (leases expire)."""
    host, _, pid = worker_id.rpartition(":")
    if host != socket.gethostname() or not pid.isdigit():
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Daemon:
    def __init__(self, inst: Institution, schedule: Schedule) -> None:
        self.inst = inst
        self.schedule = schedule
        self.stop = threading.Event()
        self.worker_name = inst.worker.worker_id
        self._last_beat: datetime | None = None

    # ------------------------------------------------------------------ lifecycle
    def install_signal_handlers(self) -> None:
        def handler(signum: int, _frame: FrameType | None) -> None:
            log.info("shutdown_requested", signal=signum)
            self.stop.set()

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def startup(self) -> list[str]:
        requeued = self.inst.stores.queue.requeue_orphans(local_worker_alive)
        self.inst.stores.audit.record(
            "system", "daemon.started", self.worker_name, {"requeued_tasks": requeued}
        )
        self.heartbeat("starting", force=True)
        return requeued

    def heartbeat(self, status: str, force: bool = False) -> None:
        now = self.inst.clock.now()
        due = self._last_beat is None or (now - self._last_beat).total_seconds() >= (
            self.schedule.heartbeat_seconds
        )
        if force or due:
            self.inst.stores.flags.set(HEARTBEAT_PREFIX + self.worker_name, status, "system")
            self._last_beat = now

    def shutdown(self) -> None:
        self.heartbeat("stopped", force=True)
        self.inst.stores.audit.record("system", "daemon.stopped", self.worker_name, {})

    # ------------------------------------------------------------------ one iteration
    def step(self) -> bool:
        if self.inst.stores.flags.kill_switch_active():
            self.heartbeat("halted: kill switch")
            return False
        progressed = False
        for cycle_id in self.inst.driver.active_cycles():
            progressed |= self.inst.driver.tick(cycle_id)
        task = self.inst.worker.run_once()
        progressed |= task is not None
        progressed |= self._chain_next_experiments()
        progressed |= self._schedule_tech_scan()
        self.heartbeat(f"running: {task.kind}" if task else "idle")
        return progressed

    def run(self, until_idle: bool = False, max_iterations: int | None = None) -> int:
        self.startup()
        iterations = 0
        try:
            while not self.stop.is_set():
                progressed = self.step()
                iterations += 1
                if max_iterations is not None and iterations >= max_iterations:
                    break
                if not progressed:
                    if until_idle and not self.inst.driver.active_cycles():
                        break
                    self.stop.wait(self.schedule.poll_interval_seconds)
        finally:
            self.shutdown()
        return iterations

    # ------------------------------------------------------------------ schedules
    def _chain_next_experiments(self) -> bool:
        if not self.schedule.auto_next_experiment:
            return False
        if len(self.inst.driver.active_cycles()) >= self.schedule.max_concurrent_cycles:
            return False
        now = self.inst.clock.now()
        since = (now - timedelta(days=1)).isoformat()
        if self.inst.fsm.created_since(RESEARCH_CYCLE, since) >= self.schedule.max_cycles_per_day:
            return False
        docs = self.inst.stores.documents
        pending = docs.list(
            NextExperimentProposal, where={"state": ProposalState.PROPOSED.value}, limit=1
        )
        if not pending:
            return False
        proposal = pending[0]
        parent = self.inst.stores.checkpoints.load(proposal.parent_cycle_id)
        if parent is None:
            return False
        objective = docs.require(ResearchObjective, parent[1]["objective_id"])
        child = self.inst.driver.submit_objective(
            objective.statement,
            objective.instruments,
            objective.topics,
            max_trials=objective.max_trials,
            proposal_id=proposal.id,
            created_by=AgentRole.META_LEARNING,
        )
        docs.put(
            proposal.model_copy(update={"state": ProposalState.SCHEDULED, "child_cycle_id": child}),
            recorded_by=CIO.value,
        )
        self.inst.stores.audit.record(
            CIO.value,
            "cycle.chained",
            child,
            {"proposal_id": proposal.id, "parent_cycle_id": proposal.parent_cycle_id},
        )
        return True

    def _schedule_tech_scan(self) -> bool:
        if not self.schedule.tech_scan_topics or self.schedule.tech_scan_interval_hours <= 0:
            return False
        now = self.inst.clock.now()
        window = int(now.timestamp() // (self.schedule.tech_scan_interval_hours * 3600))
        request = TaskRequest(
            "tech.scan",
            AgentRole.TECHNOLOGY_INNOVATION,
            {"topics": list(self.schedule.tech_scan_topics), "max_sources": 10},
            idempotency_key=f"tech-scan-{window}",
        )
        before = self.inst.stores.queue.counts().get("pending", 0)
        delegate(
            self.inst.stores.queue,
            self.inst.bus,
            self.inst.driver.cmap,
            CIO,
            request,
            correlation_id=f"sched_tech_{window}",
        )
        return self.inst.stores.queue.counts().get("pending", 0) > before


def status_snapshot(inst: Institution) -> dict[str, Any]:
    rm = ReadModels(inst.stores)
    return {"health": rm.health(), "cycles": rm.cycles(), "costs": rm.costs()}
