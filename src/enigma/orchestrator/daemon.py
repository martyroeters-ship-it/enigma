"""Long-running autonomous loop: drive cycles, execute tasks, chain experiments, run schedules."""

from __future__ import annotations

import os
import signal
import socket
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import FrameType
from typing import Any

import yaml

from enigma.agents.base import TaskRequest, delegate
from enigma.api.readmodels import HEARTBEAT_PREFIX, ReadModels
from enigma.core.hashing import sha256_hex
from enigma.core.logging import get_logger
from enigma.fsm import RESEARCH_CYCLE
from enigma.orchestrator.cio import (
    Agenda,
    Candidate,
    ResearchPrioritizer,
    ScoredCandidate,
    allocate,
)
from enigma.orchestrator.institution import Institution
from enigma.orchestrator.market_scan import run_market_scan
from enigma.orchestrator.planning import build_inputs, proposal_candidates, read_record
from enigma.schemas.common import AgentRole
from enigma.schemas.cycle import NextExperimentProposal, ProposalState
from enigma.schemas.governance import AgentDecision
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
    market_scan_interval_hours: float = 0.0

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
            market_scan_interval_hours=float(raw.get("market_scan_interval_hours", 0)),
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
        self.guidance_penalty: Callable[[Candidate], tuple[float, tuple[str, ...]]] | None = None

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

    @contextmanager
    def _heartbeat_pulse(self) -> Iterator[None]:
        """Keep the worker flag fresh while ``run_once`` is blocked in a model call.

        The dashboard marks a heartbeat older than one minute as stale. A hypothesis
        call can occupy this step for many minutes, and the write at the end of
        ``step`` does not run until the call returns. The pulse uses
        ``heartbeat_seconds`` (15 in the production schedule). An interval of 0
        leaves the pulse off so fast tests stay single-threaded. The same pulse
        renews the task lease: the lease is 30 minutes and one call can run longer
        than that while this process is still alive.
        """
        interval = self.schedule.heartbeat_seconds
        if interval <= 0:
            yield
            return
        stop = threading.Event()

        def pulse() -> None:
            while not stop.is_set():
                self.heartbeat(self._inflight_status(), force=True)
                if stop.wait(interval):
                    return

        thread = threading.Thread(target=pulse, name="enigma-heartbeat", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=5.0)

    def _inflight_status(self) -> str:
        rows = self.inst.stores.db.query(
            "SELECT id, kind FROM task_queue WHERE status = 'running' AND lease_owner = ?",
            (self.worker_name,),
        )
        lease = self.inst.worker.lease
        for row in rows:
            self.inst.stores.queue.heartbeat(str(row["id"]), self.worker_name, lease=lease)
        if not rows:
            return "running"
        return f"running: {rows[0]['kind']}"

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
        with self._heartbeat_pulse():
            task = self.inst.worker.run_once()
        progressed |= task is not None
        progressed |= self._chain_next_experiments()
        progressed |= self._schedule_tech_scan()
        progressed |= self._schedule_market_scan()
        # force: a pulse may have written moments ago, and that must not swallow
        # the status this step actually finished with.
        self.heartbeat(f"running: {task.kind}" if task else "idle", force=True)
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
        created_today = self.inst.fsm.created_since(RESEARCH_CYCLE, since)
        candidates = proposal_candidates(self.inst.stores)
        if not candidates:
            return False
        inputs = build_inputs(
            read_record(self.inst.stores),
            remaining_budget=float(self.schedule.max_cycles_per_day - created_today),
            guidance_penalty=self.guidance_penalty,
        )
        agenda = ResearchPrioritizer().rank(candidates, inputs)
        chosen = allocate(agenda, capacity=1, technology_share=0.0)
        self._record_cio_decision(agenda, chosen)
        if not chosen or chosen[0].candidate.proposal_id is None:
            return False
        proposal = docs.require(NextExperimentProposal, chosen[0].candidate.proposal_id)
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

    def _record_cio_decision(self, agenda: Agenda, chosen: tuple[ScoredCandidate, ...]) -> None:
        lines = [
            f"{s.candidate.id} score={s.score:.3f} viable={s.viable} "
            + " ".join(f"{k}={v:.2f}" for k, v in sorted(s.components.items()))
            + (f" ({'; '.join(s.reasons)})" if s.reasons else "")
            for s in agenda.ranked[:5]
        ]
        outcome = chosen[0].candidate.id if chosen else "none"
        key = "|".join([outcome, *(s.candidate.id for s in agenda.ranked)])
        self.inst.stores.documents.put(
            AgentDecision(
                id="dec_cio_" + sha256_hex(key + self.inst.clock.now().date().isoformat())[:22],
                agent=CIO,
                task_id=None,
                decision_type="research_prioritization",
                outcome=outcome,
                reason_code="max_expected_value",
                reasoning_summary=(
                    f"Ranked {len(agenda.ranked)} candidate(s) by information gain, novelty, "
                    "data readiness, agent reliability, cost and learned guidance:\n"
                    + "\n".join(lines)
                ),
                input_refs=tuple(s.candidate.id for s in agenda.ranked),
                output_refs=(outcome,) if chosen else (),
                instructions_version="cio-prioritizer-1",
                created_at=self.inst.clock.now(),
            ),
            recorded_by=CIO.value,
        )

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

    def _schedule_market_scan(self) -> bool:
        if self.schedule.market_scan_interval_hours <= 0:
            return False
        return run_market_scan(
            self.inst.stores,
            self.inst.clock.now(),
            interval_hours=self.schedule.market_scan_interval_hours,
        )


def status_snapshot(inst: Institution) -> dict[str, Any]:
    rm = ReadModels(inst.stores)
    return {"health": rm.health(), "cycles": rm.cycles(), "costs": rm.costs()}
