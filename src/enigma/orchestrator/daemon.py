"""Long-running autonomous loop: drive cycles, execute tasks, chain experiments, run schedules."""

from __future__ import annotations

import fcntl
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
from enigma.core.errors import IntegrityError
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
from enigma.schemas.market import MarketScanReport
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
    # continuous opens the next ranked backlog item with no daily count. 0 on
    # max_cycles_per_day means the same: no daily count. A positive count remains
    # a hard cap for tests and for a one-shot `enigma research` session.
    continuous: bool = False
    min_free_memory_mb: int = 512
    review_interval_hours: float = 24.0
    literature_interval_hours: float = 6.0
    consolidation_interval_hours: float = 12.0
    diminishing_returns_min_cycles: int = 3

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
            continuous=bool(raw.get("continuous", False)),
            min_free_memory_mb=int(raw.get("min_free_memory_mb", 512)),
            review_interval_hours=float(raw.get("review_interval_hours", 24)),
            literature_interval_hours=float(raw.get("literature_interval_hours", 6)),
            consolidation_interval_hours=float(raw.get("consolidation_interval_hours", 12)),
            diminishing_returns_min_cycles=int(raw.get("diminishing_returns_min_cycles", 3)),
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
        self._until_idle = False
        self._isolation_until: datetime | None = None
        self._isolation_block = ""
        self.preflight_note = ""

    # ------------------------------------------------------------------ lifecycle
    def install_signal_handlers(self) -> None:
        def handler(signum: int, _frame: FrameType | None) -> None:
            log.info("shutdown_requested", signal=signum)
            self.stop.set()

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def startup(self) -> list[str]:
        from enigma.orchestrator.quarantine import prepare_runtime  # noqa: PLC0415

        report = prepare_runtime(self.inst.settings, self.inst.stores)
        self.preflight_note = report.text().split("\n", 1)[0]
        if report.unresolved:
            self.preflight_note += f"; quarantined {len(report.unresolved)} artifact reference(s)"
        elif report.legacy:
            self.preflight_note += (
                f"; {report.legacy} absolute path(s) still resolve; run enigma home migrate"
            )
        requeued = self.inst.stores.queue.requeue_orphans(local_worker_alive)
        self.inst.stores.audit.record(
            "system",
            "daemon.started",
            self.worker_name,
            {"requeued_tasks": requeued, "unresolved_artifacts": len(report.unresolved)},
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
        if self.schedule.continuous and not self._until_idle:
            progressed |= self._continuous_start()
            progressed |= self._schedule_literature()
            progressed |= self._schedule_review()
        else:
            progressed |= self._chain_next_experiments()
        progressed |= self._schedule_tech_scan()
        progressed |= self._schedule_market_scan()
        # force: a pulse may have written moments ago, and that must not swallow
        # the status this step actually finished with.
        self.heartbeat(f"running: {task.kind}" if task else "idle", force=True)
        return progressed

    def run(
        self,
        until_idle: bool = False,
        max_iterations: int | None = None,
        on_step: Callable[[int, str], None] | None = None,
    ) -> int:
        lock = self._acquire_writer_lock()
        iterations = 0
        self._until_idle = until_idle
        try:
            requeued = self.startup()
            if on_step is not None:
                note = f"; {self.preflight_note}" if self.preflight_note else ""
                on_step(0, f"started; requeued {len(requeued)}{note}")
            while not self.stop.is_set():
                progressed = self.step()
                iterations += 1
                if on_step is not None:
                    active = ", ".join(self.inst.driver.active_cycles()) or "-"
                    on_step(iterations, f"active {active}")
                if max_iterations is not None and iterations >= max_iterations:
                    break
                if not progressed:
                    if until_idle and not self.inst.driver.active_cycles():
                        break
                    self.stop.wait(self.schedule.poll_interval_seconds)
        finally:
            self.shutdown()
            self._release_writer_lock(lock)
        return iterations

    def _acquire_writer_lock(self) -> Any:
        """One daemon per home. A second process must not tick the same cycles."""
        path = self.inst.settings.home / "control" / "daemon.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(
                f"another Enigma daemon already holds {path}; refusing a second writer"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{socket.gethostname()}:{os.getpid()}\n")
        handle.flush()
        return handle

    @staticmethod
    def _release_writer_lock(handle: Any) -> None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    # ------------------------------------------------------------------ schedules
    def _cycles_today(self) -> int:
        now = self.inst.clock.now()
        since = (now - timedelta(days=1)).isoformat()
        return self.inst.fsm.created_since(RESEARCH_CYCLE, since)

    def _slot_open(self) -> tuple[bool, str]:
        if len(self.inst.driver.active_cycles()) >= self.schedule.max_concurrent_cycles:
            return False, "a research cycle is already running"
        cap = self.schedule.max_cycles_per_day
        if cap > 0 and self._cycles_today() >= cap:
            return False, f"daily cycle cap {cap} reached"
        if self.schedule.continuous and not self._until_idle:
            reason = self._resource_block()
            if reason:
                return False, reason
        return True, ""

    def _resource_block(self) -> str:
        from enigma.orchestrator.host import available_memory_mb  # noqa: PLC0415
        from enigma.strategy.sandbox import (  # noqa: PLC0415
            SandboxRunner,
            SandboxUnavailableError,
        )

        free = available_memory_mb()
        floor = self.schedule.min_free_memory_mb
        if free is not None and free < floor:
            return f"memory pressure: {free:.0f} MiB available, floor {floor} MiB"
        now = self.inst.clock.now()
        if self._isolation_until is not None and now < self._isolation_until:
            return self._isolation_block
        try:
            SandboxRunner(allow_weak=False).check_isolation("continuous")
        except SandboxUnavailableError as exc:
            self._isolation_block = str(exc)[:400]
            self._isolation_until = now + timedelta(seconds=15)
            return self._isolation_block
        self._isolation_block = ""
        self._isolation_until = now + timedelta(seconds=60)
        return ""

    def _remaining_slots(self) -> float:
        cap = self.schedule.max_cycles_per_day
        if cap <= 0:
            return 1_000_000.0
        return float(max(cap - self._cycles_today(), 0))

    def _chain_next_experiments(self) -> bool:
        if not self.schedule.auto_next_experiment:
            return False
        open_slot, _reason = self._slot_open()
        if not open_slot:
            return False
        docs = self.inst.stores.documents
        candidates = proposal_candidates(self.inst.stores)
        if not candidates:
            return False
        inputs = build_inputs(
            read_record(self.inst.stores),
            remaining_budget=self._remaining_slots(),
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

    def _continuous_start(self) -> bool:
        """Open the highest-value backlog item. No daily count, no pause between cycles."""
        from enigma.orchestrator.backlog import BacklogPolicy, research_agenda  # noqa: PLC0415

        open_slot, reason = self._slot_open()
        policy = BacklogPolicy(
            diminishing_returns_min_cycles=self.schedule.diminishing_returns_min_cycles
        )
        agenda = research_agenda(
            self.inst.stores,
            inputs_budget=self._remaining_slots(),
            policy=policy,
        )
        viable = tuple(item for item in agenda.ranked if item.viable) if open_slot else ()
        from enigma.orchestrator.failures import block_reason, blocked_ids  # noqa: PLC0415

        now = self.inst.clock.now()
        blocked = blocked_ids(self.inst.stores, now) if open_slot else set()
        skipped = tuple(item for item in viable if item.candidate.id in blocked)
        viable = tuple(item for item in viable if item.candidate.id not in blocked)
        opened: tuple[ScoredCandidate, ...] = ()
        pause = reason if not open_slot else ""
        if open_slot and skipped and not viable:
            pause = block_reason(self.inst.stores, skipped[0].candidate.id)
        for item in viable:
            try:
                if self._open_ranked(item.candidate):
                    opened = (item,)
                    break
            except IntegrityError as exc:
                self._quarantine(item.candidate, exc)
                pause = f"quarantined {item.candidate.id}"
        self._record_cio_decision(agenda, opened)
        self._publish_operations(agenda, "" if opened else pause)
        return bool(opened)

    def _open_ranked(self, candidate: Candidate) -> bool:
        if candidate.kind == "next_experiment" and candidate.proposal_id:
            return self._open_proposal(candidate.proposal_id)
        if candidate.kind == "market_scan":
            try:
                self.inst.driver.open_objective(candidate.id)
            except (KeyError, ValueError):
                return False
            return True
        if candidate.kind == "opportunity" and len(candidate.refs) == 3:
            return self._open_opportunity(candidate.refs)
        return False

    def _quarantine(self, candidate: Candidate, exc: IntegrityError) -> None:
        from enigma.orchestrator.quarantine import (  # noqa: PLC0415
            candidate_symbols,
            quarantine_candidate,
        )

        symbols = candidate_symbols(self.inst.stores, candidate)
        quarantine_candidate(self.inst.stores, candidate, str(exc), symbols)

    def _open_proposal(self, proposal_id: str) -> bool:
        docs = self.inst.stores.documents
        proposal = docs.require(NextExperimentProposal, proposal_id)
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

    def _open_opportunity(self, refs: tuple[str, ...]) -> bool:
        from enigma.orchestrator.institution import perp  # noqa: PLC0415
        from enigma.research.mechanisms import MECHANISMS  # noqa: PLC0415

        mechanism_id, symbol, horizon = refs
        mechanism = MECHANISMS[mechanism_id]
        statement = (
            f"Does {mechanism.name} show a gross edge on {symbol} perpetuals at a "
            f"{horizon} horizon before trading costs? Direction {mechanism_id}|{symbol}|{horizon}."
        )
        try:
            cycle_id = self.inst.driver.submit_objective(
                statement,
                (perp(symbol),),
                (mechanism_id, symbol, horizon, "continuous-backlog"),
                max_trials=9,
                created_by=CIO,
            )
        except ValueError:
            return False
        self.inst.stores.audit.record(
            CIO.value,
            "cycle.backlog_opened",
            cycle_id,
            {"mechanism": mechanism_id, "symbol": symbol, "horizon": horizon},
        )
        return True

    def _publish_operations(self, agenda: Agenda, pause_reason: str) -> None:
        import json  # noqa: PLC0415

        from enigma.orchestrator.host import available_memory_mb  # noqa: PLC0415

        free = available_memory_mb()
        payload = {
            "mode": "continuous",
            "max_cycles_per_day": self.schedule.max_cycles_per_day,
            "max_concurrent_cycles": self.schedule.max_concurrent_cycles,
            "active_cycles": self.inst.driver.active_cycles(),
            "pause_reason": pause_reason,
            "memory_available_mb": None if free is None else round(free, 1),
            "memory_floor_mb": self.schedule.min_free_memory_mb,
            "backlog": [
                {
                    "id": item.candidate.id,
                    "kind": item.candidate.kind,
                    "summary": item.candidate.summary[:240],
                    "score": item.score,
                    "viable": item.viable,
                    "reasons": list(item.reasons),
                }
                for item in agenda.ranked
                if item.viable or item.reasons
            ][:24],
        }
        self.inst.stores.flags.set("operations.snapshot", json.dumps(payload), "system")

    def _schedule_literature(self) -> bool:
        if self.inst.driver.active_cycles() or self.schedule.literature_interval_hours <= 0:
            return False
        if not self.schedule.tech_scan_topics and not self.schedule.continuous:
            return False
        now = self.inst.clock.now()
        window = int(now.timestamp() // (self.schedule.literature_interval_hours * 3600))
        topics = list(self.schedule.tech_scan_topics) or [
            "crypto perpetual funding rate",
            "basis mean reversion",
            "market microstructure research",
        ]
        request = TaskRequest(
            "tech.scan",
            AgentRole.TECHNOLOGY_INNOVATION,
            {"topics": topics, "max_sources": 6},
            idempotency_key=f"literature-{window}",
        )
        before = self.inst.stores.queue.counts().get("pending", 0)
        delegate(
            self.inst.stores.queue,
            self.inst.bus,
            self.inst.driver.cmap,
            CIO,
            request,
            correlation_id=f"sched_literature_{window}",
        )
        return self.inst.stores.queue.counts().get("pending", 0) > before

    def _schedule_review(self) -> bool:
        from enigma.fsm import IMPROVEMENT_PROPOSAL  # noqa: PLC0415
        from enigma.learning.review import review_institution  # noqa: PLC0415
        from enigma.schemas.governance import ImprovementProposal, Reflection  # noqa: PLC0415
        from enigma.schemas.knowledge import Lesson  # noqa: PLC0415

        now = self.inst.clock.now()
        changed = False
        if self.schedule.review_interval_hours > 0:
            window = int(now.timestamp() // (self.schedule.review_interval_hours * 3600))
            flag = f"institution-review:{window}"
            if self.inst.stores.flags.get(flag) is None:
                proposals = review_institution(
                    self.inst.stores.documents.list(Reflection, limit=2000),
                    self.inst.stores.documents.list(Lesson, limit=2000),
                    (),
                    now,
                    window_key=f"review-{window}",
                )
                for proposal in proposals:
                    if self.inst.stores.documents.get(ImprovementProposal, proposal.id) is None:
                        self.inst.stores.documents.put(proposal, recorded_by="meta_learning")
                    if self.inst.fsm.state(IMPROVEMENT_PROPOSAL, proposal.id) is None:
                        self.inst.fsm.create(
                            IMPROVEMENT_PROPOSAL,
                            proposal.id,
                            "meta_learning",
                            reason_code="proposed",
                        )
                self.inst.stores.flags.set(flag, str(len(proposals)), "system")
                changed = True
        if self.schedule.consolidation_interval_hours > 0:
            window = int(now.timestamp() // (self.schedule.consolidation_interval_hours * 3600))
            flag = f"knowledge-consolidation:{window}"
            if self.inst.stores.flags.get(flag) is None:
                lessons = self.inst.stores.documents.list(Lesson, limit=2000)
                self.inst.stores.documents.put(
                    AgentDecision(
                        id="dec_cons_" + sha256_hex(flag)[:22],
                        agent=CIO,
                        task_id=None,
                        decision_type="knowledge_consolidation",
                        outcome=f"{len(lessons)} lessons",
                        reason_code="scheduled_review",
                        reasoning_summary=(
                            f"Consolidated {len(lessons)} stored lessons. No lesson was deleted. "
                            "Directions with a completed validation stay closed unless a lesson "
                            "records a positive gross edge destroyed by costs."
                        ),
                        input_refs=tuple(les.id for les in lessons[:30]),
                        output_refs=(),
                        instructions_version="consolidation-1",
                        created_at=now,
                    ),
                    recorded_by=CIO.value,
                )
                self.inst.stores.flags.set(flag, "done", "system")
                changed = True
        return changed

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
        from enigma.core.paths import StorageRoots  # noqa: PLC0415

        created = run_market_scan(
            self.inst.stores,
            self.inst.clock.now(),
            interval_hours=self.schedule.market_scan_interval_hours,
            roots=StorageRoots.from_settings(self.inst.settings),
        )
        opened = self._open_scan_objective()
        return created or opened

    def _open_scan_objective(self) -> bool:
        if self.schedule.continuous and not self._until_idle:
            return False
        if len(self.inst.driver.active_cycles()) >= self.schedule.max_concurrent_cycles:
            return False
        cap = self.schedule.max_cycles_per_day
        if cap > 0 and self._cycles_today() >= cap:
            return False
        reports = self.inst.stores.documents.list(MarketScanReport, limit=1)
        if not reports or not reports[0].objective_id:
            return False
        objective_id = reports[0].objective_id
        for row in self.inst.stores.checkpoints.list("research_cycle"):
            if row["state"].get("objective_id") == objective_id:
                return False
        try:
            self.inst.driver.open_objective(objective_id)
        except (KeyError, ValueError):
            return False
        return True


def status_snapshot(inst: Institution) -> dict[str, Any]:
    rm = ReadModels(inst.stores)
    return {"health": rm.health(), "cycles": rm.cycles(), "costs": rm.costs()}
