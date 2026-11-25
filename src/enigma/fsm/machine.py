"""Explicit, actor-restricted state machines persisted as append-only transition events."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from enigma.core.errors import InvalidTransitionError, PermissionDeniedError
from enigma.core.hashing import canonical_json
from enigma.store.audit import AuditLog
from enigma.store.db import Database


@dataclass(frozen=True)
class Transition:
    source: str
    target: str
    actors: frozenset[str]
    requires_artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class MachineDefinition:
    name: str
    initial: str
    terminal: frozenset[str]
    transitions: tuple[Transition, ...]
    # Transitions allowed from any non-terminal state (e.g. risk halt).
    global_transitions: tuple[Transition, ...] = ()
    states: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        states = {self.initial, *self.terminal}
        for t in (*self.transitions, *self.global_transitions):
            states.update({t.source, t.target} - {"*"})
        object.__setattr__(self, "states", frozenset(states))
        seen: set[tuple[str, str]] = set()
        for t in self.transitions:
            if (t.source, t.target) in seen:
                raise ValueError(f"duplicate transition {t.source}->{t.target} in {self.name}")
            seen.add((t.source, t.target))

    def find(self, source: str, target: str) -> Transition | None:
        for t in self.transitions:
            if t.source == source and t.target == target:
                return t
        if source not in self.terminal:
            for t in self.global_transitions:
                if t.target == target:
                    return t
        return None

    def allowed_targets(self, source: str) -> list[str]:
        targets = [t.target for t in self.transitions if t.source == source]
        if source not in self.terminal:
            targets.extend(t.target for t in self.global_transitions)
        return sorted(set(targets))


@dataclass(frozen=True)
class TransitionEvent:
    seq: int
    machine: str
    subject_id: str
    from_state: str | None
    to_state: str
    actor: str
    reason_code: str
    artifact_hashes: dict[str, str]
    ts: str


class StateMachineStore:
    """Applies transitions for any registered machine; current state is the last event."""

    def __init__(self, db: Database, audit: AuditLog) -> None:
        self.db = db
        self.audit = audit

    def state(self, machine: MachineDefinition, subject_id: str) -> str | None:
        row = self.db.query_one(
            "SELECT to_state FROM fsm_events WHERE machine = ? AND subject_id = ?"
            " ORDER BY seq DESC LIMIT 1",
            (machine.name, subject_id),
        )
        return None if row is None else str(row["to_state"])

    def create(
        self, machine: MachineDefinition, subject_id: str, actor: str, reason_code: str = "created"
    ) -> TransitionEvent:
        if self.state(machine, subject_id) is not None:
            raise InvalidTransitionError(f"{machine.name}:{subject_id} already exists")
        return self._append(machine, subject_id, None, machine.initial, actor, reason_code, {})

    def transition(
        self,
        machine: MachineDefinition,
        subject_id: str,
        target: str,
        actor: str,
        reason_code: str,
        artifact_hashes: Mapping[str, str] | None = None,
    ) -> TransitionEvent:
        artifacts = dict(artifact_hashes or {})
        try:
            return self._checked_transition(
                machine, subject_id, target, actor, reason_code, artifacts
            )
        except PermissionDeniedError:
            self.audit.record(
                actor, "fsm.transition_denied", subject_id, {"machine": machine.name, "to": target}
            )
            raise

    def _checked_transition(
        self,
        machine: MachineDefinition,
        subject_id: str,
        target: str,
        actor: str,
        reason_code: str,
        artifacts: dict[str, str],
    ) -> TransitionEvent:
        with self.db.transaction():
            current = self.state(machine, subject_id)
            if current is None:
                raise InvalidTransitionError(f"{machine.name}:{subject_id} does not exist")
            rule = machine.find(current, target)
            if rule is None:
                raise InvalidTransitionError(
                    f"{machine.name}:{subject_id} cannot go {current} -> {target}; allowed:"
                    f" {machine.allowed_targets(current)}"
                )
            if actor not in rule.actors:
                raise PermissionDeniedError(
                    f"{actor} may not move {machine.name} {current} -> {target}"
                )
            missing = [a for a in rule.requires_artifacts if a not in artifacts]
            if missing:
                raise InvalidTransitionError(
                    f"{machine.name} {current} -> {target} requires artifacts {missing}"
                )
            return self._append(machine, subject_id, current, target, actor, reason_code, artifacts)

    def _append(
        self,
        machine: MachineDefinition,
        subject_id: str,
        source: str | None,
        target: str,
        actor: str,
        reason_code: str,
        artifacts: dict[str, str],
    ) -> TransitionEvent:
        ts = datetime.now(UTC).isoformat()
        with self.db.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO fsm_events(machine, subject_id, from_state, to_state, actor,"
                " reason_code, artifact_hashes, ts) VALUES (?,?,?,?,?,?,?,?)",
                (
                    machine.name,
                    subject_id,
                    source,
                    target,
                    actor,
                    reason_code,
                    canonical_json(artifacts),
                    ts,
                ),
            )
            self.audit.record(
                actor,
                "fsm.transition",
                subject_id,
                {
                    "machine": machine.name,
                    "from": source,
                    "to": target,
                    "reason_code": reason_code,
                    "artifacts": artifacts,
                },
            )
        return TransitionEvent(
            int(cur.lastrowid or 0),
            machine.name,
            subject_id,
            source,
            target,
            actor,
            reason_code,
            artifacts,
            ts,
        )

    def history(self, machine: MachineDefinition, subject_id: str) -> list[TransitionEvent]:
        rows = self.db.query(
            "SELECT * FROM fsm_events WHERE machine = ? AND subject_id = ? ORDER BY seq",
            (machine.name, subject_id),
        )
        return [
            TransitionEvent(
                int(r["seq"]),
                r["machine"],
                r["subject_id"],
                r["from_state"],
                r["to_state"],
                r["actor"],
                r["reason_code"],
                json.loads(r["artifact_hashes"]),
                r["ts"],
            )
            for r in rows
        ]

    def subjects_in(self, machine: MachineDefinition, state: str) -> list[str]:
        rows = self.db.query(
            "SELECT e.subject_id FROM fsm_events e JOIN (SELECT subject_id, MAX(seq) s FROM"
            " fsm_events WHERE machine = ? GROUP BY subject_id) m ON e.seq = m.s"
            " WHERE e.to_state = ?",
            (machine.name, state),
        )
        return [str(r["subject_id"]) for r in rows]

    def state_counts(self, machine: MachineDefinition) -> dict[str, int]:
        rows = self.db.query(
            "SELECT e.to_state, COUNT(*) n FROM fsm_events e JOIN (SELECT subject_id, MAX(seq) s"
            " FROM fsm_events WHERE machine = ? GROUP BY subject_id) m ON e.seq = m.s"
            " GROUP BY e.to_state",
            (machine.name,),
        )
        return {str(r["to_state"]): int(r["n"]) for r in rows}
