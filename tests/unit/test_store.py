from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from enigma.core import IntegrityError, PermissionDeniedError, new_id
from enigma.schemas.common import AgentRole, Plane, SystemActor
from enigma.schemas.governance import CostRecord, Reflection, ReflectionOutcome, Task, TaskStatus
from enigma.schemas.knowledge import Lesson
from enigma.schemas.validation import RedactedValidationSummary, ValidationReport
from enigma.store import PlaneView, Stores
from enigma.store.db import MIGRATIONS

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _lesson(statement: str = "Funding carry fails after costs") -> Lesson:
    return Lesson(
        id="les_1",
        statement=statement,
        topics=("funding",),
        derived_from_ids=("x",),
        outcome="failure",
        claim_level="L2",
        created_by=AgentRole.META_LEARNING,
        created_at=NOW,
    )


def _task(**overrides: object) -> Task:
    base: dict[str, object] = dict(
        id=new_id("tsk"),
        kind="intelligence.research",
        assigned_to=AgentRole.INTELLIGENCE,
        created_by=AgentRole.CIO,
        correlation_id="cyc_1",
        created_at=NOW,
        updated_at=NOW,
    )
    base.update(overrides)
    return Task.model_validate(base)


def test_documents_are_versioned_not_overwritten(stores: Stores) -> None:
    docs = stores.documents
    assert docs.put(_lesson(), "test") == 1
    assert docs.put(_lesson("Revised statement"), "test") == 2
    assert docs.require(Lesson, "les_1").statement == "Revised statement"
    assert [x.statement for x in docs.history(Lesson, "les_1")] == [
        "Funding carry fails after costs",
        "Revised statement",
    ]
    assert docs.count(Lesson) == 1


def test_list_filters_by_json_field(stores: Stores) -> None:
    r = Reflection(
        id="ref_1",
        task_id="tsk_1",
        agent=AgentRole.QUANT_RESEARCH,
        outcome=ReflectionOutcome.FAILURE,
        what_worked="",
        what_failed="costs",
        created_at=NOW,
    )
    stores.documents.put(r, "test")
    assert len(stores.documents.list(Reflection, where={"outcome": "failure"})) == 1
    assert stores.documents.list(Reflection, where={"outcome": "success"}) == []
    with pytest.raises(ValueError):
        stores.documents.list(Reflection, where={"x') OR 1=1 --": 1})


def test_plane_view_blocks_research_from_validation_reports(stores: Stores) -> None:
    research = PlaneView(
        stores.documents,
        frozenset({Plane.RESEARCH, Plane.CONTROL}),
        frozenset({Plane.RESEARCH}),
        actor="quant_research",
    )
    with pytest.raises(PermissionDeniedError):
        research.list(ValidationReport)
    assert research.list(RedactedValidationSummary) == []
    with pytest.raises(PermissionDeniedError):
        research.put(_lesson())


def test_full_text_search_respects_planes(stores: Stores) -> None:
    lesson = _lesson()
    stores.documents.put(lesson, "test")
    stores.documents.index_text(
        lesson.id, "lesson", Plane.CONTROL, "Funding", lesson.statement, lesson.topics
    )
    assert stores.documents.search_text("funding costs", [Plane.CONTROL])[0][0] == "les_1"
    assert stores.documents.search_text("funding", [Plane.VALIDATION]) == []


def test_audit_chain_detects_tampering(stores: Stores) -> None:
    for i in range(5):
        stores.audit.record("cio", "test.action", f"s{i}", {"i": i})
    assert stores.audit.verify() == 5
    stores.db.execute("UPDATE audit_log SET details = '{\"i\":99}' WHERE seq = 3")
    with pytest.raises(IntegrityError):
        stores.audit.verify()


def test_queue_lease_expiry_allows_recovery(stores: Stores) -> None:
    q = stores.queue
    task = _task()
    q.enqueue(task)
    claimed = q.claim("worker-a", now=NOW)
    assert claimed is not None and claimed.status is TaskStatus.RUNNING
    assert q.claim("worker-b", now=NOW + timedelta(minutes=1)) is None
    reclaimed = q.claim("worker-b", now=NOW + timedelta(hours=1))
    assert reclaimed is not None and reclaimed.attempts == 2
    done = q.complete(task.id, {"ok": True})
    assert done.status is TaskStatus.SUCCEEDED and done.outputs == {"ok": True}


def test_queue_retry_then_terminal_failure(stores: Stores) -> None:
    q = stores.queue
    task = _task(max_attempts=2)
    q.enqueue(task)
    q.claim("w")
    assert q.fail(task.id, "boom").status is TaskStatus.PENDING
    q.claim("w")
    assert q.fail(task.id, "boom").status is TaskStatus.FAILED


def test_transient_failure_waits_before_the_next_claim(stores: Stores) -> None:
    q = stores.queue
    task = _task(max_attempts=3)
    q.enqueue(task)
    q.claim("w")
    pending = q.fail(task.id, "upstream timeout")
    assert pending.status is TaskStatus.PENDING
    assert pending.not_before is not None
    assert q.claim("w", now=pending.not_before - timedelta(seconds=1)) is None
    again = q.claim("w", now=pending.not_before + timedelta(seconds=1))
    assert again is not None and again.id == task.id


def test_queue_priority_and_role_filter(stores: Stores) -> None:
    q = stores.queue
    low = _task(priority=200)
    high = _task(priority=10)
    other = _task(assigned_to=AgentRole.DATA_ENGINEERING, priority=1)
    for t in (low, high, other):
        q.enqueue(t)
    got = q.claim("w", roles=(AgentRole.INTELLIGENCE,))
    assert got is not None and got.id == high.id


def test_cost_ledger_totals(stores: Stores) -> None:
    for usd in (0.5, 0.25):
        stores.costs.record(
            CostRecord(
                id=new_id("cst"),
                actor=AgentRole.INTELLIGENCE,
                task_id="t1",
                category="llm",
                provider="x",
                usd=usd,
                input_tokens=10,
                output_tokens=5,
                created_at=NOW,
            )
        )
    totals = stores.costs.totals(actor="intelligence")
    assert totals["usd"] == 0.75 and totals["tokens"] == 30


def test_checkpoints_and_flags(stores: Stores) -> None:
    stores.checkpoints.save("cyc_1", "research_cycle", "DATA_ACQUIRED", {"k": 1}, "running")
    assert stores.checkpoints.load("cyc_1") == ("DATA_ACQUIRED", {"k": 1}, "running")
    assert not stores.flags.kill_switch_active()
    stores.flags.set("kill_switch", "on", SystemActor.HUMAN.value)
    assert stores.flags.kill_switch_active()


def test_reopen_preserves_state(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    s1 = Stores.open(path)
    s1.documents.put(_lesson(), "test")
    s1.db.close()
    s2 = Stores.open(path)
    assert s2.documents.require(Lesson, "les_1").statement.startswith("Funding")
    assert s2.db.schema_version == len(MIGRATIONS)
