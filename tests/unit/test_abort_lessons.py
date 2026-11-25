from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from cycle_rig import Rig

from enigma.learning.abort import AbortClass, diagnose_abort, lessons_from_aborted_cycle
from enigma.schemas.common import ClaimLevel
from enigma.schemas.knowledge import Lesson

CASES = [
    (
        "strategy task failed: InvalidTransitionError: strategy_lifecycle:pkg_B0A888 "
        "cannot go RESEARCH_REJECTED -> DRAFT_HYPOTHESIS; allowed: []",
        "DATA_ACQUIRED",
        AbortClass.POLICY,
        "lifecycle_collision",
    ),
    (
        "no strategy trial ran successfully: AttributeError: 'dict' object has no attribute 'name'",
        "STRATEGY_EXECUTED",
        AbortClass.STRATEGY_RUNTIME,
        "strategy_runtime_failure",
    ),
    (
        "twin backtests were not reproducible",
        "STRATEGY_EXECUTED",
        AbortClass.STRATEGY_RUNTIME,
        "non_reproducible_backtest",
    ),
    (
        "IntegrityError: holdout rows changed since manifest hmf_da32136a was sealed",
        "OBJECTIVE_RECEIVED",
        AbortClass.INFRASTRUCTURE,
        "holdout_integrity",
    ),
]


def test_abort_classifiers_cover_m1_failures() -> None:
    for reason, step, expected_class, expected_root in CASES:
        diagnosis = diagnose_abort(reason, step)
        assert diagnosis.abort_class is expected_class
        assert diagnosis.root_cause == expected_root
        assert diagnosis.prevention
        lessons = lessons_from_aborted_cycle("cyc_test", reason, step=step, now=datetime.now(UTC))
        assert len(lessons) == 1
        les = lessons[0]
        assert les.outcome == "aborted"
        assert les.claim_level is ClaimLevel.L1_HYPOTHESIS
        assert expected_class.value in les.topics
        assert expected_root in les.topics
        assert "Prevention:" in les.statement


def test_driver_abort_persists_a_lesson(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    cycle = rig.submit()
    # Advance far enough that a strategy abort is realistic, then force an abort.
    assert rig.run(cycle, until="HYPOTHESIS_GENERATED") in {
        "HYPOTHESIS_GENERATED",
        "DATA_ACQUIRED",
        "COMPLETED",
    }
    rig.driver.abort(cycle, "twin backtests were not reproducible")
    state = rig.stores.checkpoints.load(cycle)[1]
    assert state["abort_reason"].startswith("twin")
    lesson_ids = state["abort_lesson_ids"]
    assert lesson_ids
    les = rig.stores.documents.require(Lesson, lesson_ids[0])
    assert les.outcome == "aborted"
    assert "non_reproducible_backtest" in les.topics
