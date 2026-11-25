from __future__ import annotations

import pytest

from enigma.core import InvalidTransitionError, PermissionDeniedError
from enigma.fsm import (
    RESEARCH_CYCLE,
    RESEARCH_CYCLE_STEPS,
    STRATEGY_LIFECYCLE,
    StateMachineStore,
)
from enigma.store import Stores


@pytest.fixture
def fsm(stores: Stores) -> StateMachineStore:
    return StateMachineStore(stores.db, stores.audit)


def _to_validation_running(fsm: StateMachineStore, sid: str) -> None:
    fsm.create(STRATEGY_LIFECYCLE, sid, "quant_research")
    fsm.transition(STRATEGY_LIFECYCLE, sid, "RESEARCH_EXPERIMENT_RUNNING", "quant_research", "go")
    fsm.transition(
        STRATEGY_LIFECYCLE,
        sid,
        "RESEARCH_CANDIDATE",
        "strategy_engineering",
        "ok",
        {"experiment_result": "sha256:x"},
    )
    fsm.transition(
        STRATEGY_LIFECYCLE,
        sid,
        "VALIDATION_QUEUED",
        "cio",
        "submit",
        {"strategy_package": "sha256:y"},
    )
    fsm.transition(STRATEGY_LIFECYCLE, sid, "VALIDATION_RUNNING", "validation_service", "start")


def test_only_validation_service_decides_validation(fsm: StateMachineStore) -> None:
    _to_validation_running(fsm, "pkg_1")
    for actor in ("cio", "quant_research", "independent_validation", "human"):
        with pytest.raises(PermissionDeniedError):
            fsm.transition(
                STRATEGY_LIFECYCLE,
                "pkg_1",
                "VALIDATION_PASSED",
                actor,
                "x",
                {"validation_report": "sha256:z"},
            )
    fsm.transition(
        STRATEGY_LIFECYCLE,
        "pkg_1",
        "VALIDATION_FAILED",
        "validation_service",
        "gates_failed",
        {"validation_report": "sha256:z"},
    )
    assert fsm.state(STRATEGY_LIFECYCLE, "pkg_1") == "VALIDATION_FAILED"


def test_failed_validation_is_terminal(fsm: StateMachineStore) -> None:
    _to_validation_running(fsm, "pkg_2")
    fsm.transition(
        STRATEGY_LIFECYCLE,
        "pkg_2",
        "VALIDATION_FAILED",
        "validation_service",
        "f",
        {"validation_report": "sha256:z"},
    )
    for target in (
        "VALIDATION_PASSED",
        "PAPER_APPROVAL_PENDING",
        "PAPER_RUNNING",
        "VALIDATION_QUEUED",
        "PAPER_STOPPED",
    ):
        with pytest.raises(InvalidTransitionError):
            fsm.transition(STRATEGY_LIFECYCLE, "pkg_2", target, "human", "relabel")


def test_transition_requires_artifacts(fsm: StateMachineStore) -> None:
    _to_validation_running(fsm, "pkg_3")
    with pytest.raises(InvalidTransitionError):
        fsm.transition(STRATEGY_LIFECYCLE, "pkg_3", "VALIDATION_PASSED", "validation_service", "x")


def test_paper_requires_human_and_risk_clearance(fsm: StateMachineStore) -> None:
    _to_validation_running(fsm, "pkg_4")
    fsm.transition(
        STRATEGY_LIFECYCLE,
        "pkg_4",
        "VALIDATION_PASSED",
        "validation_service",
        "p",
        {"validation_report": "sha256:z"},
    )
    fsm.transition(STRATEGY_LIFECYCLE, "pkg_4", "PAPER_APPROVAL_PENDING", "cio", "request")
    with pytest.raises(PermissionDeniedError):
        fsm.transition(
            STRATEGY_LIFECYCLE, "pkg_4", "PAPER_RUNNING", "cio", "x", {"risk_clearance": "c"}
        )
    with pytest.raises(InvalidTransitionError):
        fsm.transition(STRATEGY_LIFECYCLE, "pkg_4", "PAPER_RUNNING", "human", "x")
    fsm.transition(
        STRATEGY_LIFECYCLE, "pkg_4", "PAPER_RUNNING", "human", "ok", {"risk_clearance": "c"}
    )


def test_risk_can_halt_from_any_live_state(fsm: StateMachineStore) -> None:
    _to_validation_running(fsm, "pkg_5")
    fsm.transition(STRATEGY_LIFECYCLE, "pkg_5", "PAPER_STOPPED", "risk_management", "halt")
    assert fsm.state(STRATEGY_LIFECYCLE, "pkg_5") == "PAPER_STOPPED"


def test_denied_transition_is_audited(fsm: StateMachineStore, stores: Stores) -> None:
    fsm.create(STRATEGY_LIFECYCLE, "pkg_6", "quant_research")
    with pytest.raises(PermissionDeniedError):
        fsm.transition(
            STRATEGY_LIFECYCLE, "pkg_6", "RESEARCH_EXPERIMENT_RUNNING", "risk_management", "x"
        )
    actions = [e.action for e in stores.audit.entries(subject_id="pkg_6")]
    assert "fsm.transition_denied" in actions
    stores.audit.verify()


def test_research_cycle_walks_all_steps(fsm: StateMachineStore) -> None:
    fsm.create(RESEARCH_CYCLE, "cyc_1", "cio")
    for step in (*RESEARCH_CYCLE_STEPS[1:], "COMPLETED"):
        fsm.transition(RESEARCH_CYCLE, "cyc_1", step, "cio", "step_done")
    history = fsm.history(RESEARCH_CYCLE, "cyc_1")
    assert [e.to_state for e in history] == [*RESEARCH_CYCLE_STEPS, "COMPLETED"]
    with pytest.raises(InvalidTransitionError):
        fsm.transition(RESEARCH_CYCLE, "cyc_1", "ABORTED", "cio", "late")
    assert fsm.state_counts(RESEARCH_CYCLE) == {"COMPLETED": 1}


def test_cannot_skip_steps(fsm: StateMachineStore) -> None:
    fsm.create(RESEARCH_CYCLE, "cyc_2", "cio")
    with pytest.raises(InvalidTransitionError):
        fsm.transition(RESEARCH_CYCLE, "cyc_2", "VALIDATION_DECIDED", "cio", "skip")
