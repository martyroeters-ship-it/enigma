"""Institutional state machines (ARCHITECTURE.md §4, LEARNING_ARCHITECTURE.md §5)."""

from __future__ import annotations

from itertools import pairwise

from enigma.fsm.machine import MachineDefinition, Transition
from enigma.schemas.common import AgentRole, SystemActor

_CIO = AgentRole.CIO.value
_QR = AgentRole.QUANT_RESEARCH.value
_SE = AgentRole.STRATEGY_ENGINEERING.value
_VAL_SVC = SystemActor.VALIDATION_SERVICE.value
_RISK = AgentRole.RISK_MANAGEMENT.value
_RISK_SVC = SystemActor.RISK_SERVICE.value
_HUMAN = SystemActor.HUMAN.value
_META = AgentRole.META_LEARNING.value
_TECH = AgentRole.TECHNOLOGY_INNOVATION.value
_EXEC = AgentRole.EXECUTION_MONITORING.value


def _t(source: str, target: str, *actors: str, requires: tuple[str, ...] = ()) -> Transition:
    return Transition(source, target, frozenset(actors), requires)


# Strategy lifecycle. VALIDATION_FAILED is terminal: a failed package can never be relabeled;
# improvements must arrive as a new package with a new id.
STRATEGY_LIFECYCLE = MachineDefinition(
    name="strategy_lifecycle",
    initial="DRAFT_HYPOTHESIS",
    terminal=frozenset({"VALIDATION_FAILED", "RESEARCH_REJECTED", "ARCHIVED"}),
    transitions=(
        _t("DRAFT_HYPOTHESIS", "RESEARCH_EXPERIMENT_RUNNING", _CIO, _QR, _SE),
        _t(
            "RESEARCH_EXPERIMENT_RUNNING",
            "RESEARCH_CANDIDATE",
            _QR,
            _SE,
            requires=("experiment_result",),
        ),
        _t("RESEARCH_EXPERIMENT_RUNNING", "RESEARCH_REJECTED", _QR, _SE, _CIO),
        _t("RESEARCH_CANDIDATE", "VALIDATION_QUEUED", _CIO, _QR, requires=("strategy_package",)),
        _t("VALIDATION_QUEUED", "VALIDATION_RUNNING", _VAL_SVC),
        _t("VALIDATION_RUNNING", "VALIDATION_PASSED", _VAL_SVC, requires=("validation_report",)),
        _t("VALIDATION_RUNNING", "VALIDATION_FAILED", _VAL_SVC, requires=("validation_report",)),
        _t("VALIDATION_PASSED", "PAPER_APPROVAL_PENDING", _CIO),
        _t("PAPER_APPROVAL_PENDING", "PAPER_RUNNING", _HUMAN, requires=("risk_clearance",)),
        _t("PAPER_RUNNING", "PAPER_DEGRADED", _EXEC, _RISK, _RISK_SVC),
        _t("PAPER_DEGRADED", "PAPER_RUNNING", _HUMAN, requires=("risk_clearance",)),
        _t("PAPER_RUNNING", "ARCHIVED", _CIO, _HUMAN),
        _t("PAPER_STOPPED", "ARCHIVED", _CIO, _HUMAN),
        _t("VALIDATION_PASSED", "ARCHIVED", _CIO, _HUMAN),
        _t("RESEARCH_CANDIDATE", "ARCHIVED", _CIO),
    ),
    global_transitions=(_t("*", "PAPER_STOPPED", _RISK, _RISK_SVC, _HUMAN),),
)

IMPROVEMENT_PROPOSAL = MachineDefinition(
    name="improvement_proposal",
    initial="PROPOSED",
    terminal=frozenset({"REJECTED", "REVERTED"}),
    transitions=(
        _t("PROPOSED", "BENCHMARKING", _META, _CIO),
        _t("PROPOSED", "REJECTED", _META, _CIO, _HUMAN),
        _t("BENCHMARKING", "APPROVED", _VAL_SVC, _HUMAN, requires=("benchmark_result",)),
        _t("BENCHMARKING", "REJECTED", _VAL_SVC, _HUMAN, _META),
        _t("APPROVED", "ROLLED_OUT", _CIO, _HUMAN, requires=("version",)),
        _t("ROLLED_OUT", "REVERTED", _CIO, _HUMAN, _META, requires=("previous_version",)),
    ),
)

INNOVATION_PROPOSAL = MachineDefinition(
    name="innovation_proposal",
    initial="WATCHLIST",
    terminal=frozenset({"ACCEPTED_INTEGRATED", "REJECTED"}),
    transitions=(
        _t("WATCHLIST", "SPIKE", _TECH, _CIO),
        _t("WATCHLIST", "REJECTED", _TECH, _CIO, _META),
        _t("SPIKE", "BENCHMARKED", _TECH, requires=("benchmark_result",)),
        _t("SPIKE", "REJECTED", _TECH, _META),
        _t("BENCHMARKED", "ADR_REVIEW", _TECH, _META),
        _t("BENCHMARKED", "REJECTED", _TECH, _META, _HUMAN),
        _t("ADR_REVIEW", "ACCEPTED_INTEGRATED", _HUMAN, requires=("adr",)),
        _t("ADR_REVIEW", "REJECTED", _HUMAN),
    ),
)

# One research cycle = the 13-step first milestone. Each step checkpoints so a cycle resumes
# after interruption from the last completed step.
RESEARCH_CYCLE_STEPS: tuple[str, ...] = (
    "OBJECTIVE_RECEIVED",
    "DELEGATED",
    "RESEARCH_RETRIEVED",
    "KNOWLEDGE_STORED",
    "HYPOTHESIS_GENERATED",
    "DATA_ACQUIRED",
    "STRATEGY_EXECUTED",
    "BACKTEST_RECORDED",
    "VALIDATION_DECIDED",
    "REFLECTED",
    "LESSONS_STORED",
    "NEXT_EXPERIMENT_PROPOSED",
)

RESEARCH_CYCLE = MachineDefinition(
    name="research_cycle",
    initial=RESEARCH_CYCLE_STEPS[0],
    terminal=frozenset({"COMPLETED", "ABORTED"}),
    transitions=(
        *(_t(a, b, _CIO) for a, b in pairwise(RESEARCH_CYCLE_STEPS)),
        _t(RESEARCH_CYCLE_STEPS[-1], "COMPLETED", _CIO),
    ),
    global_transitions=(_t("*", "ABORTED", _CIO, _HUMAN, _RISK, _RISK_SVC),),
)

ALL_MACHINES: tuple[MachineDefinition, ...] = (
    STRATEGY_LIFECYCLE,
    IMPROVEMENT_PROPOSAL,
    INNOVATION_PROPOSAL,
    RESEARCH_CYCLE,
)
