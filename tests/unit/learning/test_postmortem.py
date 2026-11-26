"""Postmortems separate method defects from variance-only validation failures."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from enigma.learning.postmortem import (
    classify_abort,
    classify_validation_failure,
    postmortem_for_abort,
    postmortem_for_validation,
)
from enigma.schemas.common import AgentRole, ClaimLevel
from enigma.schemas.governance import ImprovementProposal, ProposalStatus, ProposalTarget
from enigma.schemas.learning import ErrorClass, Postmortem

NOW = datetime(2026, 11, 25, tzinfo=UTC)

# Failed names stored on the primary live LLM cycle summary.
_LIVE_FAILS = (
    "G02:chronological_split",
    "G03:out_of_sample",
    "G06:multiple_testing",
    "G07:deflated_sharpe",
    "G10:regime_stress",
    "G11:bootstrap",
)


def test_variance_only_gates_are_not_method_errors() -> None:
    assert (
        classify_validation_failure(
            ("G03:out_of_sample", "G06:multiple_testing", "G07:deflated_sharpe")
        )
        is ErrorClass.VARIANCE
    )


def test_live_cycle_fail_includes_a_method_gate() -> None:
    """G02 on cyc_01M89P5FHCNY5S7H4AEGSR5H4H is a split defect, not sampling noise."""
    assert classify_validation_failure(_LIVE_FAILS) is ErrorClass.METHOD
    record = postmortem_for_validation(
        "cyc_01M89P5FHCNY5S7H4AEGSR5H4H",
        summary_id="vsm_17826111185C41C2E27D89C50C",
        failed_gate_names=_LIVE_FAILS,
        now=NOW,
    )
    assert record.error_class is ErrorClass.METHOD
    assert record.claim_level is ClaimLevel.L1_HYPOTHESIS
    assert "vsm_17826111185C41C2E27D89C50C" in record.evidence_refs


def test_reproducibility_and_package_collision_are_method_errors() -> None:
    assert (
        classify_abort("twin backtests were not reproducible", "BACKTEST_RECORDED")
        is ErrorClass.METHOD
    )
    assert (
        classify_abort(
            "InvalidTransitionError: cannot go RESEARCH_REJECTED -> DRAFT",
            "STRATEGY_EXECUTED",
        )
        is ErrorClass.METHOD
    )
    record = postmortem_for_abort(
        "cyc_abort",
        abort_reason="twin backtests were not reproducible",
        step="BACKTEST_RECORDED",
        now=NOW,
    )
    assert "CPU" in record.counterfactual or "resource usage" in record.counterfactual


def test_adoption_requires_a_postmortem_id() -> None:
    common = dict(
        id="imp_cite",
        target=ProposalTarget.WORKFLOW,
        target_ref="cycle",
        change_summary="record the failure class",
        rationale="method errors were unlabeled",
        success_metric="postmortems_present",
        proposed_by=AgentRole.META_LEARNING,
        created_at=NOW,
    )
    with pytest.raises(ValidationError, match="postmortem"):
        ImprovementProposal(**common, status=ProposalStatus.APPROVED)
    cited = ImprovementProposal(
        **common, status=ProposalStatus.APPROVED, postmortem_ids=("pm_live",)
    )
    assert cited.postmortem_ids == ("pm_live",)
    proposed = ImprovementProposal(**common)
    assert proposed.status is ProposalStatus.PROPOSED
    assert proposed.postmortem_ids == ()


def test_postmortem_cannot_claim_independent_validation() -> None:
    with pytest.raises(ValidationError, match="L3"):
        Postmortem(
            id="pm_test",
            cycle_id="cyc_test",
            error_class=ErrorClass.VARIANCE,
            statement="no",
            counterfactual="no",
            claim_level=ClaimLevel.L3_INDEPENDENTLY_VALIDATED,
            created_by=AgentRole.META_LEARNING,
            created_at=NOW,
        )
