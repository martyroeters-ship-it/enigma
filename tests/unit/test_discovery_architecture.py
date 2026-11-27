"""Scientific case, acquisition refusal, intervention, and the acceptance reader."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from cycle_rig import Rig

from enigma.learning.intervention import BenchmarkComparison, apply_intervention, rollback
from enigma.orchestrator.acceptance import assess, demonstrate, refuse_active_home
from enigma.orchestrator.report import build_report
from enigma.research.checks import resolve_column
from enigma.research.inquiry import register_case
from enigma.research.intelligence_bench import run_local_suite, score_failure_diagnosis
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import ImprovementProposal, ProposalStatus, ProposalTarget
from enigma.schemas.inquiry import ScientificCase
from enigma.schemas.knowledge import EvidenceItem

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_missing_observable_is_not_replaced() -> None:
    available = frozenset({"funding_rate", "perp_close", "basis"})
    assert resolve_column("open_interest", available, "signal") is None
    assert resolve_column("returns", available, "target") == "perp_close"


def test_model_text_is_not_evidence(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "ai.sqlite3")
    demonstrate(rig.stores, rig.fsm, None, now=NOW)
    case = rig.stores.documents.list(ScientificCase, limit=1)[0]
    evidence = {item.id: item for item in rig.stores.documents.list(EvidenceItem, limit=10)}
    evidence["ev_binance_schedule"] = evidence["ev_binance_schedule"].model_copy(
        update={"is_ai_interpretation": True}
    )
    rig.stores.documents.put(evidence["ev_binance_schedule"], recorded_by="test")
    again = register_case(rig.stores, case, held_observables={"funding_rate", "basis"}, now=NOW)
    assert again.ok is False
    assert "interpretation" in again.reason


def test_demonstration_registers_a_program_and_an_intervention(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "demo.sqlite3")
    demonstrate(rig.stores, rig.fsm, None, now=NOW)
    body = assess(rig.stores, model_status="absent")
    by_name = {item["name"]: item["status"] for item in body["criteria"]}
    assert by_name["methodology"] == "pass"
    assert by_name["reproducible backtest"] == "pass"
    assert by_name["new mechanism from evidence and a model"] == "pass"
    assert by_name["falsifiable hypothesis"] == "pass"
    assert by_name["backlog replenishment"] == "pass"
    assert by_name["implemented and benchmarked improvement"] == "pass"
    assert by_name["real market data acquired"] == "blocked"
    assert by_name["independent validation"] == "blocked"
    assert body["paid_usd"] == 0
    report = build_report(rig.stores)
    assert report["improvements_adopted"] == 1
    assert "scientific_failures" in report
    assert "inference" in report


def test_intervention_rolls_back(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "roll.sqlite3")
    proposal = ImprovementProposal(
        id="imp_rollback",
        target=ProposalTarget.WORKFLOW,
        target_ref="workflow.note",
        change_summary="Replace the note.",
        rationale="The new note is the intervention.",
        proposed_change={"evaluation": "paired score on a fixed note"},
        success_metric="score",
        baseline_metric_value=0.0,
        proposed_by=AgentRole.META_LEARNING,
        created_at=NOW,
    )
    rig.stores.documents.put(proposal, recorded_by="meta_learning")
    path = tmp_path / "note.txt"
    result = apply_intervention(
        rig.stores,
        rig.fsm,
        proposal,
        baseline_text="before\n",
        candidate_text="after\n",
        evaluation="paired score on a fixed note",
        comparison=BenchmarkComparison(0.0, 1.0, "tasks", 1, True),
        path=path,
        now=NOW,
    )
    assert result.status == "adopted"
    assert path.read_text(encoding="utf-8") == "after\n"
    stored = rig.stores.documents.get(ImprovementProposal, proposal.id)
    assert stored is not None and stored.status is ProposalStatus.ROLLED_OUT
    rolled = rollback(rig.stores, rig.fsm, stored, path, "before\n")
    assert rolled.status == "reverted"
    assert path.read_text(encoding="utf-8") == "before\n"


def test_protected_target_is_not_adopted(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "protect.sqlite3")
    proposal = ImprovementProposal(
        id="imp_protected",
        target=ProposalTarget.VALIDATION_CRITERIA,
        target_ref="validation",
        change_summary="Weaken a gate.",
        rationale="This must not be adopted.",
        proposed_change={"evaluation": "no"},
        success_metric="none",
        proposed_by=AgentRole.META_LEARNING,
        requires_human_approval=True,
        created_at=NOW,
    )
    result = apply_intervention(
        rig.stores,
        rig.fsm,
        proposal,
        baseline_text="a",
        candidate_text="b",
        evaluation="no",
        comparison=BenchmarkComparison(0.0, 1.0, "tasks", 1, True),
        path=tmp_path / "no.txt",
        now=NOW,
    )
    assert result.status == "refused"


def test_bench_scores_a_structured_case_and_does_not_call_a_paid_model() -> None:
    body = {
        "mechanism": "A crowded funding payment can reverse.",
        "persistence": "The payment clock is fixed.",
        "participants": [{"name": "basis trader"}],
        "supporting_evidence_ids": ["ev_a", "ev_b"],
        "contradictory_evidence_ids": ["ev_c"],
        "predictions": [
            {
                "statement": "The sign flips.",
                "observable": "funding_rate",
                "discriminates": "funding alone",
            }
        ],
        "data_needs": [{"observable": "open_interest"}],
        "program": {"design_class": "nonlinear", "computations": [{"op": "product"}]},
        "failure_diagnosis": (
            "G02 is a chronological method failure and is not evidence against the mechanism."
        ),
    }
    result = run_local_suite({"ev_a", "ev_b", "ev_c"}, complete=lambda _prompt: json.dumps(body))
    assert result["paid_calls"] == 0
    assert result["status"] == "scored"
    assert result["passed"] == result["total"]
    assert score_failure_diagnosis("the strategy lost money").passed is False
    prose = run_local_suite(
        {"ev_a", "ev_b"},
        complete=lambda _prompt: json.dumps({"mechanism": "x", "program": "a z-score"}),
        contradictory_ids=set(),
    )
    assert prose["status"] == "scored"
    assert prose["passed"] < prose["total"]


def test_accept_refuses_the_active_home(tmp_path: Path) -> None:
    home = tmp_path / "home"
    assert refuse_active_home(home, home) != ""
    assert refuse_active_home(home, tmp_path / "other") == ""
