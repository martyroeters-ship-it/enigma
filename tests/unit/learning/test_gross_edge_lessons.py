"""A negative gross Sharpe is not a cost story."""

from __future__ import annotations

from datetime import UTC, datetime

from test_llm_mechanism_lessons import _hyp

from enigma.learning.lessons import CycleEvidence, derive_lessons, propose_next
from enigma.schemas.common import AgentRole
from enigma.schemas.research import Experiment, ExperimentStatus, Hypothesis
from enigma.schemas.validation import GateOutcome, RedactedValidationSummary

NOW = datetime(2026, 3, 1, tzinfo=UTC)
H = "sha256:" + "ab" * 32


def _exp(n: int, gross: float, net: float) -> Experiment:
    return Experiment(
        id=f"exp_{n}",
        hypothesis_id="hyp_llm",
        family_id="fam_cryptocurrency",
        trial_index=n,
        strategy_package_id="pkg_1",
        dataset_version_id="ds_1",
        research_window=(NOW, NOW),
        code_sha="abc",
        config_hash=H,
        seed=n,
        status=ExperimentStatus.SUCCEEDED,
        metrics={"gross_sharpe": gross, "net_sharpe": net, "annual_turnover": 20.0},
        created_by=AgentRole.QUANT_RESEARCH,
    )


def _summary() -> RedactedValidationSummary:
    return RedactedValidationSummary(
        id="rvs_1",
        report_id="rpt_1",
        strategy_package_id="pkg_1",
        protocol_version="0.1.0-slice",
        overall=GateOutcome.FAIL,
        gate_outcomes={"G12": GateOutcome.FAIL},
        failed_gate_names=("G12:costs",),
    )


def _evidence(hypothesis: Hypothesis, experiments: tuple[Experiment, ...]) -> CycleEvidence:
    return CycleEvidence("cyc_gross", hypothesis, experiments, _summary(), "funding_carry")


def test_negative_gross_sharpe_is_not_blamed_on_costs() -> None:
    ev = _evidence(
        _hyp(),
        (
            _exp(1, -0.4, -0.8),
            _exp(2, -0.9, -1.2),
            _exp(3, -0.2, -0.5),
        ),
    )
    lessons = derive_lessons(ev, NOW)
    topics = {t for les in lessons for t in les.topics}
    assert "absent_gross_edge" in topics
    assert "cost_fragility" not in topics
    text = " ".join(les.statement for les in lessons)
    assert "did not remove an edge" in text
    nxt = propose_next(ev, lessons, {"funding_carry": 1}, NOW)
    assert "eroded" not in nxt.rationale
    assert nxt.hypothesis_delta.get("revisit") is None


def test_positive_gross_lost_after_costs_is_cost_fragility() -> None:
    ev = _evidence(
        _hyp(),
        (
            _exp(1, 0.8, -0.2),
            _exp(2, 0.4, -0.3),
            _exp(3, 0.2, -0.1),
        ),
    )
    lessons = derive_lessons(ev, NOW)
    topics = {t for les in lessons for t in les.topics}
    assert "cost_fragility" in topics
    assert "absent_gross_edge" not in topics
    nxt = propose_next(ev, lessons, {"funding_carry": 1}, NOW)
    assert nxt.hypothesis_delta.get("revisit") == "positive_gross_destroyed_by_costs"
    assert "positive gross" in nxt.rationale
