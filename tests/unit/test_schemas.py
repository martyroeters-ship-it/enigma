from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from enigma.core import new_id, sha256_of
from enigma.schemas.common import AgentRole, ClaimLevel, Instrument, SystemActor
from enigma.schemas.messages import MessageEnvelope, MessageType
from enigma.schemas.quant import ExperimentalDesign, ParameterGrid
from enigma.schemas.research import Hypothesis
from enigma.schemas.validation import GateOutcome, GateResult, ValidationReport, aggregate

NOW = datetime(2026, 1, 1, tzinfo=UTC)
BTC_PERP = Instrument(
    symbol="BTCUSDT", base="BTC", quote="USDT", kind="perpetual", venue="binance_usdm"
)


def _gate(gid: str, ok: bool) -> GateResult:
    return GateResult(
        gate_id=gid,
        name=gid,
        result=GateOutcome.PASS if ok else GateOutcome.FAIL,
        stat=1.0,
        threshold=0.5,
        comparator=">=",
    )


def _hypothesis(**overrides: object) -> Hypothesis:
    base: dict[str, object] = dict(
        id=new_id("hyp"),
        objective_id="obj_1",
        family_id="fam_1",
        title="Funding carry",
        mechanism="Leveraged longs pay shorts when funding is positive.",
        prediction="Positive funding predicts negative next-8h perp return.",
        formal_statement="E[r_{t+1} | f_t > q] < 0",
        falsification_criteria=("Holdout net Sharpe <= 0",),
        instruments=(BTC_PERP,),
        horizon="8h",
        features=("funding_rate",),
        trial_budget=10,
        cost_model_version="cm-v1",
        generation_method="template:test",
        methodology_version="qm-0.1",
        created_by=AgentRole.QUANT_RESEARCH,
        created_at=NOW,
    )
    base.update(overrides)
    return Hypothesis.model_validate(base)


def test_aggregate_fails_closed_on_missing_gate() -> None:
    gates = (_gate("G01", True), _gate("G02", True))
    assert aggregate(gates, ("G01", "G02")) is GateOutcome.PASS
    assert aggregate(gates, ("G01", "G02", "G03")) is GateOutcome.FAIL
    assert aggregate((_gate("G01", False),), ("G01",)) is GateOutcome.FAIL
    assert aggregate(gates, ()) is GateOutcome.FAIL


def test_validation_report_overall_must_match_aggregate() -> None:
    kwargs: dict[str, object] = dict(
        id="vr_1",
        request_id="vq_1",
        strategy_package_id="pkg_1",
        protocol_version="0.1.0",
        required_gates=("G01",),
        gates=(_gate("G01", False),),
        holdout_dataset_ids=("ds_h",),
        trial_count=1,
        cost_assumptions_hash=sha256_of({}),
        validator_build_sha="abc",
        created_at=NOW,
    )
    with pytest.raises(ValidationError):
        ValidationReport.model_validate({**kwargs, "overall": GateOutcome.PASS})
    report = ValidationReport.model_validate({**kwargs, "overall": GateOutcome.FAIL})
    assert report.overall is GateOutcome.FAIL


def test_hypothesis_requires_mechanism_and_cannot_claim_l3() -> None:
    assert _hypothesis().claim_level is ClaimLevel.L1_HYPOTHESIS
    with pytest.raises(ValidationError):
        _hypothesis(mechanism="   ")
    with pytest.raises(ValidationError):
        _hypothesis(claim_level=ClaimLevel.L3_INDEPENDENTLY_VALIDATED)
    with pytest.raises(ValidationError):
        _hypothesis(falsification_criteria=())


def test_models_are_frozen_and_closed() -> None:
    h = _hypothesis()
    with pytest.raises(ValidationError):
        h.title = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        _hypothesis(unexpected_field=1)


def test_envelope_hash_must_match_payload() -> None:
    payload = {"task_id": "tsk_1"}
    env = MessageEnvelope(
        message_id="msg_1",
        correlation_id="cyc_1",
        timestamp_utc=NOW,
        from_agent=AgentRole.CIO,
        to_agent=AgentRole.INTELLIGENCE,
        message_type=MessageType.TASK_ASSIGNED,
        payload=payload,
        payload_hash=sha256_of(payload),
    )
    assert env.to_agent is AgentRole.INTELLIGENCE
    with pytest.raises(ValidationError):
        MessageEnvelope(
            message_id="msg_2",
            correlation_id="cyc_1",
            timestamp_utc=NOW,
            from_agent=SystemActor.HUMAN,
            to_agent=AgentRole.CIO,
            message_type=MessageType.TASK_ASSIGNED,
            payload=payload,
            payload_hash=sha256_of({"tampered": True}),
        )


def test_grid_size_is_the_product_of_the_listed_values() -> None:
    design = ExperimentalDesign(
        parameters=(ParameterGrid(name="W", values=(4, 8)),),
        grid_size=2,
        train_test_protocol="expanding",
        primary_metric="net_sharpe",
    )
    assert design.grid_size == 2
    mismatched = ExperimentalDesign(
        parameters=(ParameterGrid(name="W", values=(4, 8)),),
        grid_size=9,
        train_test_protocol="expanding",
        primary_metric="net_sharpe",
    )
    assert mismatched.grid_size == 2
