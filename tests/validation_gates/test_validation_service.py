from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import polars as pl
import pytest

from _synthetic import (
    BASIS_SIGN,
    CRASHES,
    LEAKY,
    MOMENTUM,
    RANDOM_SIGN,
    cost_model,
    make_panel,
    package,
)
from _validation_env import EnvFactory
from enigma.core.errors import IntegrityError
from enigma.fsm import STRATEGY_LIFECYCLE
from enigma.schemas.knowledge import FailureCondition, FailureKind
from enigma.schemas.validation import GateOutcome, RedactedValidationSummary, ValidationReport
from enigma.validation import InMemoryHoldoutSource


def _gates(report: ValidationReport) -> dict[str, GateOutcome]:
    return {g.gate_id: g.result for g in report.gates}


def test_planted_edge_passes_all_required_gates(make_env: EnvFactory) -> None:
    """Machinery check on synthetic data with a planted, PIT-observable edge (not a real result)."""
    env = make_env(make_panel(seed=11, edge=0.25))
    pkg = package(BASIS_SIGN, "BasisSign", ("basis",), {"threshold": 0.0001})
    others = [package(MOMENTUM, "Momentum", ("perp_close",), {"lookback": k}) for k in (3, 6)]
    report = env.service.validate(env.submit(pkg, other_trials=others))
    failed = {g.gate_id: g.details for g in report.gates if g.result is GateOutcome.FAIL}
    assert report.overall is GateOutcome.PASS, failed
    assert report.trial_count == 3
    assert report.protocol_version == "0.1.0-slice"
    assert report.holdout_dataset_ids == ("dsv_validation_test",)
    assert report.validator_build_sha
    assert {g.gate_id for g in report.gates} >= set(report.required_gates)
    assert env.fsm.state(STRATEGY_LIFECYCLE, pkg.id) == "VALIDATION_PASSED"
    event = env.fsm.history(STRATEGY_LIFECYCLE, pkg.id)[-1]
    assert event.actor == "validation_service"
    assert event.artifact_hashes["validation_report"].startswith("sha256:")
    assert env.service.custody.get_report(report.id) == report
    assert env.control.require(ValidationReport, report.id) == report
    summaries = env.control.list(RedactedValidationSummary, where={"report_id": report.id})
    assert summaries[0].overall is GateOutcome.PASS
    actions = [e.action for e in env.stores.audit.entries(subject_id=pkg.id, limit=500)]
    assert {"validation.started", "validation.completed", "sandbox.run"} <= set(actions)
    env.stores.audit.verify()


def test_leaky_package_fails_g01_at10(make_env: EnvFactory) -> None:
    """AT10: a package that uses the next bar's close fails G01 and overall."""
    env = make_env(make_panel(seed=12))
    # Zero latency makes the leak profitable, so every performance gate is fooled.
    pkg = package(LEAKY, "Leaky", ("perp_close",), cm=cost_model(latency_bars=0))
    report = env.service.validate(env.submit(pkg))
    g01 = next(g for g in report.gates if g.gate_id == "G01")
    assert g01.result is GateOutcome.FAIL
    assert g01.details["truncation_mismatches"]
    assert report.overall is GateOutcome.FAIL
    assert report.net_sharpe_holdout is not None and report.net_sharpe_holdout > 5
    failed_required = {
        g.gate_id
        for g in report.gates
        if g.gate_id in report.required_gates and g.result is GateOutcome.FAIL
    }
    assert failed_required == {"G01"}
    assert env.fsm.state(STRATEGY_LIFECYCLE, pkg.id) == "VALIDATION_FAILED"
    conditions = env.control.list(FailureCondition, where={"subject_id": pkg.id})
    assert any(c.kind is FailureKind.LEAKAGE and c.gate_id == "G01" for c in conditions)


@pytest.mark.parametrize("seed", [21, 22, 23])
def test_random_strategy_on_random_walk_fails(make_env: EnvFactory, seed: int) -> None:
    """False-discovery control: no edge in the data, no skill in the strategy -> FAIL.

    Costs are zeroed so the statistical gates, not trading costs, must reject it.
    """
    free = cost_model(taker_fee_bps=0, half_spread_bps=0, impact_bps_per_million=0)
    env = make_env(make_panel(seed=seed, edge=0.0))
    pkg = package(RANDOM_SIGN, "RandomSign", ("perp_close",), {"seed": seed}, cm=free)
    trials = [
        package(RANDOM_SIGN, "RandomSign", ("perp_close",), {"seed": seed * 100 + k}, cm=free)
        for k in range(4)
    ]
    report = env.service.validate(env.submit(pkg, other_trials=trials))
    assert report.overall is GateOutcome.FAIL
    assert report.trial_count == 5
    gates = _gates(report)
    assert gates["G01"] is GateOutcome.PASS  # it is honest, just useless
    assert gates["G07"] is GateOutcome.FAIL
    assert gates["G06"] is GateOutcome.FAIL


def test_momentum_on_random_walk_fails(make_env: EnvFactory) -> None:
    env = make_env(make_panel(seed=31, edge=0.0))
    report = env.service.validate(
        env.submit(package(MOMENTUM, "Momentum", ("perp_close",), {"lookback": 3}))
    )
    assert report.overall is GateOutcome.FAIL


def test_crashing_strategy_fails_closed(make_env: EnvFactory) -> None:
    env = make_env(make_panel(seed=13, n_bars=1000))
    pkg = package(CRASHES, "Crashes", ("perp_close",))
    report = env.service.validate(env.submit(pkg))
    assert report.overall is GateOutcome.FAIL
    assert set(_gates(report)) == set(report.required_gates)
    assert all(g.details["reason"].startswith("strategy failed") for g in report.gates)
    conditions = env.control.list(FailureCondition, where={"subject_id": pkg.id})
    assert {c.kind for c in conditions} == {FailureKind.IMPLEMENTATION_ERROR}
    assert env.fsm.state(STRATEGY_LIFECYCLE, pkg.id) == "VALIDATION_FAILED"


def test_capacity_violation_fails_closed(make_env: EnvFactory) -> None:
    env = make_env(make_panel(seed=14, n_bars=1000))
    tiny = cost_model(capacity_notional_usd=10_000.0)
    pkg = package(MOMENTUM, "Momentum", ("perp_close",), cm=tiny)
    report = env.service.validate(env.submit(pkg))
    assert report.overall is GateOutcome.FAIL
    kinds = {c.kind for c in env.control.list(FailureCondition, where={"subject_id": pkg.id})}
    assert kinds == {FailureKind.CAPACITY}


def test_research_window_overlapping_embargo_fails_g02(make_env: EnvFactory) -> None:
    frame = make_panel(seed=15, edge=0.25)
    env = make_env(frame)
    pkg = package(BASIS_SIGN, "BasisSign", ("basis",), {"threshold": 0.0001})
    late = frame.get_column("ts").max()
    report = env.service.validate(env.submit(pkg, window_end=late))
    g02 = next(g for g in report.gates if g.gate_id == "G02")
    assert g02.result is GateOutcome.FAIL
    assert g02.details["violations"][0]["kind"] == "experiment_window_overlaps_embargo"
    assert report.overall is GateOutcome.FAIL


def test_unregistered_package_fails_g06(make_env: EnvFactory) -> None:
    env = make_env(make_panel(seed=16, edge=0.25))
    pkg = package(BASIS_SIGN, "BasisSign", ("basis",), {"threshold": 0.0001})
    other = package(MOMENTUM, "Momentum", ("perp_close",))
    report = env.service.validate(env.submit(pkg, other_trials=[other], register=False))
    g06 = next(g for g in report.gates if g.gate_id == "G06")
    assert g06.result is GateOutcome.FAIL
    assert not g06.details["registered"]
    assert report.overall is GateOutcome.FAIL


def test_revalidating_a_decided_request_returns_the_recorded_report(
    make_env: EnvFactory,
) -> None:
    env = make_env(make_panel(seed=17, n_bars=1000))
    pkg = package(MOMENTUM, "Momentum", ("perp_close",))
    request = env.submit(pkg)
    first = env.service.validate(request)
    again = env.service.validate(request)
    assert again.id == first.id
    assert again.overall == first.overall


def test_holdout_manifest_is_sealed_and_tamper_evident(make_env: EnvFactory) -> None:
    longer = make_panel(seed=18, n_bars=1100)
    cut = longer.get_column("ts").unique().sort()[1000]
    frame = longer.filter(pl.col("ts") < cut)
    env = make_env(frame)
    symbols = ["BTCUSDT", "ETHUSDT"]
    first = env.service.seal_holdout(symbols)
    env.service.holdout_source = InMemoryHoldoutSource(longer, ("dsv_more",))
    assert env.service.seal_holdout(symbols) == first  # boundary does not move with new data
    pkg = package(MOMENTUM, "Momentum", ("perp_close",))
    request = env.submit(pkg)
    tampered = frame.with_columns(
        pl.when(pl.col("ts") == pl.col("ts").max())
        .then(pl.col("perp_close") * 1.01)
        .otherwise(pl.col("perp_close"))
        .alias("perp_close")
    )
    env.service.holdout_source = InMemoryHoldoutSource(tampered, ("dsv_tampered",))
    with pytest.raises(IntegrityError):
        env.service.validate(request)
    assert env.fsm.state(STRATEGY_LIFECYCLE, pkg.id) == "VALIDATION_QUEUED"


class _AvailabilitySource(InMemoryHoldoutSource):
    """Selects bars the way the data catalog does: `ts >= start and available_at <= end`."""

    def __init__(self, frame: pl.DataFrame) -> None:
        super().__init__(frame, ("dsv_catalog_like",))
        self._full = frame

    def available_range(self, symbols: Sequence[str]) -> tuple[datetime, datetime]:
        return self._full["ts"].min(), self._full["available_at"].max()  # type: ignore[return-value]

    def load_holdout(self, symbols: Sequence[str], start: datetime, end: datetime) -> pl.DataFrame:
        return self._full.filter(
            pl.col("symbol").is_in(list(symbols))
            & (pl.col("ts") >= start)
            & (pl.col("available_at") <= end)
        )


def test_reseal_with_availability_based_source_keeps_last_bar(make_env: EnvFactory) -> None:
    frame = make_panel(seed=19, n_bars=1000)
    env = make_env(frame)
    env.service.holdout_source = _AvailabilitySource(frame)
    symbols = ["BTCUSDT", "ETHUSDT"]
    first = env.service.seal_holdout(symbols)
    assert env.service.seal_holdout(symbols) == first
