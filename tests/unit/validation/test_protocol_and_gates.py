from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from enigma.backtest.statistics import DeflatedSharpe
from enigma.core.errors import IntegrityError
from enigma.schemas.validation import GateOutcome
from enigma.strategy.leakage import TruncationCheckResult, TruncationMismatch
from enigma.validation import PROTOCOL_SHA256, load_protocol
from enigma.validation import gates as gate_lib
from enigma.validation.protocol import DEFAULT_PROTOCOL_PATH


def test_protocol_loads_with_pinned_hash() -> None:
    protocol = load_protocol()
    assert protocol.version == "0.1.0-slice"
    assert protocol.required_gates == ("G01", "G02", "G03", "G06", "G07", "G12", "G16")
    assert protocol.periods_per_year == 1095
    assert protocol.sha256 == PROTOCOL_SHA256
    assert hashlib.sha256(DEFAULT_PROTOCOL_PATH.read_bytes()).hexdigest() == PROTOCOL_SHA256


def test_protocol_edit_fails_closed(tmp_path: Path) -> None:
    edited = tmp_path / "protocol.yaml"
    edited.write_text(
        DEFAULT_PROTOCOL_PATH.read_text().replace(
            "min_dsr_probability: 0.95", "min_dsr_probability: 0.5"
        )
    )
    with pytest.raises(IntegrityError, match="does not match pinned"):
        load_protocol(edited)
    with pytest.raises(IntegrityError):
        load_protocol(tmp_path / "missing.yaml")


def _trunc(mismatches: int, checks: int = 4) -> TruncationCheckResult:
    return TruncationCheckResult(
        checkpoints={"s0": tuple(range(checks))},
        mismatches=tuple(TruncationMismatch("s0", i, i, 1.0) for i in range(mismatches)),
    )


CFG = load_protocol().gates


def test_g01() -> None:
    clean = {"available_not_after_open": 0}
    assert gate_lib.g01_pit_integrity(CFG["G01"], clean, _trunc(0)).result is GateOutcome.PASS
    assert gate_lib.g01_pit_integrity(CFG["G01"], clean, _trunc(1)).result is GateOutcome.FAIL
    bad_ts = {"available_not_after_open": 2}
    g = gate_lib.g01_pit_integrity(CFG["G01"], bad_ts, _trunc(0))
    assert g.result is GateOutcome.FAIL and g.stat == 2.0
    no_evidence = gate_lib.g01_pit_integrity(CFG["G01"], clean, _trunc(0, checks=0))
    assert no_evidence.result is GateOutcome.FAIL


def test_g02_embargo_and_violations() -> None:
    assert gate_lib.embargo_bars(CFG["G02"], 1) == 21
    assert gate_lib.embargo_bars(CFG["G02"], 10) == 30
    ok = gate_lib.g02_chronological_split(
        CFG["G02"],
        n_bars_total=1000,
        n_holdout_bars=300,
        n_embargo_bars=21,
        n_research_bars=679,
        overlaps=[],
    )
    assert ok.result is GateOutcome.PASS
    bad = gate_lib.g02_chronological_split(
        CFG["G02"],
        n_bars_total=1000,
        n_holdout_bars=300,
        n_embargo_bars=21,
        n_research_bars=679,
        overlaps=[{"kind": "x"}],
    )
    assert bad.result is GateOutcome.FAIL


@pytest.mark.parametrize(
    ("sharpe", "bars", "top", "expected"),
    [
        (0.6, 300, 0.1, GateOutcome.PASS),
        (0.5, 270, 0.01, GateOutcome.PASS),
        (0.49, 300, 0.1, GateOutcome.FAIL),
        (2.0, 269, 0.1, GateOutcome.FAIL),
        (2.0, 300, 0.0, GateOutcome.FAIL),
    ],
)
def test_g03_boundaries(sharpe: float, bars: int, top: float, expected: GateOutcome) -> None:
    g = gate_lib.g03_out_of_sample(
        CFG["G03"], net_sharpe=sharpe, n_bars=bars, top_days_removed_pnl=top
    )
    assert g.result is expected


def test_g06_g07_g12_g16_thresholds() -> None:
    def g06(p: float, registered: bool = True) -> GateOutcome:
        return gate_lib.g06_multiple_testing(
            CFG["G06"],
            candidate_adjusted_p=p,
            candidate_raw_p=p,
            family_pvalues=[p],
            registered=registered,
            registration_problems=[],
        ).result

    assert g06(0.05) is GateOutcome.PASS
    assert g06(0.0501) is GateOutcome.FAIL
    assert g06(0.001, registered=False) is GateOutcome.FAIL

    def dsr(p: float) -> DeflatedSharpe:
        return DeflatedSharpe(p, 0.1, 0.0, 300, 0.0, 3.0, 1, 0.0)

    assert gate_lib.g07_deflated_sharpe(CFG["G07"], dsr(0.95), "x").result is GateOutcome.PASS
    assert gate_lib.g07_deflated_sharpe(CFG["G07"], dsr(0.9499), "x").result is GateOutcome.FAIL

    def g12(s: float) -> GateOutcome:
        return gate_lib.g12_costs(
            CFG["G12"],
            stressed_net_sharpe=s,
            base_net_sharpe=1.0,
            cost_model_version="v",
            cost_pnl_base=0.0,
            cost_pnl_stressed=0.0,
        ).result

    assert g12(0.01) is GateOutcome.PASS
    assert g12(0.0) is GateOutcome.FAIL

    def g16(d: float) -> GateOutcome:
        return gate_lib.g16_reproducibility(
            CFG["G16"], max_abs_delta=d, position_delta=d, metric_delta=0.0, hashes_equal=d == 0
        ).result

    assert g16(1e-9) is GateOutcome.PASS
    assert g16(2e-9) is GateOutcome.FAIL
    assert g16(float("nan")) is GateOutcome.FAIL


def test_informative_gates() -> None:
    assert gate_lib.g04_walk_forward([1.0, -1.0, 2.0]).result is GateOutcome.PASS
    assert gate_lib.g04_walk_forward([]).result is GateOutcome.FAIL
    assert gate_lib.g08_pbo(None, "few").result is GateOutcome.FAIL
    assert gate_lib.g09_parameter_sensitivity(1.0, {}).result is GateOutcome.FAIL
    assert gate_lib.g09_parameter_sensitivity(1.0, {"a+": 0.6, "a-": 0.5}).result is (
        GateOutcome.PASS
    )
    assert gate_lib.g09_parameter_sensitivity(1.0, {"a+": None}).result is GateOutcome.FAIL
    regimes = {"low": {"bars": 50.0, "net_sharpe": 1.0}, "high": {"bars": 50.0, "net_sharpe": -1.0}}
    assert gate_lib.g10_regime_stress(regimes).result is GateOutcome.FAIL
    assert gate_lib.g11_bootstrap((0.1, 2.0), 1).result is GateOutcome.PASS
    assert gate_lib.g15_benchmark(1.0, {"flat": 0.0, "bh": 1.2}).result is GateOutcome.FAIL
    for gate in (gate_lib.g04_walk_forward([1.0]), gate_lib.g11_bootstrap((0.1, 1.0), 2)):
        assert gate.params["informative"] is True


def test_clean_makes_details_json_safe() -> None:
    out = gate_lib.clean(
        {"a": np.float64(1.5), "b": (np.int64(2), float("inf")), "c": np.bool_(True)}
    )
    assert out == {"a": 1.5, "b": [2, None], "c": True}
