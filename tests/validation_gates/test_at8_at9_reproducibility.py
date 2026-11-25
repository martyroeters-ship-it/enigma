"""AT8 (twin backtests) and AT9 (twin validation reports)."""

from __future__ import annotations

import numpy as np

from _synthetic import BASIS_SIGN, MOMENTUM, make_panel, package
from _validation_env import EnvFactory
from enigma.backtest import BarBacktestEngine
from enigma.strategy import SandboxJob, SandboxRunner, symbol_arrays

EPS = 1e-9


def test_at8_twin_backtests_identical() -> None:
    frame = make_panel(seed=41, edge=0.2)
    pkg = package(BASIS_SIGN, "BasisSign", ("basis",), {"threshold": 0.0001})
    symbols = ("BTCUSDT", "ETHUSDT")
    data = {f"s{i}": symbol_arrays(frame, s, ("basis",)) for i, s in enumerate(symbols)}
    jobs = [SandboxJob(name, name) for name in data]
    runner = SandboxRunner()
    engine = BarBacktestEngine(1095)
    results = []
    for _ in range(2):
        run = runner.run(pkg, data, jobs).require_ok()
        positions = {s: run.positions[f"s{i}"] for i, s in enumerate(symbols)}
        results.append(engine.run(positions, frame, pkg.cost_model, 7, notional_usd=1e5))
    a, b = results
    assert a.result_hash == b.result_hash
    assert max(abs(a.metrics[k] - b.metrics[k]) for k in a.metrics) <= EPS
    assert float(np.max(np.abs(a.net_returns - b.net_returns))) <= EPS


def test_at9_twin_validation_reports_identical(make_env: EnvFactory) -> None:
    frame = make_panel(seed=42, edge=0.25)
    pkg = package(BASIS_SIGN, "BasisSign", ("basis",), {"threshold": 0.0001}, package_id="pkg_twin")
    trial = package(MOMENTUM, "Momentum", ("perp_close",), {"lookback": 4}, package_id="pkg_t2")
    reports = []
    for name in ("twin_a", "twin_b"):
        env = make_env(frame, name=name)
        reports.append(env.service.validate(env.submit(pkg, other_trials=[trial])))
    a, b = reports
    assert a.id != b.id
    assert a.overall is b.overall
    assert a.gates == b.gates
    assert (a.deflated_sharpe, a.pbo, a.net_sharpe_holdout, a.trial_count) == (
        b.deflated_sharpe,
        b.pbo,
        b.net_sharpe_holdout,
        b.trial_count,
    )
    assert a.cost_assumptions_hash == b.cost_assumptions_hash
