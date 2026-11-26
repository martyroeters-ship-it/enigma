"""Golden tests: engine output vs PnL computed by hand from the documented timing rules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from _synthetic import cost_model, make_panel
from enigma.backtest import BarBacktestEngine, CapacityExceededError

T0 = datetime(2024, 1, 1, tzinfo=UTC)
H8 = timedelta(hours=8)


def _frame(symbol: str, closes: list[float], funding: list[float | None]) -> pl.DataFrame:
    ts = [T0 + i * H8 for i in range(len(closes))]
    return pl.DataFrame(
        {
            "symbol": [symbol] * len(closes),
            "ts": ts,
            "available_at": [t + H8 for t in ts],
            "perp_close": closes,
            "funding_rate": funding,
        },
        schema_overrides={"ts": pl.Datetime("us", "UTC"), "available_at": pl.Datetime("us", "UTC")},
    )


CLOSES = [100.0, 110.0, 99.0, 99.0, 108.9]
FUNDING: list[float | None] = [None, 0.001, -0.002, 0.0, 0.001]
TARGETS = np.array([1.0, 1.0, -1.0, 0.0, 0.0])
# taker 4 + half-spread 1 = 5 bps linear; impact 10 bps per $1M on a $100k sleeve = 1 bp per
# unit of turnover, so cost = |d| * (5 + |d|) / 1e4.
CM = cost_model(impact_bps_per_million=10.0, capacity_notional_usd=1_000_000.0)


def test_golden_single_symbol_latency_one() -> None:
    result = BarBacktestEngine(periods_per_year=1095).run(
        {"X": TARGETS}, _frame("X", CLOSES, FUNDING), CM, seed=7, notional_usd=100_000.0
    )
    s = result.per_symbol["X"]
    np.testing.assert_allclose(s.executed, [0, 1, 1, -1, 0])
    np.testing.assert_allclose(s.held, [0, 0, 1, 1, -1])
    np.testing.assert_allclose(s.price, [0, 0, -0.1, 0, -0.1], atol=1e-15)
    # long pays positive funding, receives negative; short receives positive funding
    np.testing.assert_allclose(s.funding, [0, 0, 0.002, 0, 0.001], atol=1e-15)
    np.testing.assert_allclose(s.turnover, [0, 1, 0, 2, 1])
    np.testing.assert_allclose(s.cost, [0, 6e-4, 0, 1.4e-3, 6e-4], atol=1e-15)
    expected_net = np.array([0, -6e-4, -0.098, -1.4e-3, -0.0996])
    np.testing.assert_allclose(result.net_returns, expected_net, atol=1e-15)
    np.testing.assert_allclose(result.gross_returns, [0, 0, -0.098, 0, -0.099], atol=1e-15)
    np.testing.assert_allclose(result.equity, np.cumprod(1 + expected_net), rtol=1e-15)
    m = result.metrics
    assert m["trades"] == 3
    assert m["turnover"] == pytest.approx(4.0)
    assert m["funding_pnl"] == pytest.approx(0.003)
    assert m["cost_pnl"] == pytest.approx(-2.6e-3)
    assert m["price_pnl"] == pytest.approx(-0.2)
    assert m["exposure"] == pytest.approx(3 / 5)
    assert m["hit_rate"] == pytest.approx(0.0)
    assert m["total_return"] == pytest.approx(np.prod(1 + expected_net) - 1)
    assert m["max_drawdown"] == pytest.approx(1 - np.prod(1 + expected_net))
    mean, sd = expected_net.mean(), expected_net.std(ddof=1)
    assert m["net_sharpe"] == pytest.approx(mean / sd * np.sqrt(1095))


def test_golden_latency_zero_no_funding_and_cost_stress() -> None:
    cm0 = CM.model_copy(update={"latency_bars": 0, "apply_funding": False})
    engine = BarBacktestEngine()
    base = engine.run({"X": TARGETS}, _frame("X", CLOSES, FUNDING), cm0, 1, notional_usd=1e5)
    s = base.per_symbol["X"]
    np.testing.assert_allclose(s.held, [0, 1, 1, -1, 0])
    np.testing.assert_allclose(s.price, [0, 0.1, -0.1, 0, 0], atol=1e-15)
    np.testing.assert_allclose(s.funding, 0.0)
    stressed = engine.run(
        {"X": TARGETS}, _frame("X", CLOSES, FUNDING), cm0, 1, notional_usd=1e5, cost_multiplier=2
    )
    np.testing.assert_allclose(stressed.per_symbol["X"].cost, 2 * s.cost)
    np.testing.assert_allclose(stressed.per_symbol["X"].gross, s.gross)


def test_multi_symbol_equal_weight_and_capacity() -> None:
    frame = pl.concat([_frame("A", CLOSES, FUNDING), _frame("B", CLOSES, FUNDING)])
    engine = BarBacktestEngine()
    two = engine.run({"A": TARGETS, "B": -TARGETS}, frame, CM, 0, notional_usd=200_000.0)
    a, b = two.per_symbol["A"], two.per_symbol["B"]
    np.testing.assert_allclose(two.net_returns, (a.net + b.net) / 2)
    # each $100k sleeve has the same cost as the single-symbol golden case
    np.testing.assert_allclose(a.cost, [0, 6e-4, 0, 1.4e-3, 6e-4], atol=1e-15)
    np.testing.assert_allclose(two.series["trades"], [0, 2, 0, 2, 2])
    with pytest.raises(CapacityExceededError):
        engine.run({"A": TARGETS}, frame, CM, 0, notional_usd=1_000_001.0)
    engine.run({"A": TARGETS, "B": TARGETS}, frame, CM, 0, notional_usd=2_000_000.0)


def test_rejects_bad_positions() -> None:
    frame = _frame("X", CLOSES, FUNDING)
    engine = BarBacktestEngine()
    with pytest.raises(ValueError, match="shape"):
        engine.run({"X": TARGETS[:3]}, frame, CM, 0, notional_usd=1e5)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        engine.run({"X": TARGETS * 2}, frame, CM, 0, notional_usd=1e5)


def test_window_and_result_hash_are_deterministic() -> None:
    frame = make_panel(n_bars=300, seed=5)
    rng = np.random.default_rng(0)
    pos = {s: np.sign(rng.normal(size=300)) for s in ("BTCUSDT", "ETHUSDT")}
    engine = BarBacktestEngine()
    a = engine.run(pos, frame, CM, 11, notional_usd=1e5)
    b = engine.run(pos, frame, CM, 11, notional_usd=1e5)
    assert a.result_hash == b.result_hash
    assert engine.run(pos, frame, CM, 12, notional_usd=1e5).result_hash != a.result_hash
    cut = int(a.ts[200])
    tail = a.window(cut, None)
    assert tail.ts.size == 100
    np.testing.assert_array_equal(tail.net_returns, a.net_returns[200:])
    assert tail.metrics["n_bars"] == 100
