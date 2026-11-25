"""StrategyCodeGenerator default params include schedule fields local models read."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from enigma.schemas.quant import ParameterGrid
from enigma.strategy.codegen import _default_params


def test_default_params_include_rebalance_and_holding() -> None:
    spec = SimpleNamespace(
        parameters=(ParameterGrid(name="funding_rate_threshold", values=(-0.05, -0.02)),),
        rebalance_bars=1,
        holding_bars=8,
        label_horizon_bars=8,
    )
    params = _default_params(spec)  # type: ignore[arg-type]
    assert params["funding_rate_threshold"] == -0.05
    assert params["rebalance_bars"] == 1
    assert params["holding_bars"] == 8
    assert params["label_horizon_bars"] == 8
