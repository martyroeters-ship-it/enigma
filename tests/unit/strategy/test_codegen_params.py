"""StrategyCodeGenerator default params include schedule fields local models read."""

from __future__ import annotations

from types import SimpleNamespace

from enigma.schemas.quant import ParameterGrid
from enigma.strategy.codegen import _default_params, _normalize_generated_source


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


def test_normalize_strips_mapping_annotations_and_renames_configure() -> None:
    raw = """
import numpy as np
from enigma.strategy.interface import Strategy, StrategyMeta

class FundingRateStrategy(Strategy):
    def __configure__(self, params):
        self.params = dict(params)

    def required_columns(self) -> tuple[str, ...]:
        return ("funding_rate",)

    def metadata(self):
        return StrategyMeta(name="x", version="1", hypothesis_id="h", cost_model_version="c")

    def target_positions(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.zeros_like(data["funding_rate"])
"""
    fixed = _normalize_generated_source(raw)
    assert "def configure(" in fixed
    assert "__configure__" not in fixed
    assert "Mapping" not in fixed
    assert "tuple[str, ...]" not in fixed
    assert "def target_positions(self, data)" in fixed
