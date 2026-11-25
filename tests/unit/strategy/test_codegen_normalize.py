"""Mechanical codegen normalizers (no LLM)."""

from __future__ import annotations

from datetime import UTC, datetime

from _synthetic import cost_model
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import ExecutionMode
from enigma.schemas.quant import BaselineSpec, FeatureDef, ParameterGrid, StrategySpec
from enigma.strategy.codegen import _default_params, _normalize_generated_source, _sandbox_smoke
from enigma.strategy.package import build_package
from enigma.strategy.static_check import check_source

NOW = datetime(2026, 3, 1, tzinfo=UTC)


def _spec() -> StrategySpec:
    feature = FeatureDef(
        name="funding_rate",
        formula="funding_rate",
        inputs=("funding_rate",),
        lookback_bars=1,
        min_lag_bars=0,
    )
    draft = StrategySpec.model_construct(
        id="sspec_test",
        hypothesis_spec_id="rhs_legacy_test",
        hypothesis_id="hyp_test",
        name="funding_sign",
        signal_math="sign(funding)",
        features=(feature,),
        parameters=(ParameterGrid(name="funding_rate_threshold", values=(-0.05,)),),
        position_rule="clip(signal, -1, 1)",
        position_bounds=(-1.0, 1.0),
        rebalance_bars=1,
        holding_bars=8,
        complexity_class="rule",
        complexity_justification="test",
        baseline=BaselineSpec(
            name="sign_first_feature",
            signal_math="sign(funding_rate)",
            features=(feature,),
            parameters=(),
            position_rule="clip(sign(funding_rate), -1, 1)",
        ),
        required_columns=("funding_rate",),
        label_horizon_bars=8,
        execution_mode=ExecutionMode.LLM,
        prompt_ref="",
        provenance=(),
        created_by=AgentRole.STRATEGY_ENGINEERING,
        created_at=NOW,
        content_hash="sha256:" + ("0" * 64),
    )
    content_hash = StrategySpec.hash_fields(draft.model_dump(mode="json"))
    return StrategySpec.model_validate(
        {**draft.model_dump(mode="json"), "content_hash": content_hash}
    )


def test_default_params_include_cadence_fields() -> None:
    params = _default_params(_spec())
    assert params["funding_rate_threshold"] == -0.05
    assert params["rebalance_bars"] == 1
    assert params["holding_bars"] == 8
    assert params["label_horizon_bars"] == 8


_LLM_SLIP_SOURCE = """
import numpy as np
from enigma.strategy.interface import Strategy, StrategyMeta

class S(Strategy):
    def configure(self, params):
        self.params = dict(params)
    def required_columns(self):
        return ("funding_rate",)
    def metadata(self):
        return StrategyMeta(name="S", version="1", id="hyp_x", cost_model_version="cm-test-1")
    def target_positions(self, data):
        x = self.inputs["funding_rate"]
        thr = float(self.params["funding_rate_threshold"])
        rb = int(self.params["rebalance_bars"])
        out = np.zeros(len(x))
        for i in range(len(x)):
            if i % max(rb, 1) == 0 and x[i] < thr:
                out[i] = 1.0
            elif i > 0:
                out[i] = out[i - 1]
        return np.clip(out, -1.0, 1.0)
"""


def test_normalize_rewrites_self_inputs_and_strategy_meta_id() -> None:
    out = _normalize_generated_source(_LLM_SLIP_SOURCE)
    assert "self.inputs" not in out
    assert "data[" in out
    assert "hypothesis_id=" in out
    call = out.split("return StrategyMeta(", 1)[1].split(")", 1)[0]
    assert "hypothesis_id=" in call
    assert not any(part.strip().startswith("id=") for part in call.split(","))


def test_normalize_strips_mapping_annotations() -> None:
    src = (
        "import numpy as np\n"
        "from enigma.strategy.interface import Strategy, StrategyMeta\n\n"
        "class S(Strategy):\n"
        "    def configure(self, params: Mapping[str, float]) -> None:\n"
        "        self.params = dict(params)\n"
        "    def required_columns(self) -> tuple[str, ...]:\n"
        "        return ('funding_rate',)\n"
        "    def metadata(self) -> StrategyMeta:\n"
        "        return StrategyMeta(name='S', version='1', hypothesis_id='hyp_x')\n"
        "    def target_positions(self, data: Mapping[str, np.ndarray]) -> np.ndarray:\n"
        "        x = data['funding_rate']\n"
        "        return np.zeros(len(x))\n"
    )
    out = _normalize_generated_source(src)
    assert "Mapping" not in out
    assert "def target_positions(self, data)" in out
    assert check_source(out, "S") == ()


def test_normalized_slip_source_passes_sandbox_smoke() -> None:
    spec = _spec()
    source = _normalize_generated_source(_LLM_SLIP_SOURCE)
    assert check_source(source, "S") == ()
    package = build_package(
        source,
        "S",
        _default_params(spec),
        cost_model(),
        spec.hypothesis_id,
        name=spec.name,
        required_columns=spec.required_columns,
        created_by=AgentRole.STRATEGY_ENGINEERING,
        created_at=NOW,
        label_horizon_bars=spec.label_horizon_bars,
    )
    assert _sandbox_smoke(package, spec.required_columns, _default_params(spec)) == ()
