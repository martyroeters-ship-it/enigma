"""Every mechanism template must pass static checks, run in the real sandbox, and be PIT."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from _synthetic import make_panel
from enigma.research.mechanisms import MECHANISMS
from enigma.schemas.common import AgentRole
from enigma.schemas.research import CostModel
from enigma.strategy import (
    SandboxJob,
    SandboxRunner,
    build_package,
    sample_checkpoints,
    symbol_arrays,
    truncation_invariance_check,
)

COST = CostModel(
    version="cm-test",
    taker_fee_bps=5,
    maker_fee_bps=2,
    half_spread_bps=1,
    impact_bps_per_million=2,
    capacity_notional_usd=5e6,
)


@pytest.mark.parametrize("mechanism_id", sorted(MECHANISMS))
def test_template_builds_runs_and_is_point_in_time(mechanism_id: str) -> None:
    mech = MECHANISMS[mechanism_id]
    params = {k: v[len(v) // 2] for k, v in mech.param_grid.items()}
    pkg = build_package(
        mech.source_template,
        mech.entrypoint,
        params,
        COST,
        "hyp_test",
        name=mechanism_id,
        required_columns=mech.features,
        created_by=AgentRole.STRATEGY_ENGINEERING,
        created_at=datetime.now(UTC),
    )
    frame = make_panel(("BTCUSDT",), n_bars=400, seed=5, funding_mean=1e-4)
    data = {"s0": symbol_arrays(frame, "BTCUSDT", mech.features)}
    runner = SandboxRunner()
    result = runner.run(pkg, data, [SandboxJob("j", "s0")])
    assert result.ok, result.error
    positions = result.positions["j"]
    assert positions.shape == (400,)
    assert np.all(np.abs(positions) <= 1.0)
    assert np.any(positions != 0.0), "template never takes a position on synthetic data"
    check = truncation_invariance_check(
        pkg, data, {"s0": sample_checkpoints(400, 6, seed=2)}, runner=runner
    )
    assert check.passed, check.mismatches
