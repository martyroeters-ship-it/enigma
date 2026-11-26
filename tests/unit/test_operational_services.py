from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from _synthetic import make_panel
from enigma.agents.budget import BudgetTracker
from enigma.core.clock import FrozenClock
from enigma.orchestrator.services import OperationalServices
from enigma.research.mechanisms import MECHANISMS
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import Budget
from enigma.store import Stores
from enigma.strategy.sandbox import SandboxExecutionError, SandboxResult

REGISTRIES = Path(__file__).resolve().parents[2] / "registries"


def test_same_code_under_new_hypothesis_is_a_new_package() -> None:
    svc = OperationalServices.__new__(OperationalServices)
    svc.settings = SimpleNamespace(registries_dir=REGISTRIES)  # type: ignore[assignment]
    svc.clock = FrozenClock(datetime(2026, 1, 1, tzinfo=UTC))
    mech = MECHANISMS["basis_reversion"]
    params = {"lookback": 42, "z_entry": 2.0}

    def build(hypothesis_id: str) -> str:
        return svc.strategy_build(
            mech.source_template,
            mech.entrypoint,
            params,
            hypothesis_id,
            mech.id,
            mech.features,
            "cm-binance-usdm-v1",
        ).id

    assert build("hyp_A") == build("hyp_A")
    assert build("hyp_A") != build("hyp_B")


class _SandboxDouble:
    """Test double: reports CPU use and fails, like a crashing strategy."""

    limits = SimpleNamespace(cpu_seconds=10)

    def run(self, package: object, datasets: object, jobs: object) -> SandboxResult:
        return SandboxResult(
            run_id="run_x",
            package_id="pkg_x",
            package_hash="sha256:" + "0" * 64,
            ok=False,
            exit_code=1,
            positions={},
            truncated={},
            cpu_seconds=2.5,
            wall_seconds=3.0,
            violations=(),
            stderr_tail="boom",
            isolation="netns",
            error="boom",
        )


def test_sandbox_cpu_is_charged_even_when_the_run_fails(stores: Stores) -> None:
    svc = OperationalServices.__new__(OperationalServices)
    svc.stores = stores
    svc.sandbox = _SandboxDouble()  # type: ignore[assignment]
    svc.data_load_research = lambda _ds: make_panel(("BTCUSDT",), n_bars=50)  # type: ignore[method-assign]
    stores.documents.require = lambda *_a: SimpleNamespace(required_columns=("perp_close",))  # type: ignore[method-assign]
    tracker = BudgetTracker(Budget(), stores.costs, AgentRole.STRATEGY_ENGINEERING, "tsk_1")
    with pytest.raises(SandboxExecutionError):
        svc.backtest_run("pkg_x", "ds_x", 0, budget=tracker)
    assert tracker.cpu_seconds == 2.5
    assert stores.costs.totals()["cpu_seconds"] == 2.5
