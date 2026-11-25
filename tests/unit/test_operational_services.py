from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from enigma.core.clock import FrozenClock
from enigma.orchestrator.services import OperationalServices
from enigma.research.mechanisms import MECHANISMS

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
