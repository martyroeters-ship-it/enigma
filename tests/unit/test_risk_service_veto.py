"""Production risk limits veto paper even when validation passes.

Paper broker code stays in tree. Default limits keep paper and live capital off,
and a CLEARANCE is required before any session can open.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from enigma.core.clock import FrozenClock
from enigma.execution.paper import PaperBroker
from enigma.risk.acknowledge import acknowledge_risk
from enigma.risk.service import RiskLimits, RiskService
from enigma.schemas.common import AgentRole, SystemActor
from enigma.schemas.execution import HumanRiskAcknowledge, PaperSession
from enigma.schemas.research import CostModel, StrategyPackage
from enigma.schemas.validation import GateOutcome, RedactedValidationSummary
from enigma.store import Stores

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIMITS = ROOT / "registries" / "risk" / "limits.yaml"
NOW = datetime(2026, 3, 1, tzinfo=UTC)


def test_default_limits_keep_paper_and_live_off() -> None:
    limits = RiskLimits.load(DEFAULT_LIMITS)
    assert limits.paper_trading_enabled is False
    raw = DEFAULT_LIMITS.read_text(encoding="utf-8")
    assert "live_trading_enabled: false" in raw
    assert "paper_trading_enabled: false" in raw


def test_live_trading_flag_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "limits.yaml"
    path.write_text(
        "\n".join(
            [
                "version: risk-limits-bad",
                "paper_trading_enabled: false",
                "live_trading_enabled: true",
                "max_gross_leverage: 1.0",
                "max_sleeve_notional_usd: 1000000",
                "max_drawdown_research: 0.35",
                "min_holdout_net_sharpe: 0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="live trading"):
        RiskLimits.load(path)


def test_passing_validation_is_still_vetoed(stores: Stores) -> None:
    clock = FrozenClock(NOW)
    pkg = StrategyPackage(
        id="pkg_veto_1",
        hypothesis_id="hyp_veto",
        name="veto-test",
        source_code="class S: pass",
        entrypoint="S",
        required_columns=("close",),
        content_hash="sha256:" + ("b" * 64),
        cost_model=CostModel(
            version="cm-binance-usdm-v1",
            taker_fee_bps=4.0,
            maker_fee_bps=2.0,
            half_spread_bps=1.0,
            impact_bps_per_million=0.5,
            capacity_notional_usd=1_000_000.0,
        ),
        created_by=AgentRole.STRATEGY_ENGINEERING,
        created_at=NOW,
    )
    stores.documents.put(pkg, recorded_by=AgentRole.STRATEGY_ENGINEERING.value)
    summary = RedactedValidationSummary(
        id="vsm_veto_pass",
        report_id="vrp_veto_pass",
        strategy_package_id=pkg.id,
        protocol_version="0.1.0-slice",
        overall=GateOutcome.PASS,
        gate_outcomes={"G01": GateOutcome.PASS},
        failed_gate_names=(),
    )
    stores.documents.put(summary, recorded_by="validation_service")

    result = RiskService(stores, RiskLimits.load(DEFAULT_LIMITS), clock).evaluate(
        pkg.id, summary.id
    )
    assert result["decision"] == "veto"
    assert result["reason_codes"] == ["paper_trading_disabled"]
    assert result["checks"]["validation_passed"] is True
    assert result["checks"]["paper_trading_enabled"] is False

    with pytest.raises(ValueError, match="CLEARANCE"):
        acknowledge_risk(
            stores,
            clock,
            risk_review_id=result["review_id"],
            acknowledged_by="operator@enigma",
            note="must not open paper",
        )
    forged = HumanRiskAcknowledge(
        id="hrack_forged",
        risk_review_id=result["review_id"],
        strategy_package_id=pkg.id,
        acknowledged_by="operator@enigma",
        note="bypasses the acknowledge helper",
        created_at=NOW,
    )
    stores.documents.put(forged, recorded_by=SystemActor.HUMAN.value)
    with pytest.raises(ValueError, match="CLEARANCE"):
        PaperBroker(stores, clock).open_session(
            package_id=pkg.id,
            risk_review_id=result["review_id"],
            acknowledge_id=forged.id,
            notional_usd=10_000.0,
        )
    assert stores.documents.count(PaperSession) == 0
