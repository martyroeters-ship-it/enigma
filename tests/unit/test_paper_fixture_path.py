"""Labeled fixture: risk clearance, human ack, paper fills, and a degradation pause.

The PASS summary is not a ValidationService report and is not evidence of an edge.
Production limits stay paper-off. A FAIL summary is not paper-eligible.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from enigma.core.clock import FrozenClock
from enigma.execution import DegradationDetector, PaperBroker, PaperQuote
from enigma.risk.acknowledge import acknowledge_risk
from enigma.risk.service import RiskLimits, RiskService
from enigma.schemas.common import AgentRole
from enigma.schemas.execution import PaperOrderSide, PaperSessionStatus
from enigma.schemas.research import CostModel, StrategyPackage
from enigma.schemas.validation import GateOutcome, RedactedValidationSummary
from enigma.store import Stores

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_LIMITS = ROOT / "registries" / "risk" / "limits.yaml"
NOW = datetime(2026, 3, 1, tzinfo=UTC)


def _limits(tmp_path: Path, *, paper: bool) -> RiskLimits:
    path = tmp_path / "limits-fixture.yaml"
    path.write_text(
        "\n".join(
            [
                "version: risk-limits-fixture",
                f"paper_trading_enabled: {'true' if paper else 'false'}",
                "live_trading_enabled: false",
                "max_gross_leverage: 1.0",
                "max_sleeve_notional_usd: 1000000",
                "max_drawdown_research: 0.35",
                "min_holdout_net_sharpe: 0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return RiskLimits.load(path)


def _package(stores: Stores) -> StrategyPackage:
    pkg = StrategyPackage(
        id="pkg_fixture_paper",
        hypothesis_id="hyp_fixture",
        name="labeled-fixture",
        source_code="class S: pass",
        entrypoint="S",
        required_columns=("close",),
        content_hash="sha256:" + ("c" * 64),
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
    return pkg


def _summary(stores: Stores, package_id: str, overall: GateOutcome) -> RedactedValidationSummary:
    summary = RedactedValidationSummary(
        id="vsm_fixture_" + overall.value.lower(),
        report_id="vrp_fixture_" + overall.value.lower(),
        strategy_package_id=package_id,
        protocol_version="fixture",
        overall=overall,
        gate_outcomes={"fixture": overall},
        failed_gate_names=() if overall is GateOutcome.PASS else ("fixture",),
    )
    stores.documents.put(summary, recorded_by="fixture")
    return summary


def test_fixture_pass_clears_risk_and_degradation_pauses(stores: Stores, tmp_path: Path) -> None:
    assert "paper_trading_enabled: false" in PRODUCTION_LIMITS.read_text(encoding="utf-8")
    clock = FrozenClock(NOW)
    pkg = _package(stores)
    summary = _summary(stores, pkg.id, GateOutcome.PASS)
    review = RiskService(stores, _limits(tmp_path, paper=True), clock).evaluate(pkg.id, summary.id)
    assert review["decision"] == "clearance"
    assert review["reason_codes"] == []
    ack = acknowledge_risk(
        stores,
        clock,
        risk_review_id=review["review_id"],
        acknowledged_by="operator@enigma",
        note="labeled fixture only",
    )
    broker = PaperBroker(stores, clock)
    session = broker.open_session(
        package_id=pkg.id,
        risk_review_id=review["review_id"],
        acknowledge_id=ack.id,
        notional_usd=10_000.0,
    )
    _order, fill = broker.submit_market_order(
        session_id=session.id,
        quote=PaperQuote(symbol="BTCUSDT", mid=50_000.0, bar_ts=NOW),
        side=PaperOrderSide.BUY,
        quantity=0.01,
    )
    assert fill is not None and fill.fee_usd > 0
    alerts = DegradationDetector(
        stores, clock, broker, max_drawdown=0.05, min_fill_sharpe=-0.5
    ).evaluate(session.id, equity_path=[10_000.0, 9_000.0], fills=(), fill_pnls=())
    assert alerts and alerts[0].kind.value == "drawdown"
    paused = stores.documents.require(type(session), session.id)
    assert paused.status is PaperSessionStatus.PAUSED
    assert "paper_trading_enabled: false" in PRODUCTION_LIMITS.read_text(encoding="utf-8")


def test_fixture_fail_is_not_paper_eligible(stores: Stores, tmp_path: Path) -> None:
    clock = FrozenClock(NOW)
    pkg = _package(stores)
    summary = _summary(stores, pkg.id, GateOutcome.FAIL)
    review = RiskService(stores, _limits(tmp_path, paper=True), clock).evaluate(pkg.id, summary.id)
    assert review["decision"] == "veto"
    assert "validation_not_passed" in review["reason_codes"]
    with pytest.raises(ValueError, match="CLEARANCE"):
        acknowledge_risk(
            stores,
            clock,
            risk_review_id=review["review_id"],
            acknowledged_by="operator@enigma",
            note="must not open",
        )
