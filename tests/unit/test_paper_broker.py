"""M2 paper broker: clearance → human ack → session → fill → degradation pause."""

from __future__ import annotations

from datetime import UTC, datetime

from enigma.core.clock import FrozenClock
from enigma.execution import DegradationDetector, PaperBroker, PaperQuote
from enigma.risk.acknowledge import acknowledge_risk
from enigma.schemas.common import AgentRole, SystemActor
from enigma.schemas.cycle import RiskDecision, RiskReview
from enigma.schemas.execution import PaperOrderSide, PaperOrderStatus, PaperSessionStatus
from enigma.schemas.research import CostModel, StrategyPackage
from enigma.store import Stores

NOW = datetime(2026, 3, 1, tzinfo=UTC)


def _package(stores: Stores) -> StrategyPackage:
    pkg = StrategyPackage(
        id="pkg_paper_1",
        hypothesis_id="hyp_paper",
        name="paper-test",
        source_code="class S: pass",
        entrypoint="S",
        required_columns=("close",),
        content_hash="sha256:" + ("a" * 64),
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


def _clearance(stores: Stores, package_id: str, clock: FrozenClock) -> RiskReview:
    review = RiskReview(
        id="rsk_paper_1",
        strategy_package_id=package_id,
        decision=RiskDecision.CLEARANCE,
        reason_codes=(),
        checks={"validation_passed": True, "paper_trading_enabled": True},
        limits_version="risk-limits-test",
        created_by=SystemActor.RISK_SERVICE,
        created_at=clock.now(),
    )
    stores.documents.put(review, recorded_by=SystemActor.RISK_SERVICE.value)
    return review


def test_paper_session_requires_clearance_and_ack(stores: Stores) -> None:
    clock = FrozenClock(NOW)
    pkg = _package(stores)
    review = _clearance(stores, pkg.id, clock)
    ack = acknowledge_risk(
        stores,
        clock,
        risk_review_id=review.id,
        acknowledged_by="operator@enigma",
        note="approved for paper sleeve",
    )
    broker = PaperBroker(stores, clock)
    session = broker.open_session(
        package_id=pkg.id,
        risk_review_id=review.id,
        acknowledge_id=ack.id,
        notional_usd=10_000.0,
    )
    assert session.status is PaperSessionStatus.ACTIVE
    order, fill = broker.submit_market_order(
        session_id=session.id,
        quote=PaperQuote(symbol="BTCUSDT", mid=50_000.0, bar_ts=NOW),
        side=PaperOrderSide.BUY,
        quantity=0.01,
    )
    assert order.status is PaperOrderStatus.FILLED
    assert fill is not None
    assert fill.fee_usd > 0
    updated = stores.documents.require(type(session), session.id)
    assert updated.fill_count == 1
    assert updated.equity_usd < session.starting_equity_usd


def test_degradation_pauses_session(stores: Stores) -> None:
    clock = FrozenClock(NOW)
    pkg = _package(stores)
    review = _clearance(stores, pkg.id, clock)
    ack = acknowledge_risk(
        stores,
        clock,
        risk_review_id=review.id,
        acknowledged_by="operator@enigma",
        note="paper sleeve",
    )
    broker = PaperBroker(stores, clock)
    session = broker.open_session(
        package_id=pkg.id,
        risk_review_id=review.id,
        acknowledge_id=ack.id,
        notional_usd=10_000.0,
    )
    detector = DegradationDetector(stores, clock, broker, max_drawdown=0.05, min_fill_sharpe=-0.5)
    alerts = detector.evaluate(
        session.id,
        equity_path=[10_000.0, 9_800.0, 9_000.0],
        fills=(),
        fill_pnls=(),
    )
    assert alerts
    assert alerts[0].kind.value == "drawdown"
    paused = stores.documents.require(type(session), session.id)
    assert paused.status is PaperSessionStatus.PAUSED


def test_vetoed_review_cannot_open_paper(stores: Stores) -> None:
    clock = FrozenClock(NOW)
    pkg = _package(stores)
    review = RiskReview(
        id="rsk_veto",
        strategy_package_id=pkg.id,
        decision=RiskDecision.VETO,
        reason_codes=("paper_trading_disabled",),
        checks={"paper_trading_enabled": False},
        limits_version="risk-limits-m1-1",
        created_by=SystemActor.RISK_SERVICE,
        created_at=NOW,
    )
    stores.documents.put(review, recorded_by=SystemActor.RISK_SERVICE.value)
    try:
        acknowledge_risk(
            stores,
            clock,
            risk_review_id=review.id,
            acknowledged_by="operator",
            note="nope",
        )
        raised = False
    except ValueError:
        raised = True
    assert raised
