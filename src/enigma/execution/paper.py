"""Deterministic paper broker: fill at mid ± half-spread with fee schedule from CostModel.

No live keys, no network. Requires Risk CLEARANCE + HumanRiskAcknowledge before open.
"""

from __future__ import annotations

from dataclasses import dataclass

from enigma.core.clock import Clock
from enigma.core.hashing import sha256_hex
from enigma.schemas.common import AgentRole, UtcDatetime
from enigma.schemas.cycle import RiskDecision, RiskReview
from enigma.schemas.execution import (
    HumanRiskAcknowledge,
    PaperFill,
    PaperOrder,
    PaperOrderSide,
    PaperOrderStatus,
    PaperSession,
    PaperSessionStatus,
)
from enigma.schemas.research import StrategyPackage
from enigma.store import Stores

_ACTOR = AgentRole.EXECUTION_MONITORING


@dataclass(frozen=True)
class PaperQuote:
    symbol: str
    mid: float
    bar_ts: UtcDatetime


class PaperBroker:
    def __init__(self, stores: Stores, clock: Clock) -> None:
        self.stores = stores
        self.clock = clock

    def open_session(
        self,
        *,
        package_id: str,
        risk_review_id: str,
        acknowledge_id: str,
        notional_usd: float,
        actor: str | None = None,
    ) -> PaperSession:
        who = actor or _ACTOR.value
        docs = self.stores.documents
        package = docs.require(StrategyPackage, package_id)
        review = docs.require(RiskReview, risk_review_id)
        ack = docs.require(HumanRiskAcknowledge, acknowledge_id)
        if review.strategy_package_id != package_id:
            raise ValueError("risk review does not belong to this package")
        if ack.strategy_package_id != package_id or ack.risk_review_id != risk_review_id:
            raise ValueError("human acknowledge does not match risk review/package")
        if review.decision is not RiskDecision.CLEARANCE:
            raise ValueError("paper session requires risk CLEARANCE")
        now = self.clock.now()
        session = PaperSession(
            id="psess_" + sha256_hex(f"{package_id}|{risk_review_id}|{acknowledge_id}")[:26],
            strategy_package_id=package.id,
            risk_review_id=review.id,
            status=PaperSessionStatus.ACTIVE,
            notional_usd=notional_usd,
            starting_equity_usd=notional_usd,
            equity_usd=notional_usd,
            created_by=who,
            created_at=now,
            updated_at=now,
        )
        docs.put(session, recorded_by=who)
        self.stores.audit.record(
            who,
            "paper.session_open",
            session.id,
            {"package_id": package_id, "notional_usd": notional_usd},
        )
        return session

    def submit_market_order(
        self,
        *,
        session_id: str,
        quote: PaperQuote,
        side: PaperOrderSide,
        quantity: float,
        actor: str | None = None,
    ) -> tuple[PaperOrder, PaperFill | None]:
        who = actor or _ACTOR.value
        docs = self.stores.documents
        session = docs.require(PaperSession, session_id)
        if session.status is not PaperSessionStatus.ACTIVE:
            raise ValueError(f"session {session_id} is {session.status.value}")
        package = docs.require(StrategyPackage, session.strategy_package_id)
        now = self.clock.now()
        order_id = "pord_" + sha256_hex(f"{session_id}|{quote.symbol}|{side}|{quantity}|{now}")[:26]
        if quantity <= 0 or quote.mid <= 0:
            order = PaperOrder(
                id=order_id,
                session_id=session_id,
                strategy_package_id=session.strategy_package_id,
                symbol=quote.symbol,
                side=side,
                quantity=max(quantity, 1e-12),
                status=PaperOrderStatus.REJECTED,
                reject_reason="invalid_quantity_or_price",
                created_by=who,
                created_at=now,
            )
            docs.put(order, recorded_by=who)
            return order, None

        slip = package.cost_model.half_spread_bps / 10_000.0
        fee_bps = package.cost_model.taker_fee_bps / 10_000.0
        price = quote.mid * (1.0 + slip) if side is PaperOrderSide.BUY else quote.mid * (1.0 - slip)
        notional = price * quantity
        fee = notional * fee_bps
        order = PaperOrder(
            id=order_id,
            session_id=session_id,
            strategy_package_id=session.strategy_package_id,
            symbol=quote.symbol,
            side=side,
            quantity=quantity,
            status=PaperOrderStatus.FILLED,
            created_by=who,
            created_at=now,
        )
        fill = PaperFill(
            id="pfil_" + sha256_hex(order_id)[:26],
            order_id=order.id,
            session_id=session_id,
            strategy_package_id=session.strategy_package_id,
            symbol=quote.symbol,
            side=side,
            quantity=quantity,
            price=price,
            fee_usd=fee,
            slippage_bps=package.cost_model.half_spread_bps,
            bar_ts=quote.bar_ts,
            created_at=now,
        )
        signed = notional if side is PaperOrderSide.SELL else -notional
        equity = session.equity_usd + signed - fee
        updated = session.model_copy(
            update={
                "equity_usd": equity,
                "realized_pnl_usd": equity - session.starting_equity_usd,
                "fill_count": session.fill_count + 1,
                "updated_at": now,
            }
        )
        docs.put(order, recorded_by=who)
        docs.put(fill, recorded_by=who)
        docs.put(updated, recorded_by=who)
        self.stores.audit.record(
            who,
            "paper.fill",
            fill.id,
            {"order_id": order.id, "price": price, "fee_usd": fee, "equity_usd": equity},
        )
        return order, fill

    def pause(self, session_id: str, reason: str, actor: str | None = None) -> PaperSession:
        who = actor or _ACTOR.value
        docs = self.stores.documents
        session = docs.require(PaperSession, session_id)
        updated = session.model_copy(
            update={"status": PaperSessionStatus.PAUSED, "updated_at": self.clock.now()}
        )
        docs.put(updated, recorded_by=who)
        self.stores.audit.record(who, "paper.session_pause", session_id, {"reason": reason})
        return updated
