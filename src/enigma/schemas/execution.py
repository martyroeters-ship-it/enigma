"""Paper-execution contracts (M2). Live trading keys stay absent (ADR-010)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from enigma.schemas.common import Actor, NonEmptyStr, StrictModel, UtcDatetime


class PaperOrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class PaperOrderStatus(StrEnum):
    ACCEPTED = "accepted"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class PaperOrder(StrictModel):
    id: NonEmptyStr
    session_id: NonEmptyStr
    strategy_package_id: NonEmptyStr
    symbol: NonEmptyStr
    side: PaperOrderSide
    quantity: float = Field(gt=0)
    limit_price: float | None = None
    status: PaperOrderStatus
    reject_reason: str | None = None
    created_by: Actor
    created_at: UtcDatetime


class PaperFill(StrictModel):
    id: NonEmptyStr
    order_id: NonEmptyStr
    session_id: NonEmptyStr
    strategy_package_id: NonEmptyStr
    symbol: NonEmptyStr
    side: PaperOrderSide
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    fee_usd: float = Field(ge=0)
    slippage_bps: float = Field(ge=0)
    bar_ts: UtcDatetime
    created_at: UtcDatetime


class PaperSessionStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"


class PaperSession(StrictModel):
    id: NonEmptyStr
    strategy_package_id: NonEmptyStr
    risk_review_id: NonEmptyStr
    status: PaperSessionStatus
    notional_usd: float = Field(gt=0)
    starting_equity_usd: float = Field(gt=0)
    equity_usd: float = Field(gt=0)
    realized_pnl_usd: float = 0.0
    fill_count: int = Field(ge=0, default=0)
    created_by: Actor
    created_at: UtcDatetime
    updated_at: UtcDatetime


class DegradationKind(StrEnum):
    SHARPE_DROP = "sharpe_drop"
    DRAWDOWN = "drawdown"
    FILL_COST_SPIKE = "fill_cost_spike"


class DegradationAlert(StrictModel):
    id: NonEmptyStr
    session_id: NonEmptyStr
    strategy_package_id: NonEmptyStr
    kind: DegradationKind
    severity: float = Field(ge=0)
    message: NonEmptyStr
    metrics: dict[str, float]
    created_at: UtcDatetime


class HumanRiskAcknowledge(StrictModel):
    """Human gate before paper capital may be allocated (IMPLEMENTATION_PLAN M2)."""

    id: NonEmptyStr
    risk_review_id: NonEmptyStr
    strategy_package_id: NonEmptyStr
    acknowledged_by: NonEmptyStr
    note: NonEmptyStr
    created_at: UtcDatetime
