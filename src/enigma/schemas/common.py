"""Shared enums and base models for all typed contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
UtcDatetime = AwareDatetime
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictModel(BaseModel):
    """Immutable, closed-schema base. Updates go through `model_copy(update=...)`."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class AgentRole(StrEnum):
    CIO = "cio"
    META_LEARNING = "meta_learning"
    INTELLIGENCE = "intelligence"
    DATA_ENGINEERING = "data_engineering"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    QUANT_RESEARCH = "quant_research"
    STRATEGY_ENGINEERING = "strategy_engineering"
    INDEPENDENT_VALIDATION = "independent_validation"
    PORTFOLIO_MANAGEMENT = "portfolio_management"
    RISK_MANAGEMENT = "risk_management"
    EXECUTION_MONITORING = "execution_monitoring"
    TECHNOLOGY_INNOVATION = "technology_innovation"


class SystemActor(StrEnum):
    HUMAN = "human"
    SYSTEM = "system"
    VALIDATION_SERVICE = "validation_service"
    RISK_SERVICE = "risk_service"


Actor = AgentRole | SystemActor


class Plane(StrEnum):
    CONTROL = "control"
    RESEARCH = "research"
    VALIDATION = "validation"
    EXECUTION = "execution"


class ClaimLevel(StrEnum):
    L0_UNVERIFIED = "L0"
    L1_HYPOTHESIS = "L1"
    L2_HISTORICAL = "L2"
    L3_INDEPENDENTLY_VALIDATED = "L3"


class Instrument(StrictModel):
    """Venue-qualified instrument, e.g. BTCUSDT perpetual on binance_usdm."""

    symbol: NonEmptyStr
    base: NonEmptyStr
    quote: NonEmptyStr
    kind: NonEmptyStr = Field(description="spot | perpetual")
    venue: NonEmptyStr
    asset_class: NonEmptyStr = "crypto"

    @property
    def key(self) -> str:
        return f"{self.venue}:{self.kind}:{self.symbol}"
