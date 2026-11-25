"""Read-only prediction-market snapshot. This is research evidence, not an order."""

from __future__ import annotations

from pydantic import Field

from enigma.schemas.common import ClaimLevel, NonEmptyStr, Probability, StrictModel, UtcDatetime


class PredictionMarketSnapshot(StrictModel):
    id: NonEmptyStr
    market_id: NonEmptyStr
    question: NonEmptyStr
    probability: Probability
    volume_usd: float = Field(ge=0)
    available_at: UtcDatetime
    source: NonEmptyStr
    claim_level: ClaimLevel = ClaimLevel.L0_UNVERIFIED

    def research_note(self) -> str:
        when = self.available_at.isoformat()
        return (
            f"L0 prior {self.probability:.3f} from {self.source} at {when}. "
            "Not a validation result and not an order."
        )
