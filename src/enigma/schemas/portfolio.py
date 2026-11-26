"""Research allocation. Executable weights require an L3 claim and still are not an order."""

from __future__ import annotations

from pydantic import model_validator

from enigma.schemas.common import ClaimLevel, NonEmptyStr, StrictModel, UtcDatetime


class PortfolioProposal(StrictModel):
    id: NonEmptyStr
    strategy_ids: tuple[str, ...]
    weights: tuple[float, ...]
    claim_level: ClaimLevel
    executable: bool
    reason: NonEmptyStr
    created_at: UtcDatetime

    @model_validator(mode="after")
    def _gates(self) -> PortfolioProposal:
        if len(self.strategy_ids) != len(self.weights):
            raise ValueError("each strategy needs one weight")
        if any(weight < 0 for weight in self.weights):
            raise ValueError("weights are long-only")
        if self.weights and abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError("weights must sum to 1")
        if self.executable and self.claim_level is not ClaimLevel.L3_INDEPENDENTLY_VALIDATED:
            raise ValueError("executable allocation requires an L3 claim")
        return self
