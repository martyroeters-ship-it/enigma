"""Long-only research weights. An unvalidated sleeve cannot be marked executable."""

from __future__ import annotations

from datetime import datetime

from enigma.core.hashing import sha256_hex
from enigma.schemas.common import ClaimLevel
from enigma.schemas.portfolio import PortfolioProposal


def research_proposal(
    series: dict[str, list[float]],
    claim_levels: dict[str, ClaimLevel],
    *,
    now: datetime,
) -> PortfolioProposal:
    if not series:
        raise ValueError("portfolio research needs at least one return series")
    ids = tuple(sorted(series))
    missing = [name for name in ids if name not in claim_levels]
    if missing:
        raise ValueError(f"missing claim level for {missing[0]}")
    scales: list[float] = []
    for name in ids:
        values = series[name]
        if len(values) < 2:
            raise ValueError(f"{name} needs at least two returns")
        mean = sum(values) / len(values)
        var = sum((item - mean) ** 2 for item in values) / (len(values) - 1)
        scales.append(1.0 / var**0.5 if var > 1e-18 else 0.0)
    if sum(scales) <= 0:
        weights = tuple(1.0 / len(ids) for _ in ids)
    else:
        total = sum(scales)
        weights = tuple(scale / total for scale in scales)
    validated = all(claim_levels[name] is ClaimLevel.L3_INDEPENDENTLY_VALIDATED for name in ids)
    claim = ClaimLevel.L3_INDEPENDENTLY_VALIDATED if validated else ClaimLevel.L0_UNVERIFIED
    reason = "independently_validated" if validated else "unvalidated_strategy"
    key = "|".join(ids) + "|" + reason
    return PortfolioProposal(
        id="pfo_" + sha256_hex(key)[:26].upper(),
        strategy_ids=ids,
        weights=weights,
        claim_level=claim,
        executable=validated,
        reason=reason,
        created_at=now,
    )
