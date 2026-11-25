"""Deterministic pre-cutoff statistics for the research context."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from enigma.core.ids import new_id
from enigma.schemas.quant import StatObservation


def compute_observations(
    rows: list[dict[str, Any]],
    *,
    symbol: str,
    data_end: datetime,
    value_key: str = "funding_rate",
    min_samples: int = 8,
) -> tuple[StatObservation, ...]:
    """Compute simple mean/t-stat style observations from research-window rows.

    `rows` must already be filtered to `available_at <= research_cutoff`. This helper never
    looks past `data_end`.
    """
    values = [float(r[value_key]) for r in rows if value_key in r and r[value_key] is not None]
    if len(values) < min_samples:
        return ()
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
    std = var**0.5
    t_stat = (mean / (std / (len(values) ** 0.5))) if std > 0 else None
    return (
        StatObservation(
            id=new_id("obs"),
            symbol=symbol,
            name=f"mean_{value_key}",
            definition=f"sample mean of {value_key} over research window ending at data_end",
            value=mean,
            sample_size=len(values),
            t_stat=t_stat,
            data_end=data_end,
        ),
    )
