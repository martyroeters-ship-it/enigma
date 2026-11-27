"""Point-in-time anomaly scan.

Rows with `available_at` after the cutoff are ignored. The result is a ranking,
not an order and not a validation result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

_MECHANISM_BY_FIELD = {
    "funding_rate": "funding_crowding_reversal",
    "basis": "basis_reversion",
    "basis_bps": "basis_reversion",
}


@dataclass(frozen=True)
class Anomaly:
    symbol: str
    field: str
    index: int
    available_at: datetime
    value: float
    zscore: float
    sample_size: int
    suggested_mechanism: str


def scan_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    field: str,
    cutoff: datetime,
    window: int = 20,
    threshold: float = 2.5,
    min_samples: int = 8,
) -> tuple[Anomaly, ...]:
    """Rolling z-score on values known at or before `cutoff`, strongest first."""
    if window < 2 or min_samples < 2 or min_samples > window:
        raise ValueError("window must cover min_samples and both must be at least 2")
    usable = [
        row
        for row in rows
        if row.get("available_at") is not None
        and row["available_at"] <= cutoff
        and row.get(field) is not None
    ]
    usable.sort(key=lambda row: (row["available_at"], row.get("symbol", symbol)))
    # One value per timestamp. Repeated copies of the same print shrink the
    # variance and make every symbol that shares those copies look like the
    # same extreme z-score.
    deduped: list[Mapping[str, Any]] = []
    seen: dict[object, float] = {}
    for row in usable:
        stamp = row["available_at"]
        value = float(row[field])
        prior = seen.get(stamp)
        if prior is not None and abs(prior - value) > 1e-12:
            return ()
        if prior is None:
            seen[stamp] = value
            deduped.append(row)
    usable = deduped
    values = [float(row[field]) for row in usable]
    findings: list[Anomaly] = []
    for index, value in enumerate(values):
        sample = values[max(0, index - window + 1) : index + 1]
        if len(sample) < min_samples:
            continue
        mean = sum(sample) / len(sample)
        var = sum((item - mean) ** 2 for item in sample) / (len(sample) - 1)
        if var <= 1e-18:
            continue
        zscore = (value - mean) / var**0.5
        if abs(zscore) < threshold:
            continue
        findings.append(
            Anomaly(
                symbol=symbol,
                field=field,
                index=index,
                available_at=usable[index]["available_at"],
                value=value,
                zscore=zscore,
                sample_size=len(sample),
                suggested_mechanism=_MECHANISM_BY_FIELD.get(field, ""),
            )
        )
    findings.sort(key=lambda item: (-abs(item.zscore), item.available_at.isoformat(), item.field))
    return tuple(findings)
