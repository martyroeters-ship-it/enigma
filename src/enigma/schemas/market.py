"""Stored market observations and the scan that ranks them. A scan is not an order."""

from __future__ import annotations

from enigma.schemas.common import NonEmptyStr, StrictModel, UtcDatetime


class MarketObservation(StrictModel):
    id: NonEmptyStr
    symbol: NonEmptyStr
    field: NonEmptyStr
    available_at: UtcDatetime
    value: float


class MarketScanReport(StrictModel):
    id: NonEmptyStr
    window_key: NonEmptyStr
    cutoff: UtcDatetime
    observation_count: int
    anomaly_count: int
    top_symbol: str = ""
    top_field: str = ""
    top_zscore: float | None = None
    suggested_mechanism: str = ""
    objective_id: str | None = None
    created_at: UtcDatetime
