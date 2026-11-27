"""Sampling frequency, signal horizon, and holding period are different quantities.

Sampling frequency is the bar length of the dataset (1h, 8h, 1d). The signal horizon
is the lookback a strategy uses, counted in those bars. The holding period is how long
a position decision is labeled, also counted in sampling bars (`label_horizon_bars`).
A one-day holding period on one-hour bars is 24 bars, not a request to resample the data.
"""

from __future__ import annotations

from collections.abc import Sequence

SAMPLING_FREQUENCIES = ("1h", "8h", "1d")


def sampling_frequency(topics: Sequence[str], default: str = "8h") -> str:
    for topic in topics:
        if topic in SAMPLING_FREQUENCIES:
            return topic
    return default
