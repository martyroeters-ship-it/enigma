"""Association between public narrative and a later return. Shuffled times are the control."""

from __future__ import annotations

import random
from datetime import datetime

from enigma.schemas.narrative import NarrativeEvent


def association(
    events: tuple[NarrativeEvent, ...] | list[NarrativeEvent],
    bar_times: list[datetime],
    next_returns: list[float],
) -> float:
    """Correlate signed narrative intensity known at each bar with the next return."""
    if len(bar_times) != len(next_returns) or len(bar_times) < 3:
        raise ValueError("need at least three aligned bars and returns")
    counts = [_signed(events, bar) for bar in bar_times]
    return _pearson(counts, next_returns)


def _signed(events: tuple[NarrativeEvent, ...] | list[NarrativeEvent], bar: datetime) -> float:
    total = 0.0
    for event in events:
        if event.available_at > bar:
            continue
        title = event.title.lower()
        total += 1.0 if title.startswith("positive") else -1.0
    return total


def scramble(events: tuple[NarrativeEvent, ...], seed: int) -> tuple[NarrativeEvent, ...]:
    times = [event.available_at for event in events]
    random.Random(seed).shuffle(times)  # noqa: S311 - seeded control, not a secret
    return tuple(
        event.model_copy(update={"available_at": stamp, "event_time": stamp})
        for event, stamp in zip(events, times, strict=True)
    )


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((item - mean_x) ** 2 for item in xs)
    var_y = sum((item - mean_y) ** 2 for item in ys)
    if var_x <= 1e-18 or var_y <= 1e-18:
        return 0.0
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(xs, ys, strict=True))
    return float(cov / (var_x**0.5 * var_y**0.5))
