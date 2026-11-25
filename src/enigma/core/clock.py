"""Injectable clocks. Strategy and validation code must never read wall-clock time directly."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock:
    """Deterministic clock for tests and replay."""

    def __init__(self, start: datetime, step: timedelta = timedelta(0)) -> None:
        if start.tzinfo is None:
            raise ValueError("FrozenClock requires a timezone-aware datetime")
        self._current = start
        self._step = step

    def now(self) -> datetime:
        value = self._current
        self._current = self._current + self._step
        return value

    def advance(self, delta: timedelta) -> None:
        self._current = self._current + delta
