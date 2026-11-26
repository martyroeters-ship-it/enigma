"""Measured lift required before an improvement can be rolled out."""

from __future__ import annotations


def measured_lift(baseline: float, candidate: float, *, higher_is_better: bool = True) -> float:
    return candidate - baseline if higher_is_better else baseline - candidate


def may_roll_out(lift: float, *, postmortem_ids: tuple[str, ...], benchmark_id: str | None) -> bool:
    return lift > 0 and bool(postmortem_ids) and bool(benchmark_id)
