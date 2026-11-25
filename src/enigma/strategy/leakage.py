"""Look-ahead detection for bar strategies.

Truncation invariance: a strategy that uses only rows 0..t to decide position t must produce
the same positions[0..t] when it is given only rows 0..t. Any difference at a sampled checkpoint
is look-ahead (or hidden non-determinism); both are disqualifying.

Timestamp check: every row a decision can see must be known by the decision time. Rows are
ordered by `ts`; row t is known at `available_at[t]`, which must be after the bar open,
non-decreasing, and not before any funding event it reports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from enigma.schemas.research import StrategyPackage
from enigma.strategy.interface import ParamValue
from enigma.strategy.sandbox import SandboxJob, SandboxResult, SandboxRunner

TRUNCATION_ATOL = 1e-12


@dataclass(frozen=True)
class TruncationMismatch:
    job_id: str
    checkpoint: int
    first_bad_index: int
    max_abs_diff: float


@dataclass(frozen=True)
class TruncationCheckResult:
    checkpoints: dict[str, tuple[int, ...]]
    mismatches: tuple[TruncationMismatch, ...]

    @property
    def passed(self) -> bool:
        return not self.mismatches

    @property
    def checks(self) -> int:
        return sum(len(c) for c in self.checkpoints.values())


def sample_checkpoints(n_bars: int, k: int, seed: int, min_index: int = 1) -> tuple[int, ...]:
    """k distinct bar indices in [min_index, n_bars - 2], always including n_bars - 2."""
    hi = n_bars - 2
    if hi < min_index:
        return ()
    population = np.arange(min_index, hi + 1)
    rng = np.random.default_rng(seed)
    k_eff = min(k, population.size)
    picked = set(rng.choice(population, size=k_eff, replace=False).tolist())
    picked.add(hi)
    return tuple(sorted(int(i) for i in picked))


def truncation_jobs(
    job_id: str, dataset: str, checkpoints: Sequence[int], params: Mapping[str, ParamValue] | None
) -> SandboxJob:
    return SandboxJob(
        job_id=job_id,
        dataset=dataset,
        params=None if params is None else dict(params),
        truncations=tuple(int(t) + 1 for t in checkpoints),
    )


def evaluate_truncations(
    result: SandboxResult, checkpoints: Mapping[str, Sequence[int]]
) -> TruncationCheckResult:
    """Compare truncated re-runs in `result` against its full-length positions."""
    result.require_ok()
    mismatches: list[TruncationMismatch] = []
    for job_id, points in sorted(checkpoints.items()):
        full = result.positions[job_id]
        parts = result.truncated.get(job_id, {})
        for t in points:
            part = parts.get(t + 1)
            if part is None:
                mismatches.append(TruncationMismatch(job_id, t, -1, float("inf")))
                continue
            diff = np.abs(part - full[: t + 1])
            bad = np.flatnonzero(diff > TRUNCATION_ATOL)
            if bad.size:
                mismatches.append(TruncationMismatch(job_id, t, int(bad[0]), float(diff.max())))
    return TruncationCheckResult(
        checkpoints={k: tuple(int(t) for t in v) for k, v in sorted(checkpoints.items())},
        mismatches=tuple(mismatches),
    )


def truncation_invariance_check(
    package: StrategyPackage,
    data: Mapping[str, Mapping[str, np.ndarray]],
    checkpoints: Mapping[str, Sequence[int]],
    *,
    runner: SandboxRunner,
) -> TruncationCheckResult:
    """Run `package` on each dataset in `data` and on its truncations, in one sandbox batch."""
    jobs = [truncation_jobs(name, name, checkpoints.get(name, ()), None) for name in sorted(data)]
    result = runner.run(package, data, jobs).require_ok()
    return evaluate_truncations(result, {name: checkpoints.get(name, ()) for name in data})


def pit_timestamp_violations(frame: pl.DataFrame) -> dict[str, int]:
    """Count rows that could leak information to a decision taken at `available_at[t]`."""
    counts = {
        "available_not_after_open": 0,
        "available_decreasing": 0,
        "ts_not_increasing": 0,
        "funding_after_available": 0,
    }
    for symbol in sorted(frame.get_column("symbol").unique().to_list()):
        sub = frame.filter(pl.col("symbol") == symbol).sort("ts")
        ts = sub.get_column("ts").dt.epoch("ns").to_numpy()
        avail = sub.get_column("available_at").dt.epoch("ns").to_numpy()
        counts["available_not_after_open"] += int(np.sum(avail <= ts))
        counts["available_decreasing"] += int(np.sum(np.diff(avail) < 0))
        counts["ts_not_increasing"] += int(np.sum(np.diff(ts) <= 0))
        if "funding_time" in sub.columns:
            funding = sub.select(
                (
                    pl.col("funding_time").is_not_null()
                    & (pl.col("funding_time") > pl.col("available_at"))
                ).sum()
            ).item()
            counts["funding_after_available"] += int(funding)
    return counts
