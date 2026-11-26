"""Proper scores for probabilistic forecasts. Validation PSR/DSR stay in the gate suite."""

from __future__ import annotations

import math
from dataclasses import dataclass

_EPS = 1e-15


def _pairs(probabilities: list[float], outcomes: list[bool]) -> list[tuple[float, float]]:
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError("probabilities and outcomes must be the same non-empty length")
    rows: list[tuple[float, float]] = []
    for probability, outcome in zip(probabilities, outcomes, strict=True):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability outside [0, 1]")
        rows.append((probability, 1.0 if outcome else 0.0))
    return rows


def brier_score(probabilities: list[float], outcomes: list[bool]) -> float:
    rows = _pairs(probabilities, outcomes)
    return sum((probability - outcome) ** 2 for probability, outcome in rows) / len(rows)


def log_loss(probabilities: list[float], outcomes: list[bool]) -> float:
    rows = _pairs(probabilities, outcomes)
    total = 0.0
    for probability, outcome in rows:
        clipped = min(max(probability, _EPS), 1.0 - _EPS)
        total += -(outcome * math.log(clipped) + (1.0 - outcome) * math.log(1.0 - clipped))
    return total / len(rows)


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_probability: float
    outcome_rate: float


def reliability_bins(
    probabilities: list[float], outcomes: list[bool], *, n_bins: int = 10
) -> tuple[ReliabilityBin, ...]:
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    rows = _pairs(probabilities, outcomes)
    width = 1.0 / n_bins
    grouped: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for probability, outcome in rows:
        index = min(int(probability / width), n_bins - 1)
        grouped[index].append((probability, outcome))
    bins: list[ReliabilityBin] = []
    for index, group in enumerate(grouped):
        if not group:
            continue
        bins.append(
            ReliabilityBin(
                lower=index * width,
                upper=(index + 1) * width,
                count=len(group),
                mean_probability=sum(p for p, _ in group) / len(group),
                outcome_rate=sum(y for _, y in group) / len(group),
            )
        )
    return tuple(bins)


def expected_calibration_error(
    probabilities: list[float], outcomes: list[bool], *, n_bins: int = 10
) -> float:
    rows = _pairs(probabilities, outcomes)
    total = 0.0
    for bin_ in reliability_bins(probabilities, outcomes, n_bins=n_bins):
        total += bin_.count * abs(bin_.mean_probability - bin_.outcome_rate)
    return total / len(rows)


def mean_overconfidence(probabilities: list[float], outcomes: list[bool]) -> float:
    """Positive when issued probabilities sit above the realized rate."""
    rows = _pairs(probabilities, outcomes)
    return sum(probability - outcome for probability, outcome in rows) / len(rows)


def is_overconfident(
    probabilities: list[float], outcomes: list[bool], *, min_gap: float = 0.05
) -> bool:
    return mean_overconfidence(probabilities, outcomes) > min_gap


def calibration_panel(probabilities: list[float], outcomes: list[bool]) -> dict[str, object]:
    """Scores and equal-width bins for resolved forecasts. Empty input is an empty panel."""
    if not probabilities:
        return {
            "n": 0,
            "brier": None,
            "log_loss": None,
            "ece": None,
            "overconfident": None,
            "bins": [],
        }
    bins = [
        {
            "lower": round(bin_.lower, 4),
            "upper": round(bin_.upper, 4),
            "count": bin_.count,
            "mean_probability": round(bin_.mean_probability, 4),
            "outcome_rate": round(bin_.outcome_rate, 4),
        }
        for bin_ in reliability_bins(probabilities, outcomes)
    ]
    return {
        "n": len(probabilities),
        "brier": round(brier_score(probabilities, outcomes), 6),
        "log_loss": round(log_loss(probabilities, outcomes), 6),
        "ece": round(expected_calibration_error(probabilities, outcomes), 6),
        "overconfident": is_overconfident(probabilities, outcomes),
        "bins": bins,
    }
