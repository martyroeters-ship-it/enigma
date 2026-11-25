"""Two model classes on one series. The holdout is not used to fit either class."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fit:
    model_class: str
    holdout_mse: float
    intercept: float
    slope: float


@dataclass(frozen=True)
class HybridComparison:
    winner: str
    fits: tuple[Fit, Fit]
    loser_failure: str


def compare_holdout(x: list[float], y: list[float], *, holdout: int) -> HybridComparison:
    if holdout < 1 or len(x) != len(y) or len(x) - holdout < 2:
        raise ValueError("need a train prefix of at least 2 and a holdout of at least 1")
    train_x, test_x = x[:-holdout], x[-holdout:]
    train_y, test_y = y[:-holdout], y[-holdout:]
    mean = sum(train_y) / len(train_y)
    constant = Fit("constant_mean", _mse([mean] * holdout, test_y), mean, 0.0)
    slope, intercept = _ols(train_x, train_y)
    predicted = [intercept + slope * item for item in test_x]
    linear = Fit("ols", _mse(predicted, test_y), intercept, slope)
    winner = linear if linear.holdout_mse <= constant.holdout_mse else constant
    loser = constant if winner is linear else linear
    return HybridComparison(
        winner=winner.model_class,
        fits=(constant, linear),
        loser_failure=(
            f"{loser.model_class} holdout mse {loser.holdout_mse:.6f} "
            f"lost to {winner.model_class} {winner.holdout_mse:.6f}"
        ),
    )


def _ols(x: list[float], y: list[float]) -> tuple[float, float]:
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    var_x = sum((item - mean_x) ** 2 for item in x)
    if var_x <= 1e-18:
        return 0.0, mean_y
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    slope = cov / var_x
    return slope, mean_y - slope * mean_x


def _mse(predicted: list[float], actual: list[float]) -> float:
    return sum((p - a) ** 2 for p, a in zip(predicted, actual, strict=True)) / len(actual)
