"""Funding and basis structure measurements on a research-plane panel (Agent 05)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

METHOD = "deterministic:funding-basis-stats-v1"


def _autocorr(x: np.ndarray, lag: int = 1) -> float:
    x = x[np.isfinite(x)]
    if x.size <= lag + 2:
        return float("nan")
    a, b = x[:-lag] - x[:-lag].mean(), x[lag:] - x[lag:].mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def _half_life(x: np.ndarray) -> float:
    """AR(1) half-life in bars: ln(0.5) / ln(phi); inf when not mean-reverting."""
    x = x[np.isfinite(x)]
    if x.size < 10:
        return float("nan")
    y, lagged = x[1:] - x[1:].mean(), x[:-1] - x[:-1].mean()
    denom = float((lagged * lagged).sum())
    if denom <= 0:
        return float("nan")
    phi = float((lagged * y).sum() / denom)
    if not 0 < phi < 1:
        return float("inf")
    return math.log(0.5) / math.log(phi)


def _extreme_share(f: np.ndarray, lookback: int = 90, z: float = 2.0) -> float:
    n = f.shape[0]
    if n <= lookback:
        return float("nan")
    hits = 0
    total = 0
    for t in range(lookback - 1, n):
        window = f[t - lookback + 1 : t + 1]
        sd = float(window.std())
        if sd > 0:
            total += 1
            if abs(float(f[t] - window.mean())) / sd > z:
                hits += 1
    return hits / total if total else float("nan")


def _finite(value: float) -> float:
    return value if math.isfinite(value) else -1.0


def analyze_panel(frame: pl.DataFrame, periods_per_year: int = 1095) -> dict[str, dict[str, float]]:
    """Per-symbol statistics. Non-finite results are reported as -1.0 with the count fields."""
    out: dict[str, dict[str, float]] = {}
    for symbol in sorted(frame["symbol"].unique().to_list()):
        sub = frame.filter(pl.col("symbol") == symbol).sort("ts")
        funding = sub["funding_rate"].fill_null(np.nan).to_numpy().astype(float)
        f_obs = funding[np.isfinite(funding)]
        close = sub["perp_close"].to_numpy().astype(float)
        returns = np.diff(np.log(close)) if close.size > 1 else np.array([])
        basis = sub["basis"].fill_null(np.nan).to_numpy().astype(float)
        out[symbol] = {
            "bars": float(sub.height),
            "funding_observations": float(f_obs.size),
            "funding_mean_bps": float(f_obs.mean() * 1e4) if f_obs.size else -1.0,
            "funding_std_bps": float(f_obs.std() * 1e4) if f_obs.size else -1.0,
            "funding_positive_share": float((f_obs > 0).mean()) if f_obs.size else -1.0,
            "funding_annualized_mean": float(f_obs.mean() * periods_per_year)
            if f_obs.size
            else -1.0,
            "funding_autocorr_lag1": _finite(_autocorr(f_obs)),
            "funding_extreme_share": _finite(_extreme_share(np.nan_to_num(funding))),
            "basis_mean_bps": float(np.nanmean(basis) * 1e4) if np.isfinite(basis).any() else -1.0,
            "basis_std_bps": float(np.nanstd(basis) * 1e4) if np.isfinite(basis).any() else -1.0,
            "basis_half_life_bars": _finite(_half_life(basis)),
            "return_autocorr_lag1": _finite(_autocorr(returns)),
            "annualized_volatility": float(returns.std() * math.sqrt(periods_per_year))
            if returns.size
            else -1.0,
        }
    return out


def findings(stats: dict[str, dict[str, float]]) -> tuple[str, ...]:
    lines: list[str] = []
    for symbol, s in stats.items():
        lines.append(
            f"{symbol}: funding mean {s['funding_mean_bps']:.2f} bps/8h "
            f"(~{s['funding_annualized_mean']:.1%}/yr), positive in "
            f"{s['funding_positive_share']:.0%} of {int(s['funding_observations'])} events, "
            f"lag-1 autocorrelation {s['funding_autocorr_lag1']:.2f}."
        )
        lines.append(
            f"{symbol}: basis mean {s['basis_mean_bps']:.1f} bps, std {s['basis_std_bps']:.1f} "
            f"bps, AR(1) half-life {s['basis_half_life_bars']:.1f} bars; perp return lag-1 "
            f"autocorrelation {s['return_autocorr_lag1']:.3f}; annualized vol "
            f"{s['annualized_volatility']:.0%}."
        )
    return tuple(lines)
