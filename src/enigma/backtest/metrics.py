"""Performance metrics over per-bar portfolio series (returns as fractions of notional)."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from enigma.backtest.statistics import sharpe_ratio


def max_drawdown(net: np.ndarray) -> float:
    """Largest peak-to-trough fall of compounded equity, starting from 1.0 (positive fraction)."""
    if net.size == 0:
        return 0.0
    equity = np.concatenate([[1.0], np.cumprod(1.0 + net)])
    peaks = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peaks))


def compute_metrics(series: Mapping[str, np.ndarray], periods_per_year: float) -> dict[str, float]:
    net = series["net"]
    n = int(net.size)
    total = float(np.prod(1.0 + net) - 1.0) if n else 0.0
    if n and 1.0 + total > 0:
        annual = float((1.0 + total) ** (periods_per_year / n) - 1.0)
    else:
        annual = -1.0 if n else 0.0
    exposed = series["exposure"] > 0
    std = float(np.std(net, ddof=1)) if n > 1 else 0.0
    return {
        "n_bars": float(n),
        "net_sharpe": sharpe_ratio(net, periods_per_year),
        "gross_sharpe": sharpe_ratio(series["gross"], periods_per_year),
        "total_return": total,
        "annual_return": annual,
        "annual_volatility": std * math.sqrt(periods_per_year),
        "max_drawdown": max_drawdown(net),
        "turnover": float(np.sum(series["turnover"])),
        "annual_turnover": float(np.mean(series["turnover"]) * periods_per_year) if n else 0.0,
        "trades": float(np.sum(series["trades"])),
        "price_pnl": float(np.sum(series["price"])),
        "funding_pnl": float(np.sum(series["funding"])),
        "cost_pnl": -float(np.sum(series["cost"])),
        "gross_pnl": float(np.sum(series["gross"])),
        "net_pnl": float(np.sum(net)),
        "hit_rate": float(np.mean(net[exposed] > 0)) if np.any(exposed) else 0.0,
        "exposure": float(np.mean(series["exposure"])) if n else 0.0,
    }
