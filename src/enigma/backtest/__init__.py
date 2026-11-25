"""Deterministic bar backtesting (ADR-012) and the statistics used by validation gates."""

from enigma.backtest.engine import (
    DEFAULT_PERIODS_PER_YEAR,
    BacktestEngine,
    BacktestResult,
    BarBacktestEngine,
    CapacityExceededError,
    SymbolSeries,
)
from enigma.backtest.metrics import compute_metrics, max_drawdown
from enigma.backtest.statistics import (
    DeflatedSharpe,
    PBOResult,
    cscv_pbo,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    holm_adjusted_pvalues,
    holm_bonferroni,
    one_sided_ttest,
    probabilistic_sharpe_ratio,
    psr_from_moments,
    sharpe_ratio,
    stationary_bootstrap_ci,
    stationary_bootstrap_indices,
    top_k_days_removed_pnl,
)

__all__ = [
    "DEFAULT_PERIODS_PER_YEAR",
    "BacktestEngine",
    "BacktestResult",
    "BarBacktestEngine",
    "CapacityExceededError",
    "DeflatedSharpe",
    "PBOResult",
    "SymbolSeries",
    "compute_metrics",
    "cscv_pbo",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "holm_adjusted_pvalues",
    "holm_bonferroni",
    "max_drawdown",
    "one_sided_ttest",
    "probabilistic_sharpe_ratio",
    "psr_from_moments",
    "sharpe_ratio",
    "stationary_bootstrap_ci",
    "stationary_bootstrap_indices",
    "top_k_days_removed_pnl",
]
