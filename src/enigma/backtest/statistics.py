"""Deterministic statistics for validation: Sharpe inference, multiple testing, overfitting.

Sharpe ratios in PSR/DSR are per-period (not annualized); returns are per-bar net returns.

Probabilistic Sharpe ratio (Bailey & Lopez de Prado, 2012):
    PSR(SR*) = Phi( (SR - SR*) * sqrt(T - 1) / sqrt(1 - g3 * SR + (g4 - 1) / 4 * SR^2) )
with SR the sample Sharpe, T the number of returns, g3 the sample skewness and g4 the sample
(non-excess) kurtosis, both population moments. If the variance term is not positive the
approximation is invalid and PSR is reported as 0 (fail closed).

Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014): DSR = PSR(SR0), where SR0 is the
expected maximum Sharpe among N independent trials with zero true Sharpe and cross-trial
Sharpe variance V:
    SR0 = sqrt(V) * ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e)))
with gamma the Euler-Mascheroni constant. For N = 1, SR0 = 0 (DSR reduces to PSR(0)).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329
NS_PER_DAY = 86_400 * 1_000_000_000


def sharpe_ratio(returns: np.ndarray, periods_per_year: float = 1.0) -> float:
    """mean / sample std * sqrt(periods_per_year); 0 when undefined (n < 2 or zero variance)."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    std = float(np.std(r, ddof=1))
    if std <= 0 or not math.isfinite(std):
        return 0.0
    return float(np.mean(r) / std * math.sqrt(periods_per_year))


def _moments(returns: np.ndarray) -> tuple[float, float]:
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 3 or float(np.std(r)) == 0.0:
        return 0.0, 3.0
    skew = float(stats.skew(r, bias=True))
    kurt = float(stats.kurtosis(r, fisher=False, bias=True))
    return skew, kurt


def psr_from_moments(sr: float, sr_benchmark: float, n: int, skew: float, kurtosis: float) -> float:
    if n < 2:
        return 0.0
    var_term = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr
    if var_term <= 0 or not math.isfinite(var_term):
        return 0.0
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(var_term)
    return float(stats.norm.cdf(z))


def probabilistic_sharpe_ratio(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    r = np.asarray(returns, dtype=np.float64)
    skew, kurt = _moments(r)
    return psr_from_moments(sharpe_ratio(r), sr_benchmark, int(r.size), skew, kurt)


def expected_max_sharpe(n_trials: int, trial_sharpe_variance: float) -> float:
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1 or trial_sharpe_variance <= 0:
        return 0.0
    a = float(stats.norm.ppf(1.0 - 1.0 / n_trials))
    b = float(stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return math.sqrt(trial_sharpe_variance) * ((1.0 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b)


@dataclass(frozen=True)
class DeflatedSharpe:
    dsr: float
    sharpe: float
    sr0: float
    n_obs: int
    skew: float
    kurtosis: float
    n_trials: int
    trial_sharpe_variance: float


def deflated_sharpe_ratio(
    returns: np.ndarray, n_trials: int, trial_sharpe_variance: float
) -> DeflatedSharpe:
    r = np.asarray(returns, dtype=np.float64)
    sr = sharpe_ratio(r)
    skew, kurt = _moments(r)
    sr0 = expected_max_sharpe(n_trials, trial_sharpe_variance)
    return DeflatedSharpe(
        dsr=psr_from_moments(sr, sr0, int(r.size), skew, kurt),
        sharpe=sr,
        sr0=sr0,
        n_obs=int(r.size),
        skew=skew,
        kurtosis=kurt,
        n_trials=n_trials,
        trial_sharpe_variance=trial_sharpe_variance,
    )


def one_sided_ttest(returns: np.ndarray) -> tuple[float, float]:
    """H0: mean <= 0 vs H1: mean > 0. Returns (t, p); p = 1 when undefined."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0, 1.0
    std = float(np.std(r, ddof=1))
    if std <= 0:
        mean = float(np.mean(r))
        return (math.inf, 0.0) if mean > 0 else (0.0, 1.0)
    t = float(np.mean(r) / (std / math.sqrt(r.size)))
    return t, float(stats.t.sf(t, df=r.size - 1))


def holm_adjusted_pvalues(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def holm_bonferroni(pvalues: Sequence[float], alpha: float) -> tuple[bool, ...]:
    return tuple(bool(x <= alpha) for x in holm_adjusted_pvalues(pvalues))


def stationary_bootstrap_indices(
    n: int, n_boot: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis & Romano (1994) resampling indices, shape (n_boot, n), circular blocks."""
    if n < 1 or mean_block < 1:
        raise ValueError("n >= 1 and mean_block >= 1 required")
    p_new = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    starts = rng.integers(0, n, size=(n_boot, n))
    new_block = rng.random((n_boot, n)) < p_new
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def stationary_bootstrap_ci(
    returns: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int,
    mean_block: float,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """Percentile (alpha/2, 1 - alpha/2) interval of `statistic` under stationary bootstrap."""
    r = np.asarray(returns, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(r.size, n_boot, mean_block, rng)
    draws = np.array([statistic(r[row]) for row in idx])
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    n_combinations: int
    n_strategies: int
    n_splits: int
    logits: tuple[float, ...]


def cscv_pbo(returns_matrix: np.ndarray, n_splits: int = 16) -> PBOResult:
    """Probability of backtest overfitting via CSCV (Bailey, Borwein, Lopez de Prado, Zhu 2017).

    `returns_matrix` is (T bars, N strategies). Rows are cut into `n_splits` contiguous blocks;
    for every half/half combination, the strategy with the best in-sample Sharpe is ranked
    out-of-sample. PBO = share of combinations where its OOS relative rank w = rank/(N+1)
    has logit(w) <= 0 (i.e. at or below the OOS median).
    """
    m = np.asarray(returns_matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("need a (T, N>=2) matrix")
    if n_splits < 2 or n_splits % 2:
        raise ValueError("n_splits must be an even number >= 2")
    t_rows = (m.shape[0] // n_splits) * n_splits
    if t_rows < n_splits * 2:
        raise ValueError("not enough rows for the requested splits")
    blocks = m[:t_rows].reshape(n_splits, -1, m.shape[1])
    count = np.full(n_splits, blocks.shape[1], dtype=np.float64)
    sums = blocks.sum(axis=1)
    sumsq = (blocks**2).sum(axis=1)
    n_strat = m.shape[1]

    def _sharpe(sel: list[int]) -> np.ndarray:
        n = count[sel].sum()
        mean = sums[sel].sum(axis=0) / n
        var = (sumsq[sel].sum(axis=0) - n * mean**2) / (n - 1)
        std = np.sqrt(np.maximum(var, 0.0))
        return np.where(std > 0, mean / np.where(std > 0, std, 1.0), 0.0)

    logits: list[float] = []
    all_blocks = set(range(n_splits))
    for combo in combinations(range(n_splits), n_splits // 2):
        is_sel = list(combo)
        oos_sel = sorted(all_blocks - set(combo))
        is_sr = _sharpe(is_sel)
        oos_sr = _sharpe(oos_sel)
        best = int(np.argmax(is_sr))
        rank = float(np.sum(oos_sr <= oos_sr[best]))  # 1 = worst .. N = best
        w = rank / (n_strat + 1.0)
        logits.append(math.log(w / (1.0 - w)))
    arr = np.asarray(logits)
    return PBOResult(
        pbo=float(np.mean(arr <= 0.0)),
        n_combinations=len(logits),
        n_strategies=n_strat,
        n_splits=n_splits,
        logits=tuple(float(x) for x in arr),
    )


def top_k_days_removed_pnl(returns: np.ndarray, ts_ns: np.ndarray, k: int) -> float:
    """Sum of per-bar returns after dropping the k UTC days with the largest daily sum."""
    r = np.asarray(returns, dtype=np.float64)
    days = np.asarray(ts_ns, dtype=np.int64) // NS_PER_DAY
    _, inverse = np.unique(days, return_inverse=True)
    daily = np.bincount(inverse, weights=r)
    if k <= 0:
        return float(daily.sum())
    keep = np.sort(daily)[::-1][k:]
    return float(keep.sum())
