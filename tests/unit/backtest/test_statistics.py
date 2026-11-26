from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from enigma.backtest.statistics import (
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


def test_sharpe_ratio_known_values() -> None:
    r = np.array([0.01, -0.005, 0.02, 0.0])
    expected = r.mean() / r.std(ddof=1)
    assert sharpe_ratio(r) == pytest.approx(expected)
    assert sharpe_ratio(r, 1095) == pytest.approx(expected * math.sqrt(1095))
    assert sharpe_ratio(np.ones(5)) == 0.0
    assert sharpe_ratio(np.array([0.1])) == 0.0


def test_psr_hand_computed() -> None:
    # z = 0.1 * sqrt(99) / sqrt(1 + (3 - 1) / 4 * 0.01) = 0.99251...
    assert psr_from_moments(0.1, 0.0, 100, 0.0, 3.0) == pytest.approx(0.83953, abs=1e-4)
    assert psr_from_moments(0.0, 0.0, 100, 0.0, 3.0) == pytest.approx(0.5)
    assert psr_from_moments(1.0, 0.0, 100, 5.0, 3.0) == 0.0  # invalid variance term: fail closed


def test_dsr_matches_bailey_lopez_de_prado_2014_example() -> None:
    # Paper example: annual SR 2.5 over 1250 daily obs, skew -3, kurtosis 10, N = 100 trials,
    # annualized cross-trial Sharpe variance 0.5 -> DSR ~ 0.90.
    sr = 2.5 / math.sqrt(250)
    var = 0.5 / 250
    sr0 = expected_max_sharpe(100, var)
    assert sr0 == pytest.approx(0.1132, abs=5e-4)
    assert psr_from_moments(sr, sr0, 1250, -3.0, 10.0) == pytest.approx(0.9004, abs=3e-3)


def test_expected_max_sharpe_edge_cases_and_monotonicity() -> None:
    assert expected_max_sharpe(1, 1.0) == 0.0
    assert expected_max_sharpe(10, 0.0) == 0.0
    values = [expected_max_sharpe(n, 1.0) for n in (2, 10, 100, 1000)]
    assert values == sorted(values)
    with pytest.raises(ValueError):
        expected_max_sharpe(0, 1.0)


def test_deflated_sharpe_on_series_with_one_trial_equals_psr() -> None:
    r = np.random.default_rng(0).normal(0.05, 1.0, 500)
    dsr = deflated_sharpe_ratio(r, n_trials=1, trial_sharpe_variance=0.01)
    assert dsr.sr0 == 0.0
    assert dsr.dsr == pytest.approx(probabilistic_sharpe_ratio(r))
    more = deflated_sharpe_ratio(r, n_trials=50, trial_sharpe_variance=0.01)
    assert more.dsr < dsr.dsr


def test_holm_known_example() -> None:
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(holm_adjusted_pvalues(p), [0.03, 0.06, 0.06, 0.02])
    assert holm_bonferroni(p, 0.05) == (True, False, False, True)
    assert holm_bonferroni([0.9], 0.05) == (False,)


def test_one_sided_ttest_matches_scipy() -> None:
    r = np.random.default_rng(1).normal(0.1, 1.0, 80)
    t, p = one_sided_ttest(r)
    ref = stats.ttest_1samp(r, 0.0, alternative="greater")
    assert t == pytest.approx(ref.statistic)
    assert p == pytest.approx(ref.pvalue)
    assert one_sided_ttest(np.zeros(10)) == (0.0, 1.0)


def test_stationary_bootstrap_properties() -> None:
    rng = np.random.default_rng(2)
    idx = stationary_bootstrap_indices(50, 200, 5.0, rng)
    assert idx.shape == (200, 50) and idx.min() >= 0 and idx.max() < 50
    continuing = np.mean((idx[:, 1:] - idx[:, :-1]) % 50 == 1)
    assert continuing == pytest.approx(1 - 1 / 5.0, abs=0.03)
    r = np.random.default_rng(3).normal(0.5, 1.0, 400)
    lo, hi = stationary_bootstrap_ci(
        r, lambda x: float(x.mean()), n_boot=500, mean_block=5, alpha=0.05, seed=9
    )
    assert lo < r.mean() < hi and lo > 0.3
    again = stationary_bootstrap_ci(
        r, lambda x: float(x.mean()), n_boot=500, mean_block=5, alpha=0.05, seed=9
    )
    assert again == (lo, hi)


def test_cscv_pbo_noise_vs_dominant() -> None:
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 1, (800, 10))
    pbo_noise = cscv_pbo(noise, n_splits=8)
    assert pbo_noise.n_combinations == 70
    assert pbo_noise.pbo > 0.4  # no skill: IS winner is not reliably above OOS median
    dominant = noise.copy()
    dominant[:, 0] += 1.0
    assert cscv_pbo(dominant, n_splits=8).pbo == 0.0
    with pytest.raises(ValueError):
        cscv_pbo(noise[:, :1])


def test_top_k_days_removed() -> None:
    day = 86_400 * 10**9
    ts = np.array([0, day // 3, day, day + 1, 2 * day, 3 * day], dtype=np.int64)
    r = np.array([0.02, 0.01, -0.01, 0.0, 0.05, -0.02])
    # daily sums: 0.03, -0.01, 0.05, -0.02
    assert top_k_days_removed_pnl(r, ts, 0) == pytest.approx(0.05)
    assert top_k_days_removed_pnl(r, ts, 1) == pytest.approx(0.0)
    assert top_k_days_removed_pnl(r, ts, 2) == pytest.approx(-0.03)
