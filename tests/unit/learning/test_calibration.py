"""Brier, log loss, and the overconfidence flag for M2.1 forecasts."""

from __future__ import annotations

import math

import pytest

from enigma.learning.calibration import (
    brier_score,
    expected_calibration_error,
    is_overconfident,
    log_loss,
    mean_overconfidence,
)


def test_perfect_forecasts_score_zero() -> None:
    probabilities = [0.0, 1.0, 0.0, 1.0]
    outcomes = [False, True, False, True]
    assert brier_score(probabilities, outcomes) == 0.0
    assert log_loss(probabilities, outcomes) == pytest.approx(0.0, abs=1e-12)
    assert is_overconfident(probabilities, outcomes) is False


def test_synthetic_bias_is_overconfident() -> None:
    probabilities = [0.9] * 20
    outcomes = [False] * 20
    assert brier_score(probabilities, outcomes) == pytest.approx(0.81)
    assert mean_overconfidence(probabilities, outcomes) == pytest.approx(0.9)
    assert is_overconfident(probabilities, outcomes) is True
    assert expected_calibration_error(probabilities, outcomes) == pytest.approx(0.9)


def test_underconfident_high_hit_rate_is_not_flagged() -> None:
    probabilities = [0.4] * 10
    outcomes = [True] * 10
    assert mean_overconfidence(probabilities, outcomes) < 0
    assert is_overconfident(probabilities, outcomes) is False


def test_log_loss_penalises_confident_mistakes() -> None:
    wrong = log_loss([0.99], [False])
    unsure = log_loss([0.5], [False])
    assert wrong > unsure
    assert math.isfinite(log_loss([0.0, 1.0], [False, True]))


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="same non-empty"):
        brier_score([0.5], [True, False])
