from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from enigma.research.hybrid import compare_holdout
from enigma.research.narrative import intensity_at, parse_rss
from enigma.research.portfolio import research_proposal
from enigma.schemas.common import ClaimLevel
from enigma.schemas.portfolio import PortfolioProposal

NOW = datetime(2026, 11, 25, tzinfo=UTC)
RSS = """<?xml version="1.0"?>
<rss><channel>
<item><title>Funding crowded</title><link>https://example.test/a</link>
<description>BTC funding</description><pubDate>2026-11-01T00:00:00+00:00</pubDate></item>
<item><title>Later</title><link>https://example.test/b</link>
<description>after the bar</description><pubDate>2026-11-03T00:00:00+00:00</pubDate></item>
</channel></rss>
"""


def test_unvalidated_sleeves_are_not_executable() -> None:
    series = {"alpha": [0.01, -0.02, 0.03, 0.0], "beta": [0.02, 0.02, -0.01, 0.01]}
    proposal = research_proposal(
        series,
        {"alpha": ClaimLevel.L1_HYPOTHESIS, "beta": ClaimLevel.L3_INDEPENDENTLY_VALIDATED},
        now=NOW,
    )
    assert proposal.executable is False
    assert proposal.reason == "unvalidated_strategy"
    assert abs(sum(proposal.weights) - 1) < 1e-9
    with pytest.raises(ValidationError, match="L3"):
        PortfolioProposal(
            id="pfo_bad",
            strategy_ids=("alpha",),
            weights=(1.0,),
            claim_level=ClaimLevel.L0_UNVERIFIED,
            executable=True,
            reason="forged",
            created_at=NOW,
        )


def test_two_l3_sleeves_can_be_marked_executable_without_an_order() -> None:
    series = {"alpha": [0.01, -0.02, 0.04], "beta": [0.02, 0.01, -0.01]}
    proposal = research_proposal(
        series,
        {name: ClaimLevel.L3_INDEPENDENTLY_VALIDATED for name in series},
        now=NOW,
    )
    assert proposal.executable is True
    assert proposal.claim_level is ClaimLevel.L3_INDEPENDENTLY_VALIDATED


def test_ols_beats_the_constant_on_holdout_and_ignores_holdout_y() -> None:
    x = [float(i) for i in range(12)]
    y = [2.0 * item for item in x]
    first = compare_holdout(x, y, holdout=3)
    assert first.winner == "ols"
    assert "constant_mean" in first.loser_failure
    changed = [*y[:-3], 100.0, 100.0, 100.0]
    second = compare_holdout(x, changed, holdout=3)
    linear = next(fit for fit in second.fits if fit.model_class == "ols")
    original = next(fit for fit in first.fits if fit.model_class == "ols")
    assert linear.slope == original.slope
    assert linear.intercept == original.intercept


def test_rss_feature_ignores_text_published_after_the_bar() -> None:
    events = parse_rss(RSS)
    assert len(events) == 2
    bar = datetime(2026, 11, 2, tzinfo=UTC)
    assert intensity_at(events, bar) == 1
    assert intensity_at(events, datetime(2026, 11, 3, tzinfo=UTC)) == 2
