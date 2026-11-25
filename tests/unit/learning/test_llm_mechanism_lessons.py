"""LLM hypothesis domains are not in MECHANISMS; lessons must still derive."""

from __future__ import annotations

from datetime import UTC, datetime

from enigma.learning.lessons import CycleEvidence, derive_lessons, propose_next
from enigma.research.mechanisms import MECHANISMS
from enigma.schemas.common import AgentRole, Instrument
from enigma.schemas.research import Hypothesis

NOW = datetime(2026, 3, 1, tzinfo=UTC)
BTC = Instrument(symbol="BTCUSDT", base="BTC", quote="USDT", kind="perpetual", venue="binance")


def _hyp() -> Hypothesis:
    return Hypothesis(
        id="hyp_llm",
        objective_id="obj",
        family_id="fam_cryptocurrency",
        title="Extreme Negative Funding",
        mechanism="arbitrage",
        prediction="negative funding predicts positive returns",
        formal_statement="funding_rate < threshold",
        falsification_criteria=("holdout sharpe < 0",),
        instruments=(BTC,),
        horizon="8h",
        features=("funding_rate",),
        search_space={"funding_rate_threshold": (-0.05,)},
        trial_budget=3,
        cost_model_version="cm-binance-usdm-v1",
        evidence_ids=(),
        generation_method="llm:structured-hypothesis-v1",
        methodology_version="qm-1",
        created_by=AgentRole.QUANT_RESEARCH,
        created_at=NOW,
    )


def test_derive_lessons_accepts_unknown_llm_domain() -> None:
    ev = CycleEvidence(
        cycle_id="cyc_1",
        hypothesis=_hyp(),
        experiments=(),
        summary=None,
        mechanism_id="cryptocurrency",
    )
    lessons = derive_lessons(ev, NOW)
    assert lessons
    assert "arbitrage" in lessons[0].statement
    nxt = propose_next(ev, lessons, tested_mechanisms={}, now=NOW)
    assert nxt.mechanism_id in MECHANISMS
