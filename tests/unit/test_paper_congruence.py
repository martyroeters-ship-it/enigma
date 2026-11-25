"""G17 paper congruence gate (VALIDATION_PROTOCOL §7)."""

from __future__ import annotations

from enigma.schemas.validation import GateOutcome
from enigma.validation.gates import g17_paper_congruence


def test_g17_passes_within_band() -> None:
    gate = g17_paper_congruence(research_net_sharpe=1.0, paper_net_sharpe=0.7, paper_bars=30)
    assert gate.result is GateOutcome.PASS
    assert gate.gate_id == "G17"


def test_g17_fails_outside_band() -> None:
    gate = g17_paper_congruence(research_net_sharpe=1.0, paper_net_sharpe=-1.0, paper_bars=30)
    assert gate.result is GateOutcome.FAIL


def test_g17_fails_when_too_few_bars() -> None:
    gate = g17_paper_congruence(research_net_sharpe=1.0, paper_net_sharpe=1.0, paper_bars=5)
    assert gate.result is GateOutcome.FAIL
