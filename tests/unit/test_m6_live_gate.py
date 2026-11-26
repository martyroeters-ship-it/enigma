"""The live-capital gate stays shut. These tests do not authorize trading."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from enigma.core.errors import PermissionDeniedError
from enigma.risk.live_gate import mark_live_gate_passed, record_human_authorization
from enigma.risk.service import RiskLimits
from enigma.schemas.common import AgentRole, SystemActor

ROOT = Path(__file__).resolve().parents[2]
LIMITS = ROOT / "registries" / "risk" / "limits.yaml"


def _flags() -> tuple[bool, bool]:
    raw = yaml.safe_load(LIMITS.read_text(encoding="utf-8"))
    return bool(raw["paper_trading_enabled"]), bool(raw["live_trading_enabled"])


def test_no_actor_can_mark_the_gate_passed() -> None:
    before = LIMITS.read_text(encoding="utf-8")
    for actor in (*AgentRole, SystemActor.HUMAN, SystemActor.SYSTEM):
        with pytest.raises(PermissionDeniedError, match="cannot mark"):
            mark_live_gate_passed(actor)
    assert LIMITS.read_text(encoding="utf-8") == before
    assert _flags() == (False, False)


def test_an_agent_cannot_record_the_authorization(tmp_path: Path) -> None:
    for actor in AgentRole:
        with pytest.raises(PermissionDeniedError, match="agent cannot record"):
            record_human_authorization(
                actor,
                tmp_path,
                account="desk",
                venue="none",
                max_notional_usd=1.0,
                expires_on=date(2026, 12, 1),
            )
    assert list(tmp_path.iterdir()) == []


def test_a_human_file_does_not_enable_live_trading(tmp_path: Path) -> None:
    path = record_human_authorization(
        SystemActor.HUMAN,
        tmp_path / "human-only",
        account="desk",
        venue="none",
        max_notional_usd=1.0,
        expires_on=date(2026, 12, 1),
    )
    body = path.read_text(encoding="utf-8")
    assert "live_trading_enabled" in body
    assert _flags() == (False, False)
    limits = RiskLimits.load(LIMITS)
    assert limits.paper_trading_enabled is False
    forged = tmp_path / "forged.yaml"
    forged.write_text(
        "version: forged\npaper_trading_enabled: false\nlive_trading_enabled: true\n"
        "max_gross_leverage: 1\nmax_sleeve_notional_usd: 1\n"
        "max_drawdown_research: 1\nmin_holdout_net_sharpe: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="constitutionally prohibited"):
        RiskLimits.load(forged)
