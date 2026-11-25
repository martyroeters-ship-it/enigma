"""Atomic CostLedger reservations: committed spend + open holds enforce ceilings."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from enigma.core.errors import BudgetExceededError
from enigma.core.ids import new_id
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import CostRecord
from enigma.store.db import Database
from enigma.store.ledger import CostLedger

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _ledger(tmp_path) -> CostLedger:
    return CostLedger(Database(tmp_path / "ctrl.sqlite"))


def _cost(usd: float = 0.5, tokens_in: int = 100, tokens_out: int = 50) -> CostRecord:
    return CostRecord(
        id=new_id("cst"),
        actor=AgentRole.QUANT_RESEARCH,
        task_id="tsk_1",
        category="llm",
        provider="ollama",
        model="qwen2.5:3b",
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        usd=usd,
        http_requests=1,
        created_at=NOW,
    )


def test_schema_has_cost_reservations(tmp_path) -> None:
    db = Database(tmp_path / "ctrl.sqlite")
    assert db.schema_version >= 2
    row = db.query_one(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='cost_reservations'"
    )
    assert row is not None


def test_reserve_commit_releases_hold(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    rid = new_id("rsv")
    ledger.reserve(
        reservation_id=rid,
        actor="quant_research",
        task_id="tsk_1",
        category="llm",
        provider="ollama",
        model="qwen2.5:3b",
        usd=1.0,
        tokens=200,
        http_requests=2,
        max_usd=2.0,
        max_tokens=1000,
        max_http=10,
        now=NOW,
    )
    assert ledger.reserved_totals(actor="quant_research", task_id="tsk_1")["usd"] == 1.0
    assert ledger.exposure(actor="quant_research", task_id="tsk_1")["usd"] == 1.0

    cost = _cost(usd=0.4)
    ledger.commit_reservation(rid, cost)
    assert ledger.reserved_totals(actor="quant_research", task_id="tsk_1")["usd"] == 0.0
    assert ledger.totals(actor="quant_research", task_id="tsk_1")["usd"] == pytest.approx(0.4)
    assert ledger.exposure(actor="quant_research", task_id="tsk_1")["usd"] == pytest.approx(0.4)


def test_concurrent_reserves_cannot_overbook(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.reserve(
        reservation_id=new_id("rsv"),
        actor="quant_research",
        task_id="tsk_1",
        category="llm",
        provider="ollama",
        model="qwen2.5:3b",
        usd=0.8,
        tokens=100,
        max_usd=1.0,
        now=NOW,
    )
    with pytest.raises(BudgetExceededError, match="usd ceiling"):
        ledger.reserve(
            reservation_id=new_id("rsv"),
            actor="quant_research",
            task_id="tsk_1",
            category="llm",
            provider="ollama",
            model="qwen2.5:3b",
            usd=0.3,
            tokens=100,
            max_usd=1.0,
            now=NOW,
        )


def test_release_and_expire_free_budget(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    rid = new_id("rsv")
    ledger.reserve(
        reservation_id=rid,
        actor="quant_research",
        task_id=None,
        category="llm",
        provider="openai",
        model="gpt-4o-mini",
        usd=0.9,
        tokens=50,
        max_usd=1.0,
        ttl_seconds=60,
        now=NOW,
    )
    ledger.release_reservation(rid, now=NOW)
    ledger.reserve(
        reservation_id=new_id("rsv"),
        actor="quant_research",
        task_id=None,
        category="llm",
        provider="openai",
        model="gpt-4o-mini",
        usd=0.9,
        tokens=50,
        max_usd=1.0,
        now=NOW,
    )

    stale = new_id("rsv")
    ledger.reserve(
        reservation_id=stale,
        actor="cio",
        task_id=None,
        category="llm",
        provider="openai",
        model="gpt-4o-mini",
        usd=0.5,
        tokens=10,
        max_usd=1.0,
        ttl_seconds=1,
        now=NOW,
    )
    # Force expiry by updating expires_at into the past via a second reserve clock
    later = NOW + timedelta(seconds=5)
    assert ledger.expire_stale_reservations(now=later) >= 1
    assert ledger.reserved_totals(actor="cio")["usd"] == 0.0


def test_daily_cap_includes_reservations(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    day = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
    ledger.reserve(
        reservation_id=new_id("rsv"),
        actor="quant_research",
        task_id=None,
        category="llm",
        provider="openai",
        model="gpt-4o-mini",
        usd=0.7,
        tokens=10,
        daily_cap_usd=1.0,
        day_start=day,
        now=NOW,
    )
    with pytest.raises(BudgetExceededError, match="daily API ceiling"):
        ledger.reserve(
            reservation_id=new_id("rsv"),
            actor="intelligence",
            task_id=None,
            category="llm",
            provider="openai",
            model="gpt-4o-mini",
            usd=0.4,
            tokens=10,
            daily_cap_usd=1.0,
            day_start=day,
            now=NOW,
        )
