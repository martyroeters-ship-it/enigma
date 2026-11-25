"""Lessons from meta-learning must seal into ResearchContext for the next cycle."""

from __future__ import annotations

from datetime import UTC, datetime

from enigma.research.context import build_research_context
from enigma.schemas.common import AgentRole, ClaimLevel, Instrument
from enigma.schemas.knowledge import Lesson
from enigma.schemas.quant import DatasetInfo
from enigma.schemas.research import ResearchObjective

NOW = datetime(2026, 3, 1, tzinfo=UTC)
BTC = Instrument(symbol="BTCUSDT", base="BTC", quote="USDT", kind="perpetual", venue="binance_usdm")


def test_lessons_are_first_class_in_research_context() -> None:
    objective = ResearchObjective(
        id="obj_1",
        statement="Find a funding edge",
        instruments=(BTC,),
        created_by=AgentRole.CIO,
        created_at=NOW,
    )
    lesson = Lesson(
        id="les_1",
        statement="Funding carry fails after costs",
        topics=("funding", "cost_fragility"),
        derived_from_ids=("hyp_1",),
        outcome="failure",
        claim_level=ClaimLevel.L2_HISTORICAL,
        created_by=AgentRole.META_LEARNING,
        created_at=NOW,
    )
    ctx = build_research_context(
        objective=objective,
        research_cutoff=NOW,
        datasets=(
            DatasetInfo(
                dataset_version_id="ds_1",
                name="binance_btcusdt_1h",
                frequency="1h",
                bar_hours=1.0,
                symbols=("BTCUSDT",),
                columns=("funding_rate", "close"),
                coverage_start=NOW,
                coverage_end=NOW,
                rows=10,
            ),
        ),
        lessons=(lesson,),
    )
    assert len(ctx.lessons) == 1
    assert ctx.lessons[0].id == "les_1"
    assert "cost_fragility" in ctx.lessons[0].topics
    assert ctx.instruments == ("BTCUSDT",)
