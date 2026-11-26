"""AT3: live research against real public APIs (deselect with -m 'not network')."""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from enigma.core.clock import SystemClock
from enigma.intelligence import (
    HttpBudget,
    HttpClient,
    RawStore,
    ResearchEngine,
    ResearchQuery,
    persist_result,
)
from enigma.intelligence.providers import (
    ArxivProvider,
    BraveSearchProvider,
    OpenAlexProvider,
    WikipediaProvider,
)
from enigma.knowledge import KnowledgeBase
from enigma.schemas.common import AgentRole, ClaimLevel, Plane
from enigma.schemas.governance import Budget
from enigma.schemas.knowledge import Source
from enigma.store import PlaneView, Stores

pytestmark = pytest.mark.network


def test_live_research_returns_real_hashed_sources(tmp_path: Path, stores: Stores) -> None:
    clock = SystemClock()
    started = clock.now()
    budget = Budget(max_http_requests=30, max_wall_seconds=240)
    with HttpClient(HttpBudget(budget.max_http_requests), RawStore(tmp_path / "raw")) as client:
        providers = [
            ArxivProvider(client, clock),
            OpenAlexProvider(client, clock),
            WikipediaProvider(client, clock),
        ]
        engine = ResearchEngine(providers, client, clock)
        result = engine.run(
            ResearchQuery(
                topic="perpetual futures funding rate",
                max_sources=10,
                max_depth=1,
                citation_fanout=1,
                per_provider_limit=5,
                max_expansion_queries=0,
                author_fanout=0,
                budget=budget,
            )
        )
    assert 0 < result.http_requests_used <= 30
    assert len(result.sources) >= 3
    live_window = (started - timedelta(seconds=5), datetime.now(UTC) + timedelta(seconds=5))
    for src in result.sources:
        assert src.url.startswith("https://")
        assert live_window[0] <= src.retrieved_at <= live_window[1]  # not a fixture timestamp
        assert src.raw_uri is not None and Path(src.raw_uri).is_file()
        digest = hashlib.sha256(Path(src.raw_uri).read_bytes()).hexdigest()
        assert src.content_hash == f"sha256:{digest}"
    providers_seen = {s.provider for s in result.sources}
    assert providers_seen & {"arxiv", "openalex"}

    view = PlaneView(
        stores.documents,
        frozenset({Plane.RESEARCH, Plane.CONTROL}),
        frozenset({Plane.RESEARCH}),
        AgentRole.INTELLIGENCE.value,
    )
    kb = KnowledgeBase(view, clock)
    report = persist_result(result, kb)
    stored = view.require(Source, report.source_ids[result.sources[0].id])
    assert stored.content_hash.startswith("sha256:")
    for item in result.evidence[:3]:
        prov = kb.provenance(item.id)
        assert prov.source.url.startswith("https://")
        assert prov.effective_claim_level is ClaimLevel.L0_UNVERIFIED


@pytest.mark.skipif(not os.environ.get("BRAVE_SEARCH_API_KEY"), reason="BRAVE_SEARCH_API_KEY unset")
def test_live_brave_search_when_configured(tmp_path: Path) -> None:
    with HttpClient(HttpBudget(3), RawStore(tmp_path / "raw")) as client:
        hits = BraveSearchProvider(client, SystemClock()).search(
            "perpetual futures funding rate", 5, None
        )
    assert hits and all(h.url.startswith("http") for h in hits)
