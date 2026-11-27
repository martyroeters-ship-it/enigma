"""A named observable is registered from the public archive, or recorded as unmet."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from data_harness import NOW, RecordedServer, make_http

from enigma.core.clock import FrozenClock
from enigma.data.acquisition import fulfill
from enigma.data.binance import BinanceArchiveClient
from enigma.data.catalog import DataCatalog
from enigma.schemas.common import AgentRole, Plane
from enigma.schemas.quant import DatasetAcquisitionRequest
from enigma.schemas.research import DatasetVersion
from enigma.store import Stores

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 31, 16, tzinfo=UTC)


def test_funding_is_registered_and_open_interest_is_not_invented(
    tmp_path: Path, server: RecordedServer, clock: FrozenClock, stores: Stores
) -> None:
    client = BinanceArchiveClient(tmp_path / "raw", http=make_http(server), clock=clock)
    catalog = DataCatalog(
        tmp_path,
        stores.documents,
        Plane.RESEARCH,
        AgentRole.DATA_ENGINEERING,
        client=client,
        clock=clock,
        costs=stores.costs,
        okx_cross_check=False,
    )
    funding = DatasetAcquisitionRequest(
        id="dar_funding",
        requested_by=AgentRole.QUANT_RESEARCH,
        description="funding_rate",
        columns=("funding_rate",),
        rationale="The hypothesis names the funding settlement.",
        created_at=NOW,
    )
    interest = DatasetAcquisitionRequest(
        id="dar_oi",
        requested_by=AgentRole.QUANT_RESEARCH,
        description="open_interest",
        columns=("open_interest",),
        rationale="Positioning was requested. No archive lists it.",
        created_at=NOW,
    )
    stores.documents.put(funding, recorded_by="quant_research")
    stores.documents.put(interest, recorded_by="quant_research")
    before = len(stores.documents.list(DatasetVersion, limit=20))
    registered = fulfill(
        stores,
        funding,
        catalog,
        symbols=("BTCUSDT",),
        start=START,
        end=END,
        frequency="8h",
        now=NOW,
    )
    unmet = fulfill(
        stores,
        interest,
        catalog,
        symbols=("BTCUSDT",),
        start=START,
        end=END,
        frequency="8h",
        now=NOW,
    )
    assert registered.status == "registered"
    assert registered.content_hash.startswith("sha256:")
    assert "PIT certified=True" in registered.assessment
    assert unmet.status == "unmet"
    assert unmet.content_hash == ""
    after = stores.documents.list(DatasetVersion, limit=20)
    assert len(after) == before + 1
    assert "open_interest" not in after[-1].columns
    assert stores.costs.totals()["usd"] == 0
