"""AT15-style plane isolation at the repository and tool layers (single host, see ADR-015)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from cycle_rig import Rig

from enigma.core.errors import PermissionDeniedError
from enigma.schemas.common import Plane
from enigma.schemas.validation import ValidationReport
from enigma.store.documents import PlaneView

RESEARCH_READ = frozenset({Plane.RESEARCH, Plane.CONTROL})


def test_research_plane_cannot_read_validation_reports(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    cycle = rig.submit()
    assert rig.run(cycle) == "COMPLETED"
    report = rig.stores.documents.list(ValidationReport, limit=1)[0]
    research = PlaneView(rig.stores.documents, RESEARCH_READ, frozenset({Plane.RESEARCH}), "qr")
    with pytest.raises(PermissionDeniedError):
        research.get(ValidationReport, report.id)
    with pytest.raises(PermissionDeniedError):
        research.list(ValidationReport)
    assert not [
        hit
        for hit in research.search_text("validation report gates")
        if hit[1] == "validation_report"
    ]


def test_research_roles_cannot_reach_validation_tools(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    for role in ("quant_research", "strategy_engineering", "intelligence", "data_engineering"):
        contract = rig.contracts.get(role)  # type: ignore[arg-type]
        assert "validation.run" not in contract.tools
        assert Plane.VALIDATION not in contract.readable_planes


def test_research_acquisition_cannot_cross_cutoff() -> None:
    from enigma.orchestrator.services import OperationalServices

    svc = OperationalServices.__new__(OperationalServices)
    svc._cutoffs = {("BTCUSDT",): datetime(2024, 6, 1, tzinfo=UTC)}  # type: ignore[attr-defined]
    with pytest.raises(PermissionDeniedError):
        svc.data_acquire_research(
            ["BTCUSDT"], datetime(2020, 1, 1, tzinfo=UTC), datetime(2024, 7, 1, tzinfo=UTC), "8h"
        )
