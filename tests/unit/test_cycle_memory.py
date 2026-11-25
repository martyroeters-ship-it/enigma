from __future__ import annotations

from pathlib import Path

from cycle_rig import Rig

from enigma.schemas.learning import ForecastRecord, Postmortem


def test_completed_cycle_resolves_a_forecast_and_stores_a_postmortem(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "c.sqlite3")
    cycle = rig.submit()
    assert rig.run(cycle) == "COMPLETED"
    forecasts = rig.stores.documents.list(ForecastRecord)
    assert len(forecasts) == 1
    assert forecasts[0].outcome is False
    assert forecasts[0].resolved_at is not None
    assert 0.0 < forecasts[0].probability < 1.0
    postmortems = rig.stores.documents.list(Postmortem)
    assert len(postmortems) == 1
    assert postmortems[0].cycle_id == cycle
    assert postmortems[0].error_class.value == "variance"
