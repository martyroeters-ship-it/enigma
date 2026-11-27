"""Discovery opens a new question only after the seeded backlog is scientifically exhausted."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from cycle_rig import Rig
from test_daemon import _daemon

from enigma.backtest import BarBacktestEngine
from enigma.core.hashing import sha256_of
from enigma.data.panel import periods_per_year
from enigma.orchestrator.backlog import HORIZONS, SYMBOLS, research_agenda
from enigma.orchestrator.market_scan import run_market_scan
from enigma.orchestrator.report import render_report
from enigma.research.discovery import replenish, resolve_mechanism
from enigma.research.hypothesis import MechanismScore, build_hypothesis
from enigma.research.mechanisms import MECHANISMS
from enigma.schemas.common import AgentRole, Instrument
from enigma.schemas.knowledge import EvidenceItem, Source, SourceKind
from enigma.schemas.market import MarketObservation, MarketScanReport
from enigma.schemas.quant import DatasetAcquisitionRequest
from enigma.schemas.research import (
    CostModel,
    DataQualityCheck,
    DataQualityReport,
    DatasetVersion,
    DiscoveredMechanism,
    Hypothesis,
    ResearchObjective,
)
from enigma.schemas.validation import GateOutcome, RedactedValidationSummary
from enigma.strategy import build_package

NOW = datetime(2026, 11, 27, tzinfo=UTC)
VOLUME_CLAIM = (
    "Elevated perpetual quote volume is positively associated with higher subsequent "
    "perpetual returns. The association is a question about quote volume, not funding."
)
INTEREST_CLAIM = (
    "Rising open interest ahead of a crowded move is a positioning question the "
    "current panel cannot measure."
)


def _instrument(symbol: str) -> Instrument:
    return Instrument(
        symbol=symbol, base=symbol[:-4], quote="USDT", kind="perpetual", venue="binance_usdm"
    )


def _fail_direction(
    rig: Rig, mechanism_id: str, symbol: str, horizon: str, *, summary: str = ""
) -> None:
    hyp_id = f"hyp_{mechanism_id}_{symbol}_{horizon}"
    hyp = Hypothesis(
        id=hyp_id,
        objective_id=f"obj_{mechanism_id}_{symbol}_{horizon}",
        family_id=f"fam_{mechanism_id}",
        title=mechanism_id,
        mechanism=mechanism_id,
        prediction="seed",
        formal_statement="seeded question",
        falsification_criteria=("gross sharpe <= 0",),
        instruments=(_instrument(symbol),),
        horizon=horizon,
        features=("basis",),
        trial_budget=1,
        cost_model_version="cm-test",
        generation_method="template:test",
        methodology_version="0.1.0",
        created_by=AgentRole.QUANT_RESEARCH,
        created_at=NOW,
    )
    rig.stores.documents.put(hyp, recorded_by="test")
    state: dict[str, object] = {"hypothesis_id": hyp_id, "validation_overall": "FAIL"}
    if summary:
        state["validation_summary_id"] = summary
    rig.stores.checkpoints.save(
        f"cyc_{mechanism_id}_{symbol}_{horizon}",
        "research_cycle",
        "COMPLETED",
        state,
        "completed",
    )


def _exhaust(rig: Rig) -> None:
    for mechanism_id in MECHANISMS:
        for symbol in SYMBOLS:
            for horizon in HORIZONS:
                _fail_direction(rig, mechanism_id, symbol, horizon)


def _evidence(rig: Rig, evidence_id: str, claim: str) -> None:
    rig.stores.documents.put(
        Source(
            id=f"src_{evidence_id}",
            kind=SourceKind.PREPRINT,
            provider="test",
            url=f"https://example.invalid/{evidence_id}",
            title=claim[:80],
            content_hash=sha256_of(claim),
            retrieved_by=AgentRole.INTELLIGENCE,
            retrieved_at=NOW,
        ),
        recorded_by="intelligence",
    )
    rig.stores.documents.put(
        EvidenceItem(
            id=evidence_id,
            source_id=f"src_{evidence_id}",
            excerpt=claim,
            claim=claim,
            topics=("discovery",),
            extraction_method="verbatim",
            is_ai_interpretation=False,
            created_by=AgentRole.INTELLIGENCE,
            created_at=NOW,
        ),
        recorded_by="intelligence",
    )


def test_a_viable_backlog_is_not_replaced(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "viable.sqlite3")
    _evidence(rig, "evi_early", VOLUME_CLAIM)
    assert replenish(rig.stores, NOW) is None
    assert rig.stores.documents.list(DiscoveredMechanism) == []


def test_empty_evidence_does_not_invent_a_question(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "empty.sqlite3")
    _exhaust(rig)
    assert replenish(rig.stores, NOW, backlog_exhausted=True) is None
    assert rig.stores.documents.list(DiscoveredMechanism) == []


def test_a_few_failures_do_not_retire_the_mechanism(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "few.sqlite3")
    for horizon in HORIZONS:
        _fail_direction(rig, "basis_reversion", "BTCUSDT", horizon)
    agenda = research_agenda(rig.stores, inputs_budget=10)
    viable = {item.candidate.id for item in agenda.ranked if item.viable}
    assert "opp_basis_reversion_ETHUSDT_1d" in viable
    assert "opp_basis_reversion_BTCUSDT_1d" not in viable


def test_a_method_gate_is_not_evidence_against_the_direction(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "method.sqlite3")
    rig.stores.documents.put(
        RedactedValidationSummary(
            id="vsm_method",
            report_id="vrp_method",
            strategy_package_id="pkg_method",
            protocol_version="0.1.0-slice",
            overall=GateOutcome.FAIL,
            gate_outcomes={"G02": GateOutcome.FAIL},
            failed_gate_names=("G02:chronological_split",),
        ),
        recorded_by="validation_service",
    )
    _fail_direction(rig, "basis_reversion", "BTCUSDT", "1d", summary="vsm_method")
    agenda = research_agenda(rig.stores, inputs_budget=10)
    match = [
        item for item in agenda.ranked if item.candidate.id == "opp_basis_reversion_BTCUSDT_1d"
    ]
    assert match and match[0].viable is True


def test_exhausted_seeds_discover_one_new_mechanism_and_a_reproducible_probe(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "disc.sqlite3")
    _exhaust(rig)
    _evidence(rig, "evi_volume", VOLUME_CLAIM)
    objective_id = replenish(rig.stores, NOW, backlog_exhausted=True)
    assert objective_id is not None
    objective = rig.stores.documents.require(ResearchObjective, objective_id)
    mechanism_id = next(topic for topic in objective.topics if topic.startswith("disc_"))
    assert mechanism_id not in MECHANISMS
    stored = rig.stores.documents.require(DiscoveredMechanism, mechanism_id)
    assert stored.features == ("perp_quote_volume",)
    assert stored.origin == "evidence"
    mechanism = resolve_mechanism(rig.stores.documents, mechanism_id)
    assert mechanism is not None
    hypothesis = build_hypothesis(
        objective,
        mechanism,
        MechanismScore(mechanism.id, 1, 0, 1, 0, stored.evidence_ids, ()),
        objective.instruments,
        cycle_id="cyc_accept",
        now=NOW,
        dataset_version_ids=("ds_accept",),
    )
    rig.stores.documents.put(hypothesis, recorded_by="quant_research")
    package = build_package(
        mechanism.source_template,
        mechanism.entrypoint,
        {"lookback": 21, "z_entry": 1.5, "sign": 1.0},
        _cost(),
        hypothesis.id,
        name="discovered",
        required_columns=mechanism.features,
        created_by=AgentRole.STRATEGY_ENGINEERING,
        created_at=NOW,
    )
    frame = _volume_frame()
    first, second = _run(package.source_code, frame), _run(package.source_code, frame)
    assert first.result_hash == second.result_hash
    assert first.result_hash
    daemon = _daemon(rig, max_cycles_per_day=0, continuous=True, review_interval_hours=0)
    daemon.run(max_iterations=1)
    cycles = [
        row
        for row in rig.stores.checkpoints.list("research_cycle")
        if row["state"].get("objective_id") == objective_id
    ]
    assert cycles
    opened = rig.stores.documents.require(ResearchObjective, cycles[0]["state"]["objective_id"])
    assert mechanism_id in opened.topics
    text = render_report(rig.stores, NOW)
    assert "Discovered mechanisms:" in text
    assert mechanism_id in text
    assert "Costs:" in text
    _evidence(
        rig,
        "evi_next",
        "Spot price continuation is positively associated with higher subsequent "
        "perpetual returns on a later question.",
    )
    # The opened cycle records the first objective, so a distinct claim can be registered.
    follow = replenish(rig.stores, NOW, backlog_exhausted=True)
    assert follow is not None and follow != objective_id


def test_missing_data_becomes_a_request_not_a_hypothesis(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "data.sqlite3")
    _exhaust(rig)
    _evidence(rig, "evi_oi", INTEREST_CLAIM)
    assert replenish(rig.stores, NOW, backlog_exhausted=True) is None
    requests = rig.stores.documents.list(DatasetAcquisitionRequest)
    assert len(requests) == 1
    assert requests[0].columns == ("open_interest",)
    assert rig.stores.documents.list(DiscoveredMechanism) == []


def test_duplicate_observations_are_not_two_discoveries(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "scan.sqlite3")
    now = datetime(2026, 11, 27, 12, tzinfo=UTC)
    start = now - timedelta(hours=30)
    for index in range(20):
        value = 10.0 if index == 19 else 0.0
        for copy in (0, 1):
            for symbol in ("BTCUSDT", "ETHUSDT"):
                rig.stores.documents.put(
                    MarketObservation(
                        id=f"mob_{symbol}_{index}_{copy}",
                        symbol=symbol,
                        field="funding_rate",
                        available_at=start + timedelta(hours=index),
                        value=value,
                    ),
                    recorded_by="test",
                )
    assert run_market_scan(rig.stores, now, interval_hours=1) is True
    first = rig.stores.documents.list(MarketScanReport)[0]
    assert first.top_symbol in {"BTCUSDT", "ETHUSDT"}
    assert "identical series" in first.note
    assert first.objective_id
    later = now + timedelta(hours=2)
    assert run_market_scan(rig.stores, later, interval_hours=1) is True
    reports = rig.stores.documents.list(MarketScanReport)
    assert len(reports) == 2
    second = next(item for item in reports if item.id != first.id)
    assert second.objective_id is None
    assert "historical" in second.note


def _cost() -> CostModel:
    return CostModel(
        version="cm-test-1",
        taker_fee_bps=4.0,
        maker_fee_bps=2.0,
        half_spread_bps=1.0,
        impact_bps_per_million=10.0,
        capacity_notional_usd=1_000_000.0,
    )


def _volume_frame() -> pl.DataFrame:
    rng = np.random.default_rng(4)
    n = 80
    start = datetime(2024, 1, 1, tzinfo=UTC)
    step = timedelta(hours=8)
    volume = np.abs(rng.normal(1.0, 0.4, n))
    volume[-1] = volume[:-1].mean() + 8 * volume[:-1].std()
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    ts = [start + i * step for i in range(n)]
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * n,
            "ts": ts,
            "available_at": [item + step for item in ts],
            "perp_close": close,
            "perp_quote_volume": volume,
            "funding_rate": np.zeros(n),
        }
    )


def _run(source: str, frame: pl.DataFrame):
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - the compiled probe, same source the package stores
    probe = namespace["DiscoveredProbe"]()
    probe.configure({"lookback": 21, "z_entry": 1.5, "sign": 1.0})
    positions = probe.target_positions(
        {"perp_quote_volume": frame.get_column("perp_quote_volume").to_numpy()}
    )
    return BarBacktestEngine(1095).run(
        {"BTCUSDT": positions}, frame, _cost(), seed=1, notional_usd=100_000.0
    )


def test_dataset_frequency_is_recorded_on_the_version(tmp_path: Path) -> None:
    """Advertised horizon and stored frequency are the same field the cycle will request."""
    assert periods_per_year("8h") == 1095
    assert periods_per_year("1h") != 1095
    version = DatasetVersion(
        id="ds_freq",
        name="freq",
        instruments=(_instrument("BTCUSDT"),),
        venue="binance_usdm",
        frequency="8h",
        schema_version="t",
        columns=("perp_close",),
        time_start=NOW,
        time_end=NOW,
        row_count=1,
        content_hash=sha256_of({"n": 1}),
        uri="memory://freq",
        pit_certified=True,
        quality=DataQualityReport(checks=(DataQualityCheck(name="pit", passed=True),)),
        provenance={"test": True},
        created_by=AgentRole.DATA_ENGINEERING,
        created_at=NOW,
    )
    assert version.frequency == "8h"
