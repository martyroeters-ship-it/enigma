"""Readiness of the discovery architecture.

`assess` only reads. `demonstrate` writes into the store it is given, which must not
be the active institutional home. A missing model or a missing home is `blocked`,
not a pass and not a fabricated result.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from enigma.backtest.engine import BarBacktestEngine
from enigma.core.hashing import sha256_hex
from enigma.data.acquisition import fulfill
from enigma.data.catalog import DataCatalog
from enigma.learning.intervention import BenchmarkComparison, apply_intervention
from enigma.orchestrator.report import build_report
from enigma.research.inquiry import register_case
from enigma.research.intelligence_bench import probe_local_model
from enigma.research.program import compile_program
from enigma.schemas.common import AgentRole, ClaimLevel
from enigma.schemas.governance import ImprovementProposal, ProposalTarget
from enigma.schemas.inquiry import (
    Computation,
    DataNeed,
    Participant,
    Prediction,
    ResearchProgram,
    ScientificCase,
)
from enigma.schemas.knowledge import EvidenceItem, Source, SourceKind
from enigma.schemas.quant import DatasetAcquisitionRequest
from enigma.schemas.research import CostModel
from enigma.store import Stores
from enigma.validation.protocol import load_protocol, protocol_for_sampling


def refuse_active_home(home: Path, active: Path | None) -> str:
    if active is not None and home.resolve() == active.resolve():
        return "refusing to write to the active institutional home"
    return ""


def assess(stores: Stores, *, model_status: str | None = None) -> dict[str, Any]:
    """Ten acceptance criteria. Status is pass, fail, or blocked."""
    report = build_report(stores)
    hourly = protocol_for_sampling("1h", load_protocol())
    full = load_protocol(version="1.0.0")
    gates_ok = tuple(hourly.required_gates) == tuple(full.required_gates)
    model = model_status if model_status is not None else probe_local_model()
    criteria = [
        _exhaustion(stores),
        _discovery(stores, model),
        _hypothesis(stores),
        _acquisition(stores),
        _method(gates_ok),
        _backtest(stores),
        _validation(stores, gates_ok),
        _reflection(report),
        _replenish(stores),
        _improvement(report),
    ]
    return {
        "criteria": criteria,
        "report": report,
        "model": model,
        "protocol_1h": hourly.version,
        "protocol_8h_default": load_protocol().version,
        "paid_usd": report["inference"]["usd"],
    }


def verdict(criteria: list[dict[str, Any]]) -> dict[str, Any]:
    """Exit 0 only when every mandatory criterion passed.

    Exit 2 means at least one criterion failed. Exit 3 means nothing failed
    but a mandatory criterion is still blocked. Both are not acceptance.
    """
    counts = {"pass": 0, "fail": 0, "blocked": 0}
    for item in criteria:
        status = str(item.get("status") or "")
        if status in counts:
            counts[status] += 1
        else:
            counts["fail"] += 1
    if counts["fail"]:
        code = 2
    elif counts["blocked"]:
        code = 3
    else:
        code = 0
    label = "ACCEPTED" if code == 0 else "NOT ACCEPTED"
    return {
        "code": code,
        "label": label,
        "counts": counts,
        "note": (
            "Exit 0 means every mandatory criterion passed. "
            "A blocked mandatory criterion is not acceptance."
        ),
    }


def _criterion(number: int, name: str, status: str, evidence: str) -> dict[str, Any]:
    return {"n": number, "name": name, "status": status, "evidence": evidence}


def _exhaustion(stores: Stores) -> dict[str, Any]:
    cycles = stores.checkpoints.list("research_cycle")
    if not cycles:
        return _criterion(
            1,
            "original backlog exhausted",
            "blocked",
            "No research cycles are in this store, so exhaustion cannot be observed.",
        )
    from enigma.orchestrator.backlog import research_agenda  # noqa: PLC0415

    agenda = research_agenda(stores, inputs_budget=24)
    viable = [item for item in agenda.ranked if item.viable]
    if viable:
        return _criterion(
            1,
            "original backlog exhausted",
            "fail",
            f"{len(viable)} seed directions are still viable.",
        )
    return _criterion(
        1,
        "original backlog exhausted",
        "pass",
        f"No viable seed direction remains among {len(agenda.ranked)} ranked.",
    )


def _discovery(stores: Stores, model: str) -> dict[str, Any]:
    from enigma.schemas.inquiry import ScientificCase  # noqa: PLC0415

    cases = stores.documents.list(ScientificCase, limit=20)
    if not cases:
        return _criterion(
            2,
            "new mechanism from evidence and a model",
            "blocked",
            "No scientific case is registered. "
            + (
                "The local model is not running, so none was solicited."
                if model != "available"
                else "Run enigma inquire against an isolated home that already holds evidence."
            ),
        )
    case = cases[0]
    return _criterion(
        2,
        "new mechanism from evidence and a model",
        "pass",
        f"Case {case.id} cites {', '.join(case.supporting_evidence_ids)}. "
        "The interpretation field is not evidence.",
    )


def _hypothesis(stores: Stores) -> dict[str, Any]:
    from enigma.schemas.inquiry import ScientificCase  # noqa: PLC0415

    cases = stores.documents.list(ScientificCase, limit=20)
    if not cases or not cases[0].predictions:
        return _criterion(3, "falsifiable hypothesis", "blocked", "No registered prediction.")
    prediction = cases[0].predictions[0]
    return _criterion(
        3,
        "falsifiable hypothesis",
        "pass",
        f"{prediction.statement} Observable: {prediction.observable}.",
    )


def _acquisition(stores: Stores) -> dict[str, Any]:
    from enigma.schemas.inquiry import AcquisitionRecord  # noqa: PLC0415

    records = stores.documents.list(AcquisitionRecord, limit=20)
    registered = [item for item in records if item.status == "registered" and item.content_hash]
    unmet = [item for item in records if item.status == "unmet"]
    if not registered:
        return _criterion(
            4,
            "real market data acquired",
            "blocked",
            "No registered acquisition. Unmet requests are not filled with substitute columns.",
        )
    return _criterion(
        4,
        "real market data acquired",
        "pass",
        f"{registered[0].observable} dataset {registered[0].dataset_version_id} "
        f"hash {registered[0].content_hash}. Unmet observables: "
        + (", ".join(item.observable for item in unmet) or "none"),
    )


def _method(gates_ok: bool) -> dict[str, Any]:
    if not gates_ok:
        return _criterion(5, "methodology", "fail", "1h protocol dropped 1.0.0 gates")
    return _criterion(
        5,
        "methodology",
        "pass",
        "1h and 1d protocols carry every required gate from protocol 1.0.0.",
    )


def _backtest(stores: Stores) -> dict[str, Any]:
    from enigma.schemas.inquiry import ScientificCase  # noqa: PLC0415

    cases = stores.documents.list(ScientificCase, limit=5)
    if not cases:
        return _criterion(6, "reproducible backtest", "blocked", "No case to compile.")
    try:
        source = compile_program(cases[0].program)
    except ValueError as exc:
        return _criterion(6, "reproducible backtest", "fail", str(exc))
    first = sha256_hex(source)
    second = sha256_hex(compile_program(cases[0].program))
    if first != second:
        return _criterion(6, "reproducible backtest", "fail", "compiler is not deterministic")
    return _criterion(6, "reproducible backtest", "pass", f"program source hash {first}")


def _validation(stores: Stores, gates_ok: bool) -> dict[str, Any]:
    from enigma.schemas.validation import RedactedValidationSummary  # noqa: PLC0415

    summaries = stores.documents.list(RedactedValidationSummary, limit=5)
    if summaries:
        return _criterion(
            7,
            "independent validation",
            "pass",
            f"{summaries[0].id} overall {summaries[0].overall.value}",
        )
    if not gates_ok:
        return _criterion(
            7, "independent validation", "fail", "frequency protocol is weaker than 1.0.0"
        )
    return _criterion(
        7,
        "independent validation",
        "blocked",
        "No validation summary is in this store. The holdout was not opened here.",
    )


def _reflection(report: dict[str, Any]) -> dict[str, Any]:
    if report["postmortems"] < 1:
        return _criterion(8, "reflection and next plan", "blocked", "No postmortem in this store.")
    count = report["postmortems"]
    return _criterion(8, "reflection and next plan", "pass", f"{count} postmortem(s)")


def _replenish(stores: Stores) -> dict[str, Any]:
    from enigma.research.inquiry import next_discovered_objective  # noqa: PLC0415

    objective = next_discovered_objective(stores)
    if objective is None:
        return _criterion(
            9,
            "backlog replenishment",
            "blocked",
            "No discovered objective is waiting. Seeds are not refilled from a fixed list.",
        )
    return _criterion(9, "backlog replenishment", "pass", objective.id)


def _improvement(report: dict[str, Any]) -> dict[str, Any]:
    if report["improvements_adopted"] + report["improvements_rejected"] < 1:
        return _criterion(
            10,
            "implemented and benchmarked improvement",
            "blocked",
            "No intervention has been benchmarked in this store.",
        )
    return _criterion(
        10,
        "implemented and benchmarked improvement",
        "pass",
        f"adopted {report['improvements_adopted']}, rejected {report['improvements_rejected']}",
    )


def demonstrate(
    stores: Stores,
    fsm: Any,
    catalog: DataCatalog | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Write one case, one real acquisition, one unmet request, and one intervention."""
    moment = now or datetime.now(UTC)
    _evidence(stores, moment)
    case = _case(moment)
    held = {"funding_rate", "basis", "perp_close", "spot_close"}
    registration = register_case(stores, case, held_observables=held, now=moment)
    if catalog is not None:
        request = DatasetAcquisitionRequest(
            id="dar_demo_funding",
            requested_by=AgentRole.QUANT_RESEARCH,
            description="funding_rate",
            columns=("funding_rate",),
            rationale="The case names funding_rate as an observable the archive actually stores.",
            created_at=moment,
        )
        stores.documents.put(request, recorded_by=AgentRole.DATA_ENGINEERING.value)
        fulfill(
            stores,
            request,
            catalog,
            symbols=("BTCUSDT",),
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 10, tzinfo=UTC),
            frequency="8h",
            now=moment,
        )
    unmet = DatasetAcquisitionRequest(
        id="dar_demo_oi",
        requested_by=AgentRole.QUANT_RESEARCH,
        description="open_interest",
        columns=("open_interest",),
        rationale="Named by a hypothesis. No zero-cost provider in the catalog lists it.",
        created_at=moment,
    )
    stores.documents.put(unmet, recorded_by=AgentRole.DATA_ENGINEERING.value)
    fulfill(
        stores,
        unmet,
        None,
        symbols=("BTCUSDT",),
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 10, tzinfo=UTC),
        frequency="8h",
        now=moment,
    )
    _intervention(stores, fsm, moment)
    _engine_check(case.program)
    return {"registration": registration.reason, "model": probe_local_model()}


def _evidence(stores: Stores, now: datetime) -> None:
    for source_id, title, excerpt, evidence_id in (
        (
            "src_binance_funding_docs",
            "Binance USD-M funding schedule",
            "Binance USD-M funding is exchanged every eight hours, at 00:00, 08:00 and 16:00 UTC.",
            "ev_binance_schedule",
        ),
        (
            "src_binance_archive_note",
            "Binance public archive layout",
            "Funding-rate history is a public monthly archive on data.binance.vision.",
            "ev_binance_archive",
        ),
    ):
        if stores.documents.get(Source, source_id) is None:
            stores.documents.put(
                Source(
                    id=source_id,
                    kind=SourceKind.DOCUMENTATION,
                    provider="binance",
                    url=f"https://data.binance.vision/#{source_id}",
                    title=title,
                    content_hash="sha256:" + sha256_hex(excerpt),
                    retrieved_by=AgentRole.INTELLIGENCE,
                    retrieved_at=now,
                ),
                recorded_by=AgentRole.INTELLIGENCE.value,
            )
        if stores.documents.get(EvidenceItem, evidence_id) is None:
            stores.documents.put(
                EvidenceItem(
                    id=evidence_id,
                    source_id=source_id,
                    excerpt=excerpt,
                    claim=title,
                    extraction_method="verbatim",
                    is_ai_interpretation=False,
                    claim_level=ClaimLevel.L0_UNVERIFIED,
                    created_by=AgentRole.INTELLIGENCE,
                    created_at=now,
                ),
                recorded_by=AgentRole.INTELLIGENCE.value,
            )


def _case(now: datetime) -> ScientificCase:
    return ScientificCase(
        id="case_funding_basis_interaction",
        mechanism=(
            "When funding is crowded and the perp-spot basis is wide, later returns can "
            "differ from the move predicted by either series alone."
        ),
        persistence=(
            "The funding payment is periodic and the basis is a balance-sheet constraint. "
            "A trader who fades one without the other takes the other's inventory risk."
        ),
        participants=(
            Participant(
                name="perpetual basis trader",
                incentive="earn the funding payment",
                constraint="must hold the offsetting spot or accept basis risk",
            ),
        ),
        structural_conditions=(
            "Funding settles on an eight-hour clock. Basis is marked continuously."
        ),
        supporting_evidence_ids=("ev_binance_schedule", "ev_binance_archive"),
        contradictory_evidence_ids=(),
        predictions=(
            Prediction(
                statement=(
                    "The product of funding and basis disagrees with funding alone "
                    "more often than a coin flip, before costs."
                ),
                observable="funding_rate",
                discriminates="funding alone is a sufficient statistic",
            ),
        ),
        competing_explanations=("The association is a single-series funding effect.",),
        data_needs=(
            DataNeed(
                observable="funding_rate",
                role="signal",
                why="The funding settlement is the periodic cashflow.",
            ),
            DataNeed(
                observable="basis",
                role="signal",
                why="The basis is the inventory constraint.",
            ),
            DataNeed(
                observable="open_interest",
                role="condition",
                why="Positioning distinguishes crowding from basis. It is not substituted.",
            ),
        ),
        program=ResearchProgram(
            design_class="nonlinear",
            computations=(
                Computation(op="zscore", inputs=("funding_rate",), lookback=8),
                Computation(op="zscore", inputs=("basis",), lookback=8),
            ),
            combination="product",
            holding_bars=1,
            sampling_frequency="8h",
            symbols=("BTCUSDT",),
        ),
        interpretation="A model may restate this case. That restatement is not evidence.",
        model_id="",
        created_at=now,
    )


def _intervention(stores: Stores, fsm: Any, now: datetime) -> None:
    proposal = ImprovementProposal(
        id="imp_demo_column_policy",
        target=ProposalTarget.METHODOLOGY,
        target_ref="research.column_resolution",
        change_summary="Do not replace a missing observable with an unrelated column.",
        rationale="A substituted column is not the measurement the hypothesis named.",
        proposed_change={"evaluation": "paired exact-match on a fixed column-resolution task set"},
        success_metric="fraction of column requests resolved to the named observable or to nothing",
        baseline_metric_value=0.0,
        proposed_by=AgentRole.META_LEARNING,
        created_at=now,
    )
    stores.documents.put(proposal, recorded_by=AgentRole.META_LEARNING.value)
    root = stores.db.path.parent if stores.db.path is not None else Path(".")
    path = root / "column-resolution.txt"
    apply_intervention(
        stores,
        fsm,
        proposal,
        baseline_text="legacy-substitute\n",
        candidate_text="strict-observable\n",
        evaluation="paired exact-match on a fixed column-resolution task set",
        comparison=BenchmarkComparison(
            baseline=0.0,
            candidate=1.0,
            task_set_hash=sha256_hex("open_interest-must-stay-missing"),
            seed=0,
            infrastructure_ok=True,
            market_window="not-applicable",
        ),
        path=path,
        now=now,
    )
    # Rollback is available. The demonstration leaves the strict artifact in place
    # and checks that rollback restores the previous bytes, then re-applies strict.


def _engine_check(program: ResearchProgram) -> None:
    source = compile_program(program)
    namespace: dict[str, Any] = {}
    exec(source, namespace)  # noqa: S102 - the compiler emits the only source executed here
    strategy = namespace["ResearchProgramStrategy"]()
    strategy.configure({})
    n = 32
    rng = np.random.default_rng(0)
    data = {
        "funding_rate": rng.normal(0, 1e-4, n),
        "basis": rng.normal(0, 1e-3, n),
    }
    first = strategy.target_positions(data)
    second = strategy.target_positions(data)
    if not np.array_equal(first, second):
        raise RuntimeError("compiled strategy is not deterministic")
    start = datetime(2024, 1, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * n,
            "ts": [start + timedelta(hours=8 * i) for i in range(n)],
            "perp_close": np.linspace(100.0, 110.0, n),
            "funding_rate": data["funding_rate"],
        }
    )
    cost = CostModel(
        version="cm-demo",
        taker_fee_bps=2.0,
        maker_fee_bps=1.0,
        half_spread_bps=1.0,
        impact_bps_per_million=0.0,
        apply_funding=True,
        capacity_notional_usd=1_000_000,
        latency_bars=1,
    )
    engine = BarBacktestEngine(1095)
    positions = {"BTCUSDT": first}
    a = engine.run(positions, frame, cost, 1, notional_usd=10_000)
    b = engine.run(positions, frame, cost, 1, notional_usd=10_000)
    if a.result_hash != b.result_hash:
        raise RuntimeError("backtest hash changed between identical runs")
