"""Research-cycle rig with TEST DOUBLES for the external services.

The CIO driver, role agents, worker, tool gateway, FSMs, stores, bus, risk service and core
tools are the real implementations; only the research engine, data catalog, sandbox/backtest
and validation service are replaced by the doubles below, which never claim to be operational.
The real services are exercised end-to-end by tests/network and `enigma research`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from enigma.agents import AgentWorker, ContractRegistry
from enigma.comm import MessageBus
from enigma.core.clock import SystemClock
from enigma.core.config import Settings
from enigma.core.hashing import sha256_of
from enigma.fsm import RESEARCH_CYCLE, STRATEGY_LIFECYCLE, StateMachineStore
from enigma.orchestrator.bindings import CoreTools, Services, build_registry
from enigma.orchestrator.cycle import CycleDriver, HoldoutPolicy
from enigma.orchestrator.roles import ALL_ROLE_AGENTS
from enigma.risk.service import RiskLimits, RiskService
from enigma.schemas.common import AgentRole, Instrument, Plane, SystemActor
from enigma.schemas.knowledge import EvidenceItem, Source, SourceKind
from enigma.schemas.research import (
    CostModel,
    DataQualityCheck,
    DataQualityReport,
    DatasetVersion,
    StrategyPackage,
)
from enigma.schemas.validation import (
    GateOutcome,
    GateResult,
    RedactedValidationSummary,
    ValidationReport,
    ValidationRequest,
)
from enigma.store import Stores

ROOT = Path(__file__).resolve().parents[2]


def isolated_settings(db: Path) -> Settings:
    """Settings whose home is the test directory, not the process environment."""
    home = db.parent
    return Settings(
        home=home,
        research_dir=home / "research",
        validation_dir=home / "validation",
        execution_dir=home / "execution",
        db_path=db,
        registries_dir=ROOT / "registries",
    )
COST = CostModel(
    version="cm-test",
    taker_fee_bps=5,
    maker_fee_bps=2,
    half_spread_bps=1,
    impact_bps_per_million=2,
    capacity_notional_usd=5e6,
)
RESEARCH_CUTOFF = datetime(2024, 6, 1, tzinfo=UTC)
CLAIMS = (
    "Funding rates in perpetual futures reflect crowding of leveraged longs; extreme positive "
    "funding precedes reversal as crowded positioning unwinds.",
    "High funding rate periods are associated with lower subsequent perpetual returns, "
    "consistent with crowding and leverage reversal.",
    "Basis between perpetual and spot mean reverts with a short half-life.",
)


class FakeServices:
    """Test doubles for the plane services, recording calls."""

    def __init__(self, stores: Stores, fail_gate: str | None = "costs") -> None:
        self.s = stores
        self.frames: dict[str, pl.DataFrame] = {}
        self.calls: list[str] = []
        self.fail_gate = fail_gate

    def intel_research(
        self, topic: str, keywords: list[str], max_sources: int, depth: int, budget: Any
    ) -> dict[str, Any]:
        self.calls.append("intel")
        now = datetime.now(UTC)
        src_ids, ev_ids = [], []
        for i, claim in enumerate(CLAIMS):
            src = Source(
                id=f"src_t{i}",
                kind=SourceKind.PREPRINT,
                provider="test-double",
                url=f"https://example.invalid/{i}",
                title=f"Doc {i}",
                content_hash=sha256_of(claim),
                retrieved_by=AgentRole.INTELLIGENCE,
                retrieved_at=now,
            )
            ev = EvidenceItem(
                id=f"evi_t{i}",
                source_id=src.id,
                excerpt=claim,
                claim=claim,
                topics=("funding",),
                extraction_method="test-double",
                created_by=AgentRole.INTELLIGENCE,
                created_at=now,
            )
            self.s.documents.put(src, recorded_by="intelligence")
            self.s.documents.put(ev, recorded_by="intelligence")
            self.s.documents.index_text(ev.id, "evidence", Plane.RESEARCH, claim[:60], claim, ())
            src_ids.append(src.id)
            ev_ids.append(ev.id)
        return {
            "source_ids": src_ids,
            "evidence_ids": ev_ids,
            "providers": {"test-double": 3},
            "queries": [topic],
            "duplicates": 0,
        }

    def data_acquire_research(
        self, symbols: list[str], start: datetime, end: datetime, frequency: str, budget: Any
    ) -> dict[str, Any]:
        self.calls.append("data")
        rng = np.random.default_rng(7)
        rows = []
        n = 900
        for sym in symbols:
            ts = [start + timedelta(hours=8 * k) for k in range(n)]
            price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
            funding = rng.normal(1e-4, 2e-4, n)
            for k in range(n):
                rows.append(
                    {
                        "symbol": sym,
                        "ts": ts[k],
                        "available_at": ts[k] + timedelta(hours=8),
                        "perp_close": float(price[k]),
                        "spot_close": float(price[k] * 0.999),
                        "funding_rate": float(funding[k]),
                        "basis": 0.001 + float(rng.normal(0, 5e-4)),
                    }
                )
        frame = pl.DataFrame(rows)
        ds = DatasetVersion(
            id="dsv_test",
            name="test-double-panel",
            instruments=tuple(
                Instrument(symbol=s, base=s[:-4], quote="USDT", kind="perpetual", venue="test")
                for s in symbols
            ),
            venue="test",
            frequency=frequency,
            schema_version="t",
            columns=tuple(frame.columns),
            time_start=start,
            time_end=end,
            row_count=frame.height,
            content_hash=sha256_of(rows[:3]),
            uri="memory://",
            pit_certified=True,
            quality=DataQualityReport(checks=(DataQualityCheck(name="pit", passed=True),)),
            provenance={"test_double": True},
            created_by=AgentRole.DATA_ENGINEERING,
            created_at=datetime.now(UTC),
        )
        self.s.documents.put(ds, recorded_by="data_engineering")
        self.frames[ds.id] = frame
        return {
            "dataset_version_id": ds.id,
            "content_hash": ds.content_hash,
            "pit_certified": True,
            "rows": frame.height,
            "failed_checks": [],
        }

    def data_load_research(self, dataset_version_id: str) -> pl.DataFrame:
        return self.frames[dataset_version_id]

    def strategy_build(
        self,
        source_code: str,
        entrypoint: str,
        params: dict[str, Any],
        hypothesis_id: str,
        name: str,
        required_columns: tuple[str, ...],
        cost_model_version: str,
    ) -> StrategyPackage:
        h = sha256_of([source_code, params, hypothesis_id])
        return StrategyPackage(
            id="pkg_" + h[7:31],
            hypothesis_id=hypothesis_id,
            name=name,
            source_code=source_code,
            entrypoint=entrypoint,
            params=params,
            required_columns=tuple(required_columns),
            content_hash=h,
            cost_model=COST,
            created_by=AgentRole.STRATEGY_ENGINEERING,
            created_at=datetime.now(UTC),
        )

    def backtest_run(
        self, package_id: str, dataset_version_id: str, seed: int, budget: Any = None
    ) -> dict[str, Any]:
        self.calls.append("backtest")
        score = int(sha256_of(package_id)[7:15], 16) / 0xFFFFFFFF
        metrics = {
            "net_sharpe": score - 0.3,
            "gross_sharpe": score,
            "max_drawdown": 0.2,
            "annual_turnover": 40.0,
            "annual_return": 0.05,
        }
        return {
            "metrics": metrics,
            "result_hash": sha256_of([package_id, seed, metrics]),
            "artifact_uri": None,
            # Resource usage varies between identical runs; it must not count as a result.
            "resources": {"sandbox_cpu_seconds": float(len(self.calls))},
        }

    def validation_run(self, request_id: str) -> dict[str, Any]:
        self.calls.append("validation")
        fsm = StateMachineStore(self.s.db, self.s.audit)
        req = self.s.documents.require(ValidationRequest, request_id)
        val = SystemActor.VALIDATION_SERVICE
        if fsm.state(STRATEGY_LIFECYCLE, req.strategy_package_id) == "VALIDATION_QUEUED":
            fsm.transition(
                STRATEGY_LIFECYCLE, req.strategy_package_id, "VALIDATION_RUNNING", val, "started"
            )
        names = {"G01": "pit_integrity", "G12": "costs"}
        gates = tuple(
            GateResult(
                gate_id=g,
                name=n,
                stat=1.0,
                threshold=0.0,
                comparator=">=",
                result=GateOutcome.FAIL if n == self.fail_gate else GateOutcome.PASS,
            )
            for g, n in names.items()
        )
        overall = GateOutcome.FAIL if self.fail_gate else GateOutcome.PASS
        report = ValidationReport(
            id="vrp_" + request_id[4:],
            request_id=request_id,
            strategy_package_id=req.strategy_package_id,
            protocol_version="test",
            required_gates=tuple(names),
            gates=gates,
            overall=overall,
            holdout_dataset_ids=("dsv_holdout",),
            trial_count=len(req.experiment_ids),
            cost_assumptions_hash=sha256_of(COST),
            validator_build_sha="test",
            created_at=datetime.now(UTC),
        )
        self.s.documents.put(report, recorded_by="validation_service")
        summary = RedactedValidationSummary(
            id="vsm_" + request_id[4:],
            report_id=report.id,
            strategy_package_id=req.strategy_package_id,
            protocol_version="test",
            overall=overall,
            gate_outcomes={g.name: g.result for g in gates},
            failed_gate_names=tuple(g.name for g in gates if g.result is GateOutcome.FAIL),
        )
        self.s.documents.put(summary, recorded_by="validation_service")
        target = "VALIDATION_PASSED" if overall is GateOutcome.PASS else "VALIDATION_FAILED"
        fsm.transition(
            STRATEGY_LIFECYCLE,
            req.strategy_package_id,
            target,
            val,
            "decided",
            {"validation_report": report.id},
        )
        return {
            "summary_id": summary.id,
            "overall": overall.value,
            "protocol_version": "test",
            "failed_gates": list(summary.failed_gate_names),
            "report_id": report.id,
        }

    def gateway_status(self) -> dict[str, Any]:
        return {"missing_credentials": ["OPENAI_API_KEY or ANTHROPIC_API_KEY"]}

    def services(self) -> Services:
        return Services(
            self.intel_research,
            self.data_acquire_research,
            self.data_load_research,
            self.strategy_build,
            self.backtest_run,
            self.validation_run,
            self.gateway_status,
            lambda symbols: RESEARCH_CUTOFF,
        )


class Rig:
    def __init__(
        self, db: Path, fakes: FakeServices | None = None, fail_gate: str | None = "costs"
    ):
        self.stores = Stores.open(db)
        self.fakes = fakes or FakeServices(self.stores, fail_gate)
        self.fakes.s = self.stores
        self.bus = MessageBus(self.stores.db, self.stores.audit)
        self.fsm = StateMachineStore(self.stores.db, self.stores.audit)
        self.contracts = ContractRegistry.load(ROOT / "registries" / "agents")
        clock = SystemClock()
        risk = RiskService(
            self.stores, RiskLimits.load(ROOT / "registries/risk/limits.yaml"), clock
        )
        tools = build_registry(CoreTools(self.stores, self.fsm, risk, clock), self.fakes.services())
        agents = {cls.role: cls() for cls in ALL_ROLE_AGENTS}
        self.worker = AgentWorker(self.stores, self.bus, self.contracts, tools, agents)
        self.driver = CycleDriver(
            self.stores,
            self.bus,
            self.fsm,
            self.contracts,
            HoldoutPolicy.load(ROOT / "registries/validation/holdout_policy.yaml"),
            lambda symbols: RESEARCH_CUTOFF,
            clock,
        )

    def submit(self) -> str:
        instruments = tuple(
            Instrument(symbol=s, base=s[:-4], quote="USDT", kind="perpetual", venue="binance_usdm")
            for s in ("BTCUSDT", "ETHUSDT")
        )
        return self.driver.submit_objective(
            "Does perpetual funding-rate crowding predict BTC/ETH reversals net of costs?",
            instruments,
            ("funding rate", "crowding"),
            max_trials=9,
        )

    def run(self, cycle_id: str, until: str | None = None, max_steps: int = 500) -> str:
        for _ in range(max_steps):
            state = self.fsm.state(RESEARCH_CYCLE, cycle_id)
            if state in ("COMPLETED", "ABORTED") or (until and state == until):
                return str(state)
            progressed = self.driver.tick(cycle_id)
            ran = self.worker.run_once()
            if not progressed and ran is None:
                break
        return str(self.fsm.state(RESEARCH_CYCLE, cycle_id))
