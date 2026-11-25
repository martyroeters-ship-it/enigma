"""Validation test environment: control store, FSM, validation service on synthetic data."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from _synthetic import START
from enigma.core.clock import FrozenClock
from enigma.core.ids import new_id
from enigma.fsm import STRATEGY_LIFECYCLE, StateMachineStore
from enigma.schemas.common import AgentRole, Instrument, Plane
from enigma.schemas.research import Experiment, Hypothesis, StrategyPackage
from enigma.schemas.validation import ValidationRequest
from enigma.store import PlaneView, Stores
from enigma.validation import InMemoryHoldoutSource, ValidationService

NOW = datetime(2026, 11, 25, 12, tzinfo=UTC)


@dataclass
class Env:
    stores: Stores
    service: ValidationService
    research: PlaneView
    control: PlaneView
    fsm: StateMachineStore
    validation_dir: Path
    frame: pl.DataFrame

    def submit(
        self,
        pkg: StrategyPackage,
        *,
        family: str = "fam_test",
        other_trials: Sequence[StrategyPackage] = (),
        window_end: datetime | None = None,
        register: bool = True,
    ) -> ValidationRequest:
        symbols = sorted(self.frame.get_column("symbol").unique().to_list())
        if self.research.get(Hypothesis, pkg.hypothesis_id) is None:
            self.research.put(_hypothesis(pkg.hypothesis_id, family, symbols))
        cutoff = self.service.seal_holdout(symbols, pkg.label_horizon_bars)
        exp_ids: list[str] = []
        for index, trial in enumerate((pkg, *other_trials), start=1):
            if self.research.get(StrategyPackage, trial.id) is None:
                self.research.put(trial)
            if not register and trial.id == pkg.id:
                continue
            exp = Experiment(
                id=f"exp_{trial.id}",
                hypothesis_id=trial.hypothesis_id,
                family_id=family,
                trial_index=index,
                strategy_package_id=trial.id,
                dataset_version_id="ds_research_test",
                research_window=(START, window_end or cutoff),
                code_sha="test",
                config_hash=trial.content_hash,
                seed=0,
                created_by=AgentRole.QUANT_RESEARCH,
            )
            self.research.put(exp)
            if trial.id == pkg.id:
                exp_ids.append(exp.id)
        queue_package(self.fsm, pkg)
        return ValidationRequest(
            id=new_id("vrq"),
            strategy_package_id=pkg.id,
            experiment_ids=tuple(exp_ids) or (f"exp_{pkg.id}",),
            family_id=family,
            dataset_version_id="ds_research_test",
            requested_by="cio",
            requested_at=NOW,
        )


def queue_package(fsm: StateMachineStore, pkg: StrategyPackage) -> None:
    fsm.create(STRATEGY_LIFECYCLE, pkg.id, "quant_research")
    fsm.transition(
        STRATEGY_LIFECYCLE, pkg.id, "RESEARCH_EXPERIMENT_RUNNING", "quant_research", "go"
    )
    fsm.transition(
        STRATEGY_LIFECYCLE,
        pkg.id,
        "RESEARCH_CANDIDATE",
        "strategy_engineering",
        "ok",
        {"experiment_result": "sha256:" + "0" * 64},
    )
    fsm.transition(
        STRATEGY_LIFECYCLE,
        pkg.id,
        "VALIDATION_QUEUED",
        "cio",
        "submit",
        {"strategy_package": pkg.content_hash},
    )


def _hypothesis(hyp_id: str, family: str, symbols: Sequence[str]) -> Hypothesis:
    return Hypothesis(
        id=hyp_id,
        objective_id="obj_test",
        family_id=family,
        title="synthetic test hypothesis",
        mechanism="synthetic data generated for tests",
        prediction="test",
        formal_statement="E[r_{t+1} | basis_t] != 0",
        falsification_criteria=("holdout net Sharpe < 0.5",),
        instruments=tuple(
            Instrument(symbol=s, base=s[:-4], quote="USDT", kind="perpetual", venue="binance_usdm")
            for s in symbols
        ),
        horizon="8h",
        features=("basis",),
        trial_budget=20,
        cost_model_version="cm-test-1",
        generation_method="template:test",
        methodology_version="test",
        created_by=AgentRole.QUANT_RESEARCH,
        created_at=NOW,
    )


EnvFactory = Callable[..., Env]


def build_env(root: Path, frame: pl.DataFrame) -> Env:
    stores = Stores.open(root / "control" / "enigma.sqlite3")
    fsm = StateMachineStore(stores.db, stores.audit)
    validation_view = PlaneView(
        stores.documents,
        readable=frozenset({Plane.CONTROL, Plane.RESEARCH, Plane.VALIDATION}),
        writable=frozenset({Plane.CONTROL, Plane.VALIDATION}),
        actor="validation_service",
    )
    research_view = PlaneView(
        stores.documents,
        readable=frozenset({Plane.CONTROL, Plane.RESEARCH}),
        writable=frozenset({Plane.CONTROL, Plane.RESEARCH}),
        actor="quant_research",
    )
    validation_dir = root / "validation"
    service = ValidationService(
        validation_dir,
        validation_view,
        InMemoryHoldoutSource(frame, ("dsv_validation_test",)),
        stores.audit,
        fsm,
        clock=FrozenClock(NOW),
    )
    return Env(stores, service, research_view, validation_view, fsm, validation_dir, frame)
