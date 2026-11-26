"""Independent validation service (validation plane).

`validate(request)`:
1. Pre-flight (no state change on failure): load package, hypothesis, family experiments, the
   pinned protocol, the sealed holdout manifest and frame, and confirm sandbox isolation.
2. STRATEGY_LIFECYCLE VALIDATION_QUEUED -> VALIDATION_RUNNING (as validation_service).
3. Run the package in the sandbox on the full validation-plane series (with truncation re-runs
   for look-ahead detection), re-apply the package cost model in our own engine (research
   metrics are never read), and compute every gate from numbers.
4. overall = aggregate(gates, protocol.required_gates) -- nothing else can set it.
5. Persist the full report (validation DB + validation-plane document), a redacted summary
   (control plane), FailureConditions for failed gates, audit entries, and move the FSM to
   VALIDATION_PASSED / VALIDATION_FAILED with the report hash.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from enigma.backtest.engine import BacktestResult, BarBacktestEngine, CapacityExceededError
from enigma.backtest.statistics import (
    cscv_pbo,
    deflated_sharpe_ratio,
    holm_adjusted_pvalues,
    one_sided_ttest,
    sharpe_ratio,
    stationary_bootstrap_ci,
    top_k_days_removed_pnl,
)
from enigma.core.clock import Clock, SystemClock
from enigma.core.errors import IntegrityError, InvalidTransitionError, PermissionDeniedError
from enigma.core.hashing import sha256_hex, sha256_of
from enigma.core.ids import new_id
from enigma.fsm.definitions import STRATEGY_LIFECYCLE
from enigma.fsm.machine import StateMachineStore
from enigma.schemas.common import Plane, SystemActor
from enigma.schemas.knowledge import FailureCondition, FailureKind
from enigma.schemas.research import DatasetVersion, Experiment, Hypothesis, StrategyPackage
from enigma.schemas.validation import (
    GateOutcome,
    GateResult,
    HoldoutManifest,
    RedactedValidationSummary,
    ValidationReport,
    ValidationRequest,
    aggregate,
)
from enigma.store.audit import AuditLog
from enigma.store.documents import PlaneView
from enigma.strategy.data import symbol_arrays
from enigma.strategy.interface import ParamValue
from enigma.strategy.leakage import (
    evaluate_truncations,
    pit_timestamp_violations,
    sample_checkpoints,
)
from enigma.strategy.sandbox import SandboxJob, SandboxResult, SandboxRunner
from enigma.strategy.static_check import StrategyRejectedError
from enigma.validation import gates as gate_lib
from enigma.validation.build import validator_build_sha
from enigma.validation.holdout import HoldoutCustody, HoldoutSource, holdout_split_index
from enigma.validation.protocol import ValidationProtocol, load_protocol

ACTOR = SystemActor.VALIDATION_SERVICE.value
TRUNCATION_CHECKPOINTS_PER_SYMBOL = 8
DEFAULT_NOTIONAL_USD = 100_000.0

SummaryWriter = Callable[[tuple[GateResult, ...], GateOutcome], str]


def report_id_for(request_id: str) -> str:
    """One report per validation request, so a retried request finds its existing result."""
    return "vrp_" + sha256_hex(request_id)[:26].upper()


def summary_id_for(request_id: str) -> str:
    return "vsm_" + sha256_hex(request_id)[:26].upper()


def render_summary(gates: tuple[GateResult, ...], overall: GateOutcome) -> str:
    """Deterministic, non-authoritative text. It is written after `overall` is fixed."""
    failed = [f"{g.gate_id}:{g.name}" for g in gates if g.result is GateOutcome.FAIL]
    return f"overall={overall.value}; failed gates: {', '.join(failed) if failed else 'none'}"


@dataclass(frozen=True)
class _Inputs:
    request: ValidationRequest
    package: StrategyPackage
    symbols: tuple[str, ...]
    experiments: tuple[Experiment, ...]
    registration_problems: tuple[str, ...]
    manifest: HoldoutManifest
    frame: pl.DataFrame
    bars: np.ndarray
    holdout_start_ns: int
    embargo_start_ns: int
    n_embargo_bars: int
    n_research_bars: int
    seed: int


class _StrategyFailureError(Exception):
    def __init__(self, kind: FailureKind, reason: str) -> None:
        super().__init__(reason)
        self.kind = kind
        self.reason = reason


def _ns(value: datetime) -> int:
    return int(pl.Series([value]).dt.epoch("ns")[0])


def _dataset_names(symbols: Sequence[str]) -> dict[str, str]:
    return {f"s{i}": sym for i, sym in enumerate(symbols)}


class ValidationService:
    def __init__(
        self,
        validation_dir: Path,
        control_documents: PlaneView,
        holdout_source: HoldoutSource,
        audit: AuditLog,
        fsm_store: StateMachineStore,
        *,
        sandbox: SandboxRunner | None = None,
        clock: Clock | None = None,
        protocol: ValidationProtocol | None = None,
        notional_usd: float = DEFAULT_NOTIONAL_USD,
        frequency: str = "8h",
        summary_writer: SummaryWriter | None = None,
    ) -> None:
        needed_read = {Plane.CONTROL, Plane.RESEARCH}
        needed_write = {Plane.CONTROL, Plane.VALIDATION}
        if not needed_read <= control_documents.readable:
            raise PermissionDeniedError("validation needs read access to control and research")
        if not needed_write <= control_documents.writable:
            raise PermissionDeniedError("validation needs write access to control and validation")
        self.protocol = protocol or load_protocol()
        self.custody = HoldoutCustody(validation_dir)
        self.documents = control_documents
        self.holdout_source = holdout_source
        self.audit = audit
        self.fsm = fsm_store
        self.sandbox = sandbox or SandboxRunner(audit=audit, actor=ACTOR)
        self.clock = clock or SystemClock()
        self.notional_usd = float(notional_usd)
        self.frequency = frequency
        self.summary_writer = summary_writer
        self.engine = BarBacktestEngine(self.protocol.periods_per_year)

    # ---------------------------------------------------------------- public API

    def seal_holdout(self, symbols: Sequence[str], label_horizon_bars: int = 1) -> datetime:
        """Seal (or reuse) the holdout for `symbols`; return the research cutoff.

        Research data must have `ts` strictly before the returned instant (embargo start).
        Only the boundary is returned; no holdout rows leave the validation plane.
        """
        manifest, frame = self.custody.seal(
            symbols,
            self.frequency,
            self.protocol.version,
            float(self.protocol.gate("G02")["holdout_fraction"]),
            self.holdout_source,
            self.clock,
        )
        bars = frame.get_column("ts").unique().sort()
        split = int(bars.search_sorted(manifest.holdout_start))
        embargo = gate_lib.embargo_bars(self.protocol.gate("G02"), label_horizon_bars)
        cutoff = bars[max(split - embargo, 0)]
        assert isinstance(cutoff, datetime)
        return cutoff

    def validate(self, request: ValidationRequest) -> ValidationReport:
        existing = self.custody.get_report(report_id_for(request.id))
        if existing is not None:
            return existing
        inputs = self._preflight(request)
        package = inputs.package
        # A request interrupted mid-run resumes from VALIDATION_RUNNING; gates are deterministic.
        if self.fsm.state(STRATEGY_LIFECYCLE, package.id) != "VALIDATION_RUNNING":
            self.fsm.transition(
                STRATEGY_LIFECYCLE,
                package.id,
                "VALIDATION_RUNNING",
                ACTOR,
                "validation_started",
                {"strategy_package": package.content_hash},
            )
        self.audit.record(
            ACTOR,
            "validation.started",
            package.id,
            {
                "request_id": request.id,
                "package_hash": package.content_hash,
                "protocol_version": self.protocol.version,
                "protocol_sha256": self.protocol.sha256,
            },
        )
        failure: _StrategyFailureError | None = None
        extras: dict[str, float | None] = {}
        try:
            gates, extras = self._evaluate(inputs)
        except _StrategyFailureError as exc:
            failure = exc
        except Exception as exc:  # fail closed: an unevaluable package cannot pass
            failure = _StrategyFailureError(
                FailureKind.OPERATIONAL, f"validator error: {type(exc).__name__}: {exc}"
            )
            self.audit.record(ACTOR, "validation.error", package.id, {"error": failure.reason})
        if failure is not None:
            gates = tuple(
                gate_lib.failed_gate(gid, str(self.protocol.gate(gid)["name"]), failure.reason, {})
                for gid in self.protocol.required_gates
            )
        return self._finish(inputs, gates, extras, failure)

    # ---------------------------------------------------------------- pre-flight

    def _preflight(self, request: ValidationRequest) -> _Inputs:
        package = self.documents.require(StrategyPackage, request.strategy_package_id)
        state = self.fsm.state(STRATEGY_LIFECYCLE, package.id)
        if state not in ("VALIDATION_QUEUED", "VALIDATION_RUNNING"):
            raise InvalidTransitionError(
                f"package {package.id} is in {state}, validation requires VALIDATION_QUEUED"
            )
        hypothesis = self.documents.require(Hypothesis, package.hypothesis_id)
        symbols = tuple(sorted({i.symbol for i in hypothesis.instruments}))
        experiments = tuple(
            sorted(
                self.documents.list(Experiment, where={"family_id": request.family_id}),
                key=lambda e: (e.trial_index, e.id),
            )
        )
        problems: list[str] = []
        family_ids = {e.id for e in experiments}
        problems.extend(
            f"experiment {eid} not registered in family {request.family_id}"
            for eid in request.experiment_ids
            if eid not in family_ids
        )
        if not any(e.strategy_package_id == package.id for e in experiments):
            problems.append(f"package {package.id} has no registered experiment in the family")
        self.sandbox.check_isolation(package.id)
        g02 = self.protocol.gate("G02")
        manifest, frame = self.custody.seal(
            symbols,
            self.frequency,
            self.protocol.version,
            float(g02["holdout_fraction"]),
            self.holdout_source,
            self.clock,
        )
        missing = [c for c in package.required_columns if c not in frame.columns]
        if missing:
            raise IntegrityError(f"holdout frame lacks required columns {missing}")
        bars = np.unique(frame.get_column("ts").dt.epoch("ns").to_numpy().astype(np.int64))
        holdout_start_ns = _ns(manifest.holdout_start)
        split = int(np.searchsorted(bars, holdout_start_ns))
        if split != holdout_split_index(bars.size, manifest.holdout_fraction):
            raise IntegrityError(f"holdout boundary of {manifest.id} no longer matches the data")
        n_embargo = gate_lib.embargo_bars(g02, package.label_horizon_bars)
        research_end_idx = max(split - n_embargo, 0)
        return _Inputs(
            request=request,
            package=package,
            symbols=symbols,
            experiments=experiments,
            registration_problems=tuple(problems),
            manifest=manifest,
            frame=frame,
            bars=bars,
            holdout_start_ns=holdout_start_ns,
            embargo_start_ns=int(bars[research_end_idx]),
            n_embargo_bars=split - research_end_idx,
            n_research_bars=research_end_idx,
            seed=int(package.content_hash.removeprefix("sha256:")[:8], 16),
        )

    # ---------------------------------------------------------------- evaluation

    def _run(
        self,
        package: StrategyPackage,
        frame: pl.DataFrame,
        symbols: Sequence[str],
        *,
        checkpoints: Mapping[str, Sequence[int]] | None = None,
        param_sets: Mapping[str, Mapping[str, ParamValue]] | None = None,
    ) -> SandboxResult:
        names = _dataset_names(symbols)
        datasets = {
            name: symbol_arrays(frame, sym, package.required_columns) for name, sym in names.items()
        }
        sets: list[tuple[str, Mapping[str, ParamValue] | None]] = (
            list(param_sets.items()) if param_sets else [("", None)]
        )
        jobs: list[SandboxJob] = []
        for prefix, params in sets:
            for name in names:
                points = (checkpoints or {}).get(name, ())
                jobs.append(
                    SandboxJob(
                        job_id=f"{prefix}{name}",
                        dataset=name,
                        params=params,
                        truncations=tuple(t + 1 for t in points),
                    )
                )
        try:
            return self.sandbox.run(package, datasets, jobs)
        except StrategyRejectedError as exc:
            raise _StrategyFailureError(FailureKind.IMPLEMENTATION_ERROR, str(exc)) from exc

    def _backtest(
        self,
        package: StrategyPackage,
        frame: pl.DataFrame,
        symbols: Sequence[str],
        positions: Mapping[str, np.ndarray],
        seed: int,
        prefix: str = "",
        cost_multiplier: float = 1.0,
    ) -> BacktestResult:
        names = _dataset_names(symbols)
        by_symbol = {sym: positions[f"{prefix}{name}"] for name, sym in names.items()}
        try:
            return self.engine.run(
                by_symbol,
                frame,
                package.cost_model,
                seed,
                notional_usd=self.notional_usd,
                cost_multiplier=cost_multiplier,
            )
        except CapacityExceededError as exc:
            raise _StrategyFailureError(FailureKind.CAPACITY, str(exc)) from exc

    def _evaluate(self, x: _Inputs) -> tuple[tuple[GateResult, ...], dict[str, float | None]]:
        p = self.protocol
        package, frame, symbols, seed = x.package, x.frame, x.symbols, x.seed
        names = _dataset_names(symbols)
        # Same seed for every symbol: aligned symbols share checkpoints, so the child's
        # shortest-first ordering never shows one symbol's later bars before another's earlier.
        checkpoints = {
            name: sample_checkpoints(
                frame.filter(pl.col("symbol") == sym).height,
                TRUNCATION_CHECKPOINTS_PER_SYMBOL,
                seed,
            )
            for name, sym in names.items()
        }

        run_a = self._run(package, frame, symbols, checkpoints=checkpoints)
        if not run_a.ok:
            raise _StrategyFailureError(
                FailureKind.IMPLEMENTATION_ERROR,
                f"strategy failed in sandbox: {run_a.error}"
                + (f"; policy violations: {len(run_a.violations)}" if run_a.violations else ""),
            )
        full = self._backtest(package, frame, symbols, run_a.positions, seed)
        holdout = full.window(x.holdout_start_ns, None)
        research = full.window(None, x.embargo_start_ns)
        hm = holdout.metrics
        ppy = p.periods_per_year

        g01 = gate_lib.g01_pit_integrity(
            p.gate("G01"), pit_timestamp_violations(frame), evaluate_truncations(run_a, checkpoints)
        )
        g02 = gate_lib.g02_chronological_split(
            p.gate("G02"),
            n_bars_total=int(x.bars.size),
            n_holdout_bars=int(x.bars.size - np.searchsorted(x.bars, x.holdout_start_ns)),
            n_embargo_bars=x.n_embargo_bars,
            n_research_bars=x.n_research_bars,
            overlaps=self._research_overlaps(x),
        )
        g03_cfg = p.gate("G03")
        g03 = gate_lib.g03_out_of_sample(
            g03_cfg,
            net_sharpe=hm["net_sharpe"],
            n_bars=int(hm["n_bars"]),
            top_days_removed_pnl=top_k_days_removed_pnl(
                holdout.net_returns, holdout.ts, int(g03_cfg["top_days_removed"])
            ),
        )

        trials = self._trial_returns(x, research)
        n_trials = max(len(x.experiments), 1)
        g06 = self._g06(x, holdout, trials)
        trial_srs = [sharpe_ratio(r) for r in trials.values() if r is not None]
        null_var = 1.0 / max(research.ts.size, 1)
        if len(trial_srs) >= 2:
            variance = max(float(np.var(trial_srs, ddof=1)), null_var)
            source = "max(cross-trial variance of recomputed research Sharpes, 1/T_research)"
        else:
            variance = null_var
            source = "1/T_research (fewer than two recomputable trials)"
        dsr = deflated_sharpe_ratio(holdout.net_returns, n_trials, variance)
        g07 = gate_lib.g07_deflated_sharpe(p.gate("G07"), dsr, source)

        g12_cfg = p.gate("G12")
        stressed = self._backtest(
            package,
            frame,
            symbols,
            run_a.positions,
            seed,
            cost_multiplier=float(g12_cfg["cost_stress_multiplier"]),
        ).window(x.holdout_start_ns, None)
        g12 = gate_lib.g12_costs(
            g12_cfg,
            stressed_net_sharpe=stressed.metrics["net_sharpe"],
            base_net_sharpe=hm["net_sharpe"],
            cost_model_version=package.cost_model.version,
            cost_pnl_base=hm["cost_pnl"],
            cost_pnl_stressed=stressed.metrics["cost_pnl"],
        )
        g16 = self._g16(x, run_a, holdout)

        informative = self._informative(x, full, holdout, research, trials)
        pbo_gate = next(g for g in informative if g.gate_id == "G08")
        gates = tuple(
            sorted((g01, g02, g03, g06, g07, g12, g16, *informative), key=lambda g: g.gate_id)
        )
        extras = {
            "deflated_sharpe": dsr.dsr,
            "pbo": pbo_gate.stat,
            "net_sharpe_holdout": hm["net_sharpe"],
            "periods_per_year": ppy,
        }
        return gates, extras

    def _research_overlaps(self, x: _Inputs) -> list[dict[str, Any]]:
        overlaps: list[dict[str, Any]] = []
        for exp in x.experiments:
            end_ns = _ns(exp.research_window[1])
            if end_ns > x.embargo_start_ns:
                overlaps.append({"kind": "experiment_window_overlaps_embargo", "id": exp.id})
        dataset_ids = {x.request.dataset_version_id, *(e.dataset_version_id for e in x.experiments)}
        for dataset_id in sorted(dataset_ids):
            dataset = self.documents.get(DatasetVersion, dataset_id)
            if dataset is not None and _ns(dataset.time_end) >= x.embargo_start_ns:
                overlaps.append({"kind": "research_dataset_overlaps_embargo", "id": dataset_id})
        return overlaps

    def _trial_returns(
        self, x: _Inputs, candidate_research: BacktestResult
    ) -> dict[str, np.ndarray | None]:
        """Research-window net returns per family package, recomputed by validation."""
        research_frame = x.frame.filter(pl.col("ts").dt.epoch("ns") < x.embargo_start_ns)
        returns: dict[str, np.ndarray | None] = {x.package.id: candidate_research.net_returns}
        for package_id in sorted({e.strategy_package_id for e in x.experiments}):
            if package_id in returns:
                continue
            trial = self.documents.get(StrategyPackage, package_id)
            if trial is None or research_frame.is_empty():
                returns[package_id] = None
                continue
            try:
                result = self._run(trial, research_frame, x.symbols)
                if not result.ok:
                    returns[package_id] = None
                    continue
                bt = self._backtest(trial, research_frame, x.symbols, result.positions, x.seed)
                returns[package_id] = bt.net_returns
            except (_StrategyFailureError, IntegrityError, ValueError, KeyError):
                returns[package_id] = None
        return returns

    def _g06(
        self, x: _Inputs, holdout: BacktestResult, trials: Mapping[str, np.ndarray | None]
    ) -> GateResult:
        _, candidate_p = one_sided_ttest(holdout.net_returns)
        pvalues: list[float] = []
        candidate_idx: list[int] = []
        for exp in x.experiments:
            if exp.strategy_package_id == x.package.id:
                candidate_idx.append(len(pvalues))
                pvalues.append(candidate_p)
            else:
                r = trials.get(exp.strategy_package_id)
                pvalues.append(1.0 if r is None else one_sided_ttest(r)[1])
        if not candidate_idx:
            candidate_idx.append(len(pvalues))
            pvalues.append(candidate_p)
        adjusted = holm_adjusted_pvalues(pvalues)
        return gate_lib.g06_multiple_testing(
            self.protocol.gate("G06"),
            candidate_adjusted_p=float(max(adjusted[i] for i in candidate_idx)),
            candidate_raw_p=candidate_p,
            family_pvalues=pvalues,
            registered=not x.registration_problems,
            registration_problems=x.registration_problems,
        )

    def _g16(self, x: _Inputs, run_a: SandboxResult, holdout_a: BacktestResult) -> GateResult:
        cfg = self.protocol.gate("G16")
        run_b = self._run(x.package, x.frame, x.symbols)
        if not run_b.ok:
            return gate_lib.failed_gate(
                "G16", str(cfg["name"]), f"twin run failed: {run_b.error}", cfg
            )
        position_delta = max(
            float(np.max(np.abs(run_a.positions[k] - run_b.positions[k]), initial=0.0))
            if run_a.positions[k].shape == run_b.positions[k].shape
            else math.inf
            for k in run_b.positions
        )
        full_b = self._backtest(x.package, x.frame, x.symbols, run_b.positions, x.seed)
        holdout_b = full_b.window(x.holdout_start_ns, None)
        metric_delta = max(
            abs(holdout_a.metrics[k] - holdout_b.metrics[k]) for k in holdout_a.metrics
        )
        return gate_lib.g16_reproducibility(
            cfg,
            max_abs_delta=max(position_delta, metric_delta),
            position_delta=position_delta,
            metric_delta=metric_delta,
            hashes_equal=holdout_a.result_hash == holdout_b.result_hash,
        )

    def _informative(
        self,
        x: _Inputs,
        full: BacktestResult,
        holdout: BacktestResult,
        research: BacktestResult,
        trials: Mapping[str, np.ndarray | None],
    ) -> list[GateResult]:
        ppy = self.protocol.periods_per_year
        candidate_sharpe = holdout.metrics["net_sharpe"]

        folds = np.array_split(full.net_returns, int(gate_lib.INFORMATIVE_GATES["G04"]["n_folds"]))
        g04 = gate_lib.g04_walk_forward([sharpe_ratio(f, ppy) for f in folds if f.size > 1])

        cfg08 = gate_lib.INFORMATIVE_GATES["G08"]
        matrix_cols = [r for r in trials.values() if r is not None]
        if len(matrix_cols) < 2:
            g08 = gate_lib.g08_pbo(None, "fewer than two recomputable strategies in the family")
        else:
            try:
                g08 = gate_lib.g08_pbo(
                    cscv_pbo(np.column_stack(matrix_cols), int(cfg08["n_splits"]))
                )
            except ValueError as exc:
                g08 = gate_lib.g08_pbo(None, str(exc))

        g09 = gate_lib.g09_parameter_sensitivity(candidate_sharpe, self._neighbours(x))
        g10 = gate_lib.g10_regime_stress(self._regimes(x, holdout))

        cfg11 = gate_lib.INFORMATIVE_GATES["G11"]
        ci = stationary_bootstrap_ci(
            holdout.net_returns,
            lambda r: sharpe_ratio(r, ppy),
            n_boot=int(cfg11["n_boot"]),
            mean_block=float(cfg11["mean_block_bars"]),
            alpha=float(cfg11["alpha"]),
            seed=x.seed,
        )
        g11 = gate_lib.g11_bootstrap(ci, x.seed)

        n_by_symbol = {sym: x.frame.filter(pl.col("symbol") == sym).height for sym in x.symbols}
        buy_hold = self.engine.run(
            {sym: np.ones(n) for sym, n in n_by_symbol.items()},
            x.frame,
            x.package.cost_model,
            x.seed,
            notional_usd=self.notional_usd,
        ).window(x.holdout_start_ns, None)
        g15 = gate_lib.g15_benchmark(
            candidate_sharpe, {"flat": 0.0, "buy_and_hold_perp": buy_hold.metrics["net_sharpe"]}
        )
        return [g04, g08, g09, g10, g11, g15]

    def _neighbours(self, x: _Inputs) -> dict[str, float | None]:
        step = float(gate_lib.INFORMATIVE_GATES["G09"]["relative_step"])
        param_sets: dict[str, dict[str, ParamValue]] = {}
        labels: dict[str, str] = {}
        for key in sorted(x.package.params):
            value = x.package.params[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            if isinstance(value, int):
                delta_i = max(1, round(abs(value) * step))
                candidates: list[ParamValue] = [value - delta_i, value + delta_i]
            else:
                delta_f = abs(value) * step if value != 0 else step
                candidates = [value - delta_f, value + delta_f]
            for j, candidate in enumerate(candidates):
                prefix = f"p{len(param_sets)}_"
                param_sets[prefix] = {**x.package.params, key: candidate}
                labels[prefix] = f"{key}{'-' if j == 0 else '+'}"
        if not param_sets:
            return {}
        result = self._run(x.package, x.frame, x.symbols, param_sets=param_sets)
        out: dict[str, float | None] = {}
        for prefix, label in labels.items():
            if not result.ok:
                out[label] = None
                continue
            try:
                bt = self._backtest(
                    x.package, x.frame, x.symbols, result.positions, x.seed, prefix=prefix
                )
            except _StrategyFailureError:
                out[label] = None
                continue
            out[label] = bt.window(x.holdout_start_ns, None).metrics["net_sharpe"]
        return out

    def _regimes(self, x: _Inputs, holdout: BacktestResult) -> dict[str, dict[str, float]]:
        """Holdout net Sharpe by realized-vol tercile. Vol at bar t uses closes up to t-1;
        tercile cut points come from the research period only."""
        window = int(gate_lib.INFORMATIVE_GATES["G10"]["vol_window_bars"])
        vol = (
            x.frame.sort(["symbol", "ts"])
            .with_columns(
                pl.col("perp_close").log().diff().over("symbol").alias("_lr"),
            )
            .with_columns(
                pl.col("_lr").rolling_std(window_size=window).shift(1).over("symbol").alias("_vol")
            )
            .group_by("ts")
            .agg(pl.col("_vol").mean())
            .sort("ts")
        )
        ts = vol.get_column("ts").dt.epoch("ns").to_numpy().astype(np.int64)
        values = vol.get_column("_vol").fill_null(float("nan")).to_numpy().astype(np.float64)
        research_vals = values[(ts < x.embargo_start_ns) & np.isfinite(values)]
        if research_vals.size < 3:
            return {}
        lo, hi = np.quantile(research_vals, [1.0 / 3.0, 2.0 / 3.0])
        lookup = dict(zip(ts.tolist(), values.tolist(), strict=True))
        hv = np.array([lookup.get(int(t), math.nan) for t in holdout.ts])
        regimes: dict[str, dict[str, float]] = {}
        for label, mask in (
            ("low_vol", hv <= lo),
            ("mid_vol", (hv > lo) & (hv <= hi)),
            ("high_vol", hv > hi),
        ):
            r = holdout.net_returns[mask & np.isfinite(hv)]
            regimes[label] = {
                "bars": float(r.size),
                "net_sharpe": sharpe_ratio(r, self.protocol.periods_per_year),
            }
        return regimes

    # ---------------------------------------------------------------- persistence

    def _finish(
        self,
        x: _Inputs,
        gates: tuple[GateResult, ...],
        extras: Mapping[str, float | None],
        failure: _StrategyFailureError | None,
    ) -> ValidationReport:
        package = x.package
        overall = aggregate(gates, self.protocol.required_gates)
        writer = self.summary_writer or render_summary
        report = ValidationReport(
            id=report_id_for(x.request.id),
            request_id=x.request.id,
            strategy_package_id=package.id,
            protocol_version=self.protocol.version,
            required_gates=self.protocol.required_gates,
            gates=gates,
            overall=overall,
            holdout_dataset_ids=x.manifest.dataset_ids,
            trial_count=max(len(x.experiments), 1),
            deflated_sharpe=gate_lib.clean(extras.get("deflated_sharpe")),
            pbo=gate_lib.clean(extras.get("pbo")),
            net_sharpe_holdout=gate_lib.clean(extras.get("net_sharpe_holdout")),
            cost_assumptions_hash=sha256_of(
                {
                    "cost_model": package.cost_model,
                    "notional_usd": self.notional_usd,
                    "periods_per_year": self.protocol.periods_per_year,
                }
            ),
            validator_build_sha=validator_build_sha(),
            summary=writer(gates, overall),
            created_at=self.clock.now(),
        )
        report_hash = sha256_of(report)
        self.custody.store_report(report)
        self.documents.put(report)
        summary = RedactedValidationSummary(
            id=summary_id_for(x.request.id),
            report_id=report.id,
            strategy_package_id=package.id,
            protocol_version=report.protocol_version,
            overall=report.overall,
            gate_outcomes={g.gate_id: g.result for g in gates},
            failed_gate_names=tuple(
                f"{g.gate_id}:{g.name}" for g in gates if g.result is GateOutcome.FAIL
            ),
        )
        self.documents.put(summary)
        required = set(self.protocol.required_gates)
        for gate in gates:
            if gate.result is GateOutcome.PASS:
                continue
            kind = failure.kind if failure is not None else gate_lib.GATE_FAILURE_KIND[gate.gate_id]
            scope = "required" if gate.gate_id in required else "informative"
            reason = gate.details.get("reason")
            self.documents.put(
                FailureCondition(
                    id=new_id("fc"),
                    kind=kind,
                    description=(
                        f"{gate.gate_id} {gate.name} ({scope}) failed: "
                        + (
                            str(reason)
                            if reason
                            else f"stat={gate.stat} {gate.comparator} threshold={gate.threshold}"
                            " not met"
                        )
                    )[:1000],
                    subject_id=package.id,
                    gate_id=gate.gate_id,
                    created_by=SystemActor.VALIDATION_SERVICE,
                    created_at=self.clock.now(),
                )
            )
        target = "VALIDATION_PASSED" if overall is GateOutcome.PASS else "VALIDATION_FAILED"
        self.fsm.transition(
            STRATEGY_LIFECYCLE,
            package.id,
            target,
            ACTOR,
            "gates_passed" if overall is GateOutcome.PASS else "gates_failed",
            {"validation_report": report_hash},
        )
        self.audit.record(
            ACTOR,
            "validation.completed",
            package.id,
            {
                "report_id": report.id,
                "report_hash": report_hash,
                "overall": overall.value,
                "failed_gates": list(summary.failed_gate_names),
                "validator_build_sha": report.validator_build_sha,
            },
        )
        return report
