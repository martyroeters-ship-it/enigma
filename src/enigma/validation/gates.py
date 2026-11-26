"""Deterministic gate functions. Each returns a GateResult computed from numbers only.

Required gates (G01, G02, G03, G06, G07, G12, G16) take thresholds from the pinned protocol.
Informative gates (G04, G08, G09, G10, G11, G15) are not in `required_gates` for 0.1.0-slice, so
they never change `overall`; their thresholds live in `INFORMATIVE_GATES` below.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from enigma.backtest.statistics import DeflatedSharpe, PBOResult
from enigma.schemas.knowledge import FailureKind
from enigma.schemas.validation import GateOutcome, GateResult
from enigma.strategy.leakage import TruncationCheckResult

INFORMATIVE_GATES: dict[str, dict[str, Any]] = {
    "G04": {"name": "walk_forward", "n_folds": 6, "min_positive_fraction": 0.5},
    "G08": {"name": "pbo", "max_pbo": 0.5, "n_splits": 16},
    "G09": {"name": "parameter_sensitivity", "relative_step": 0.10, "min_ratio": 0.5},
    "G10": {"name": "regime_stress", "vol_window_bars": 21, "min_regime_bars": 30},
    "G11": {"name": "bootstrap", "n_boot": 1000, "mean_block_bars": 10, "alpha": 0.05},
    "G15": {"name": "benchmark_dominance", "benchmarks": ["flat", "buy_and_hold_perp"]},
    "G17": {
        "name": "paper_congruence",
        "max_abs_sharpe_delta": 0.5,
        "min_paper_bars": 21,
    },
}

GATE_FAILURE_KIND: dict[str, FailureKind] = {
    "G01": FailureKind.LEAKAGE,
    "G02": FailureKind.DATA_INTEGRITY,
    "G03": FailureKind.OVERFIT,
    "G04": FailureKind.OVERFIT,
    "G06": FailureKind.INSUFFICIENT_SIGNIFICANCE,
    "G07": FailureKind.INSUFFICIENT_SIGNIFICANCE,
    "G08": FailureKind.OVERFIT,
    "G09": FailureKind.OVERFIT,
    "G10": FailureKind.REGIME_DEPENDENCE,
    "G11": FailureKind.INSUFFICIENT_SIGNIFICANCE,
    "G12": FailureKind.COST_FRAGILITY,
    "G15": FailureKind.BENCHMARK_INFERIOR,
    "G16": FailureKind.IRREPRODUCIBLE,
    "G17": FailureKind.COST_FRAGILITY,
}

_COMPARATORS = {
    ">=": lambda s, t: s >= t,
    ">": lambda s, t: s > t,
    "<=": lambda s, t: s <= t,
    "<": lambda s, t: s < t,
    "==": lambda s, t: s == t,
}


def clean(value: Any) -> Any:
    """JSON-safe, deterministic copy: NumPy -> Python, tuples -> lists, non-finite -> None."""
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [clean(v) for v in value]
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer | int):
        return int(value)
    if isinstance(value, np.floating | float):
        f = float(value)
        return f if math.isfinite(f) else None
    return value


def compare(stat: float | None, threshold: float, comparator: str) -> bool:
    if stat is None or not math.isfinite(stat):
        return False
    return bool(_COMPARATORS[comparator](stat, threshold))


def make_gate(
    gate_id: str,
    name: str,
    passed: bool,
    stat: float | None,
    threshold: float | None,
    comparator: str,
    params: Mapping[str, Any],
    details: Mapping[str, Any],
) -> GateResult:
    stat_clean = clean(stat) if stat is not None else None
    return GateResult(
        gate_id=gate_id,
        name=name,
        result=GateOutcome.PASS if passed else GateOutcome.FAIL,
        stat=stat_clean,
        threshold=clean(threshold) if threshold is not None else None,
        comparator=comparator,
        params=clean(dict(params)),
        details=clean(dict(details)),
    )


def failed_gate(gate_id: str, name: str, reason: str, params: Mapping[str, Any]) -> GateResult:
    """A gate that could not be evaluated. Fails closed."""
    return make_gate(gate_id, name, False, None, None, "n/a", params, {"reason": reason})


def g01_pit_integrity(
    cfg: Mapping[str, Any],
    timestamp_counts: Mapping[str, int],
    truncation: TruncationCheckResult,
) -> GateResult:
    max_violations = int(cfg["max_violations"])
    violations = sum(timestamp_counts.values()) + len(truncation.mismatches)
    evidence = truncation.checks > 0
    return make_gate(
        "G01",
        str(cfg["name"]),
        evidence and violations <= max_violations,
        float(violations),
        float(max_violations),
        "<=",
        {**cfg, "truncation_atol": 1e-12},
        {
            "timestamp_violations": dict(timestamp_counts),
            "truncation_checks": truncation.checks,
            "truncation_checkpoints": truncation.checkpoints,
            "truncation_mismatches": [
                {
                    "job": m.job_id,
                    "checkpoint": m.checkpoint,
                    "first_bad_index": m.first_bad_index,
                    "max_abs_diff": m.max_abs_diff,
                }
                for m in truncation.mismatches[:20]
            ],
            "evidence": "ok" if evidence else "no truncation checkpoints could be sampled",
        },
    )


def embargo_bars(cfg: Mapping[str, Any], label_horizon_bars: int) -> int:
    return max(
        int(cfg["min_embargo_bars"]), int(cfg["embargo_horizon_multiple"]) * label_horizon_bars
    )


def g02_chronological_split(
    cfg: Mapping[str, Any],
    *,
    n_bars_total: int,
    n_holdout_bars: int,
    n_embargo_bars: int,
    n_research_bars: int,
    overlaps: Sequence[Mapping[str, Any]],
) -> GateResult:
    problems = [dict(o) for o in overlaps]
    expected_holdout = math.floor(float(cfg["holdout_fraction"]) * n_bars_total)
    if n_holdout_bars < expected_holdout:
        problems.append({"kind": "holdout_too_small", "bars": n_holdout_bars})
    if n_research_bars < 1:
        problems.append({"kind": "no_research_period_before_embargo"})
    return make_gate(
        "G02",
        str(cfg["name"]),
        not problems,
        float(len(problems)),
        0.0,
        "<=",
        dict(cfg),
        {
            "n_bars_total": n_bars_total,
            "n_holdout_bars": n_holdout_bars,
            "n_embargo_bars": n_embargo_bars,
            "n_research_bars": n_research_bars,
            "violations": problems[:50],
        },
    )


def g03_out_of_sample(
    cfg: Mapping[str, Any], *, net_sharpe: float, n_bars: int, top_days_removed_pnl: float
) -> GateResult:
    min_sharpe = float(cfg["min_holdout_net_sharpe"])
    enough_bars = n_bars >= int(cfg["min_holdout_bars"])
    robust = top_days_removed_pnl > 0
    return make_gate(
        "G03",
        str(cfg["name"]),
        enough_bars and robust and compare(net_sharpe, min_sharpe, ">="),
        net_sharpe,
        min_sharpe,
        ">=",
        dict(cfg),
        {
            "holdout_bars": n_bars,
            "enough_bars": enough_bars,
            "pnl_after_top_days_removed": top_days_removed_pnl,
            "not_driven_by_few_days": robust,
        },
    )


def g06_multiple_testing(
    cfg: Mapping[str, Any],
    *,
    candidate_adjusted_p: float,
    candidate_raw_p: float,
    family_pvalues: Sequence[float],
    registered: bool,
    registration_problems: Sequence[str],
) -> GateResult:
    alpha = float(cfg["alpha"])
    needs_registration = bool(cfg.get("require_registered_trials", True))
    ok_registration = registered or not needs_registration
    return make_gate(
        "G06",
        str(cfg["name"]),
        ok_registration and compare(candidate_adjusted_p, alpha, "<="),
        candidate_adjusted_p,
        alpha,
        "<=",
        dict(cfg),
        {
            "candidate_raw_p": candidate_raw_p,
            "family_size": len(family_pvalues),
            "family_pvalues": list(family_pvalues),
            "registered": registered,
            "registration_problems": list(registration_problems),
        },
    )


def g07_deflated_sharpe(
    cfg: Mapping[str, Any], dsr: DeflatedSharpe, variance_source: str
) -> GateResult:
    threshold = float(cfg["min_dsr_probability"])
    return make_gate(
        "G07",
        str(cfg["name"]),
        compare(dsr.dsr, threshold, ">="),
        dsr.dsr,
        threshold,
        ">=",
        dict(cfg),
        {
            "sharpe_per_bar": dsr.sharpe,
            "sr0_per_bar": dsr.sr0,
            "n_obs": dsr.n_obs,
            "skew": dsr.skew,
            "kurtosis": dsr.kurtosis,
            "n_trials": dsr.n_trials,
            "trial_sharpe_variance": dsr.trial_sharpe_variance,
            "variance_source": variance_source,
        },
    )


def g12_costs(
    cfg: Mapping[str, Any],
    *,
    stressed_net_sharpe: float,
    base_net_sharpe: float,
    cost_model_version: str,
    cost_pnl_base: float,
    cost_pnl_stressed: float,
) -> GateResult:
    threshold = float(cfg["min_stressed_net_sharpe"])
    return make_gate(
        "G12",
        str(cfg["name"]),
        compare(stressed_net_sharpe, threshold, ">"),
        stressed_net_sharpe,
        threshold,
        ">",
        dict(cfg),
        {
            "base_net_sharpe": base_net_sharpe,
            "cost_model_version": cost_model_version,
            "holdout_cost_pnl": cost_pnl_base,
            "holdout_cost_pnl_stressed": cost_pnl_stressed,
        },
    )


def g16_reproducibility(
    cfg: Mapping[str, Any],
    *,
    max_abs_delta: float,
    position_delta: float,
    metric_delta: float,
    hashes_equal: bool,
) -> GateResult:
    threshold = float(cfg["max_abs_metric_delta"])
    return make_gate(
        "G16",
        str(cfg["name"]),
        compare(max_abs_delta, threshold, "<="),
        max_abs_delta,
        threshold,
        "<=",
        dict(cfg),
        {
            "position_max_abs_delta": position_delta,
            "metric_max_abs_delta": metric_delta,
            "result_hashes_equal": hashes_equal,
        },
    )


def _informative(gate_id: str) -> dict[str, Any]:
    return {**INFORMATIVE_GATES[gate_id], "informative": True}


def _cfg(gate_id: str, override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    base = _informative(gate_id)
    if override is None:
        return base
    return {**base, **dict(override), "informative": base.get("informative", True)}


def g04_walk_forward(
    fold_sharpes: Sequence[float], cfg: Mapping[str, Any] | None = None
) -> GateResult:
    conf = _cfg("G04", cfg)
    positive = sum(1 for s in fold_sharpes if s > 0)
    fraction = positive / len(fold_sharpes) if fold_sharpes else 0.0
    threshold = float(conf["min_positive_fraction"])
    return make_gate(
        "G04",
        conf["name"],
        bool(fold_sharpes) and fraction >= threshold,
        fraction,
        threshold,
        ">=",
        conf,
        {"fold_net_sharpes": list(fold_sharpes)},
    )


def g08_pbo(
    pbo: PBOResult | None, reason: str = "", cfg: Mapping[str, Any] | None = None
) -> GateResult:
    conf = _cfg("G08", cfg)
    if pbo is None:
        return failed_gate("G08", conf["name"], reason or "PBO not computable", conf)
    threshold = float(conf["max_pbo"])
    return make_gate(
        "G08",
        conf["name"],
        compare(pbo.pbo, threshold, "<"),
        pbo.pbo,
        threshold,
        "<",
        conf,
        {
            "n_strategies": pbo.n_strategies,
            "n_combinations": pbo.n_combinations,
            "n_splits": pbo.n_splits,
        },
    )


def g09_parameter_sensitivity(
    candidate_sharpe: float,
    neighbours: Mapping[str, float | None],
    cfg: Mapping[str, Any] | None = None,
) -> GateResult:
    conf = _cfg("G09", cfg)
    if not neighbours:
        return failed_gate("G09", conf["name"], "no numeric parameters to perturb", conf)
    values = list(neighbours.values())
    if any(v is None for v in values):
        worst: float | None = None
    else:
        worst = min(v for v in values if v is not None)
    threshold = float(conf["min_ratio"]) * max(candidate_sharpe, 0.0)
    return make_gate(
        "G09",
        conf["name"],
        candidate_sharpe > 0 and compare(worst, threshold, ">="),
        worst,
        threshold,
        ">=",
        conf,
        {"candidate_holdout_net_sharpe": candidate_sharpe, "neighbours": dict(neighbours)},
    )


def g10_regime_stress(
    regimes: Mapping[str, Mapping[str, float]], cfg: Mapping[str, Any] | None = None
) -> GateResult:
    conf = _cfg("G10", cfg)
    eligible = {
        k: v["net_sharpe"] for k, v in regimes.items() if v["bars"] >= conf["min_regime_bars"]
    }
    worst = min(eligible.values()) if eligible else None
    return make_gate(
        "G10",
        conf["name"],
        len(eligible) == len(regimes) and compare(worst, 0.0, ">"),
        worst,
        0.0,
        ">",
        conf,
        {"regimes": {k: dict(v) for k, v in regimes.items()}},
    )


def g11_bootstrap(
    ci: tuple[float, float], seed: int, cfg: Mapping[str, Any] | None = None
) -> GateResult:
    conf = _cfg("G11", cfg)
    return make_gate(
        "G11",
        conf["name"],
        compare(ci[0], 0.0, ">"),
        ci[0],
        0.0,
        ">",
        {**conf, "seed": seed},
        {"net_sharpe_ci": list(ci)},
    )


def g15_benchmark(
    candidate_sharpe: float,
    benchmarks: Mapping[str, float],
    cfg: Mapping[str, Any] | None = None,
) -> GateResult:
    conf = _cfg("G15", cfg)
    best = max(benchmarks.values())
    margin = candidate_sharpe - best
    return make_gate(
        "G15",
        conf["name"],
        compare(margin, 0.0, ">"),
        margin,
        0.0,
        ">",
        conf,
        {"candidate_net_sharpe": candidate_sharpe, "benchmark_net_sharpes": dict(benchmarks)},
    )


def g17_paper_congruence(
    *,
    research_net_sharpe: float,
    paper_net_sharpe: float | None,
    paper_bars: int,
    cfg: Mapping[str, Any] | None = None,
) -> GateResult:
    """Paper vs research Sharpe within band (VALIDATION_PROTOCOL §7)."""
    conf = _cfg("G17", cfg)
    min_bars = int(conf["min_paper_bars"])
    band = float(conf["max_abs_sharpe_delta"])
    if paper_net_sharpe is None or paper_bars < min_bars:
        return failed_gate(
            "G17",
            conf["name"],
            f"need >= {min_bars} paper bars with a Sharpe; got bars={paper_bars}",
            conf,
        )
    delta = abs(paper_net_sharpe - research_net_sharpe)
    return make_gate(
        "G17",
        conf["name"],
        compare(delta, band, "<="),
        delta,
        band,
        "<=",
        conf,
        {
            "research_net_sharpe": research_net_sharpe,
            "paper_net_sharpe": paper_net_sharpe,
            "paper_bars": paper_bars,
        },
    )
