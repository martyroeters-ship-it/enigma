"""Derive and check StrategySpec against a registered hypothesis."""

from __future__ import annotations

from datetime import datetime

from enigma.core.ids import new_id
from enigma.schemas.common import Actor
from enigma.schemas.governance import ExecutionMode
from enigma.schemas.quant import (
    BaselineSpec,
    CheckFailure,
    ConsistencyReport,
    FeatureDef,
    RegisteredHypothesis,
    StrategySpec,
)


def derive_spec(
    registered: RegisteredHypothesis,
    *,
    name: str,
    signal_math: str,
    features: tuple[FeatureDef, ...],
    position_rule: str,
    baseline: BaselineSpec,
    created_by: Actor,
    created_at: datetime,
    execution_mode: ExecutionMode = ExecutionMode.DETERMINISTIC,
    prompt_ref: str = "",
    rebalance_bars: int = 1,
) -> StrategySpec:
    proposal = registered.proposal
    placeholder = "sha256:" + ("0" * 64)
    draft = StrategySpec.model_construct(
        id=new_id("sspec"),
        hypothesis_spec_id=registered.id,
        hypothesis_id=registered.hypothesis_id,
        name=name,
        signal_math=signal_math,
        features=features,
        parameters=proposal.design.parameters,
        position_rule=position_rule,
        position_bounds=(-1.0, 1.0),
        rebalance_bars=rebalance_bars,
        holding_bars=proposal.horizon_bars,
        complexity_class=proposal.complexity_class,
        complexity_justification=proposal.complexity_rationale,
        baseline=baseline,
        required_columns=proposal.columns(),
        label_horizon_bars=proposal.horizon_bars,
        execution_mode=execution_mode,
        prompt_ref=prompt_ref,
        provenance=(),
        created_by=created_by,
        created_at=created_at,
        content_hash=placeholder,
    )
    content_hash = StrategySpec.hash_fields(draft.model_dump(mode="json"))
    return StrategySpec.model_validate(
        {**draft.model_dump(mode="json"), "content_hash": content_hash}
    )


def check_consistency(
    spec: StrategySpec,
    registered: RegisteredHypothesis,
    *,
    available_columns: frozenset[str] | None = None,
) -> ConsistencyReport:
    failures: list[CheckFailure] = []
    checks: dict[str, bool] = {}

    horizon_ok = spec.label_horizon_bars == registered.proposal.horizon_bars
    checks["horizon_match"] = horizon_ok
    if not horizon_ok:
        failures.append(
            CheckFailure(
                check="horizon_match",
                detail=(
                    f"spec horizon {spec.label_horizon_bars} != "
                    f"hypothesis {registered.proposal.horizon_bars}"
                ),
            )
        )

    grid_ok = spec.search_space() == registered.proposal.search_space()
    checks["grid_match"] = grid_ok
    if not grid_ok:
        failures.append(CheckFailure(check="grid_match", detail="parameter grid mismatch"))

    cols = frozenset(spec.required_columns)
    hyp_cols = frozenset(registered.proposal.columns())
    cols_ok = cols == hyp_cols
    checks["columns_match"] = cols_ok
    if not cols_ok:
        failures.append(
            CheckFailure(
                check="columns_match", detail=f"spec {sorted(cols)} != hyp {sorted(hyp_cols)}"
            )
        )

    if available_columns is not None:
        missing = cols - available_columns
        checks["columns_available"] = not missing
        if missing:
            failures.append(
                CheckFailure(
                    check="columns_available",
                    detail=f"missing columns: {sorted(missing)}",
                )
            )

    for feat in spec.features:
        if feat.min_lag_bars < 0:
            checks["no_lookahead"] = False
            failures.append(
                CheckFailure(
                    check="no_lookahead",
                    detail=f"feature {feat.name} min_lag_bars={feat.min_lag_bars}",
                )
            )
    checks.setdefault("no_lookahead", True)

    lo, hi = spec.position_bounds
    bounds_ok = -1.0 <= lo <= 0.0 <= hi <= 1.0
    checks["position_bounds"] = bounds_ok
    if not bounds_ok:
        failures.append(CheckFailure(check="position_bounds", detail=str(spec.position_bounds)))

    return ConsistencyReport(
        spec_id=spec.id,
        hypothesis_spec_id=registered.id,
        checks=checks,
        failures=tuple(failures),
    )
