"""Programmatic gates a hypothesis proposal must pass before registration."""

from __future__ import annotations

import re

from enigma.schemas.quant import (
    CheckFailure,
    HypothesisProposal,
    PriorHypothesisOutcome,
    ResearchContext,
)

_TOKEN = re.compile(r"[a-z0-9]+")
MAX_GRID_SIZE = 24
MAX_PARAM_VALUES = 6


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def fingerprint(proposal: HypothesisProposal) -> str:
    cols = ",".join(proposal.columns())
    return (
        f"{proposal.domain}|{proposal.market_mechanism}|{proposal.signal_definition}|"
        f"{cols}|{proposal.horizon_bars}"
    )


def prior_fingerprint(prior: PriorHypothesisOutcome) -> str:
    cols = ",".join(prior.columns)
    return f"{prior.mechanism}|{prior.signal}|{cols}|{prior.horizon_bars}"


def check_proposal(  # noqa: PLR0912 - each gate is an independent explicit failure path
    proposal: HypothesisProposal,
    context: ResearchContext,
    *,
    similarity_threshold: float = 0.85,
) -> tuple[CheckFailure, ...]:
    """Return every failed check (empty tuple means the proposal may proceed)."""
    failures: list[CheckFailure] = []
    evidence_ids = context.evidence_ids()
    columns = context.columns_by_dataset()
    instruments = frozenset(context.instruments)

    for eid in proposal.supporting_evidence_ids:
        if eid not in evidence_ids:
            failures.append(
                CheckFailure(check="evidence_exists", detail=f"unknown supporting evidence {eid}")
            )
    for eid in proposal.contradictory_evidence_ids:
        if eid not in evidence_ids:
            failures.append(
                CheckFailure(
                    check="evidence_exists", detail=f"unknown contradictory evidence {eid}"
                )
            )

    for use in proposal.required_data:
        available = columns.get(use.dataset_id)
        if available is None:
            failures.append(
                CheckFailure(
                    check="dataset_available",
                    detail=f"dataset {use.dataset_id} is not in the research context",
                )
            )
        elif use.column not in available:
            failures.append(
                CheckFailure(
                    check="column_available",
                    detail=f"column {use.column} missing from dataset {use.dataset_id}",
                )
            )
        if use.role in {"signal", "control"} and use.min_lag_bars < 0:
            failures.append(
                CheckFailure(
                    check="no_lookahead",
                    detail=f"{use.role} column {use.column} has min_lag_bars={use.min_lag_bars}",
                )
            )

    for inst in proposal.instruments:
        if inst not in instruments:
            failures.append(
                CheckFailure(check="instrument_allowed", detail=f"instrument {inst} not in context")
            )

    if proposal.design.grid_size > MAX_GRID_SIZE:
        failures.append(
            CheckFailure(
                check="grid_size",
                detail=f"grid_size {proposal.design.grid_size} exceeds max {MAX_GRID_SIZE}",
            )
        )
    for param in proposal.design.parameters:
        if len(param.values) > MAX_PARAM_VALUES:
            failures.append(
                CheckFailure(
                    check="param_values",
                    detail=(
                        f"parameter {param.name} has {len(param.values)} values; "
                        f"max {MAX_PARAM_VALUES}"
                    ),
                )
            )

    if proposal.execution_costs.cost_model_version != context.cost_model_version:
        failures.append(
            CheckFailure(
                check="cost_model",
                detail=(
                    f"proposal cost model {proposal.execution_costs.cost_model_version} != "
                    f"context {context.cost_model_version}"
                ),
            )
        )

    signal_fp = (
        f"{proposal.market_mechanism}|{proposal.signal_definition}|"
        f"{','.join(proposal.columns())}|{proposal.horizon_bars}"
    )
    for prior in context.prior_hypotheses:
        if signal_fp == prior_fingerprint(prior):
            failures.append(
                CheckFailure(
                    check="novelty",
                    detail=f"duplicate fingerprint of prior hypothesis {prior.hypothesis_id}",
                )
            )
            break
        if jaccard(proposal.signal_definition, prior.signal) >= similarity_threshold:
            failures.append(
                CheckFailure(
                    check="novelty",
                    detail=f"too similar to prior hypothesis {prior.hypothesis_id}",
                )
            )
            break

    return tuple(failures)


def acquisition_needed(
    proposal: HypothesisProposal, context: ResearchContext
) -> list[tuple[str, tuple[str, ...], str]]:
    """Return (description, columns, rationale) for missing datasets/columns."""
    columns = context.columns_by_dataset()
    missing: list[tuple[str, tuple[str, ...], str]] = []
    for use in proposal.required_data:
        available = columns.get(use.dataset_id)
        if available is None:
            missing.append(
                (
                    f"dataset {use.dataset_id}",
                    (use.column,),
                    f"required for {use.role} in proposal {proposal.title}",
                )
            )
        elif use.column not in available:
            missing.append(
                (
                    f"column {use.column} on {use.dataset_id}",
                    (use.column,),
                    f"required for {use.role} in proposal {proposal.title}",
                )
            )
    return missing
