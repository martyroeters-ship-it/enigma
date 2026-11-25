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


_EMPTY_EVIDENCE = frozenset(
    {"", "none", "null", "n/a", "na", "nil", "-", "unknown", "not_applicable"}
)


def _is_empty_evidence_sentinel(eid: str) -> bool:
    """LLMs often emit 'none' instead of an empty contradictory list; treat as no citation."""
    return eid.strip().lower() in _EMPTY_EVIDENCE


def scrub_evidence_sentinels(proposal: HypothesisProposal) -> HypothesisProposal:
    """Drop none/null sentinel ids before checks and persistence."""
    supporting = tuple(
        eid for eid in proposal.supporting_evidence_ids if not _is_empty_evidence_sentinel(eid)
    )
    contradictory = tuple(
        eid for eid in proposal.contradictory_evidence_ids if not _is_empty_evidence_sentinel(eid)
    )
    if (
        supporting == proposal.supporting_evidence_ids
        and contradictory == proposal.contradictory_evidence_ids
    ):
        return proposal
    return proposal.model_copy(
        update={
            "supporting_evidence_ids": supporting,
            "contradictory_evidence_ids": contradictory,
        }
    )


def normalize_proposal(
    proposal: HypothesisProposal, context: ResearchContext
) -> HypothesisProposal:
    """Mechanical repairs for common LLM slips (not a template hypothesis body)."""
    proposal = scrub_evidence_sentinels(proposal)
    evidence_ids = context.evidence_ids()
    columns = context.columns_by_dataset()
    known_datasets = frozenset(columns)
    only_dataset = next(iter(known_datasets)) if len(known_datasets) == 1 else None

    required = []
    for use in proposal.required_data:
        updates: dict[str, object] = {}
        if use.dataset_id not in known_datasets and only_dataset is not None:
            updates["dataset_id"] = only_dataset
        if use.role in {"signal", "control"} and use.min_lag_bars < 0:
            updates["min_lag_bars"] = 0
        required.append(use.model_copy(update=updates) if updates else use)

    supporting = tuple(eid for eid in proposal.supporting_evidence_ids if eid in evidence_ids)
    contradictory = tuple(
        eid for eid in proposal.contradictory_evidence_ids if eid in evidence_ids
    )
    # If the model invented short ids, keep order by remapping ev_N → Nth context evidence.
    if not supporting and proposal.supporting_evidence_ids and context.evidence:
        remapped: list[str] = []
        ordered = [e.id for e in context.evidence]
        for eid in proposal.supporting_evidence_ids:
            if _is_empty_evidence_sentinel(eid):
                continue
            match = re.fullmatch(r"ev[_-]?(\d+)", eid.strip(), flags=re.I)
            if match:
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(ordered):
                    remapped.append(ordered[idx])
        supporting = tuple(dict.fromkeys(remapped))

    instruments = tuple(i for i in proposal.instruments if i in frozenset(context.instruments))
    if not instruments:
        instruments = tuple(context.instruments)

    cost = proposal.execution_costs
    if cost.cost_model_version != context.cost_model_version:
        cost = cost.model_copy(update={"cost_model_version": context.cost_model_version})

    return proposal.model_copy(
        update={
            "required_data": tuple(required),
            "supporting_evidence_ids": supporting,
            "contradictory_evidence_ids": contradictory,
            "instruments": instruments,
            "execution_costs": cost,
        }
    )


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

    valid_supporting = 0
    for eid in proposal.supporting_evidence_ids:
        if _is_empty_evidence_sentinel(eid):
            failures.append(
                CheckFailure(
                    check="evidence_exists",
                    detail="supporting evidence cannot be a none/null sentinel",
                )
            )
            continue
        if eid not in evidence_ids:
            failures.append(
                CheckFailure(check="evidence_exists", detail=f"unknown supporting evidence {eid}")
            )
        else:
            valid_supporting += 1
    if valid_supporting == 0 and evidence_ids:
        failures.append(
            CheckFailure(
                check="evidence_exists",
                detail="at least one supporting evidence id from the context is required",
            )
        )
    for eid in proposal.contradictory_evidence_ids:
        if _is_empty_evidence_sentinel(eid):
            continue
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
