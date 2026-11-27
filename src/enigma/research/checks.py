"""Programmatic gates a hypothesis proposal must pass before registration."""

from __future__ import annotations

import re

from enigma.research.mechanisms import canonical_domain
from enigma.schemas.quant import (
    CheckFailure,
    ColumnUse,
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

# Local models invent OHLC/return names that are not in the PIT panel. Map synonyms onto the
# first available real column; never invent columns that are not in the sealed dataset.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "returns": ("perp_close", "spot_close", "close"),
    "return": ("perp_close", "spot_close", "close"),
    "fwd_return": ("perp_close", "spot_close", "close"),
    "forward_return": ("perp_close", "spot_close", "close"),
    "log_return": ("perp_close", "spot_close", "close"),
    "price": ("perp_close", "spot_close", "close"),
    "close": ("perp_close", "spot_close", "close"),
    "open": ("perp_open", "open"),
    "high": ("perp_high", "high"),
    "low": ("perp_low", "low"),
    "volume": ("perp_volume", "volume"),
    "quote_volume": ("perp_quote_volume", "quote_volume"),
    "funding": ("funding_rate",),
    "fundingrate": ("funding_rate",),
    "funding_rate": ("funding_rate",),
    "basis": ("basis",),
    "spot": ("spot_close", "spot"),
}


def _is_empty_evidence_sentinel(eid: str) -> bool:
    """LLMs often emit 'none' instead of an empty contradictory list; treat as no citation."""
    return eid.strip().lower() in _EMPTY_EVIDENCE


def resolve_column(name: str, available: frozenset[str], role: str) -> str | None:
    """Map an LLM column name onto a sealed dataset column, or None if nothing fits."""
    if name in available:
        return name
    key = re.sub(r"[^a-z0-9]+", "", name.lower())
    # Normalise common separators: funding-rate / funding_rate / FundingRate
    spaced = re.sub(r"[\s\-]+", "_", name.strip().lower())
    for candidate in _COLUMN_ALIASES.get(spaced, ()) + _COLUMN_ALIASES.get(key, ()):
        if candidate in available:
            return candidate
    # An unmatched name is missing data. It is not replaced with another column.
    del role
    return None


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


def _normalize_required_data(
    proposal: HypothesisProposal,
    columns: dict[str, frozenset[str]],
    only_dataset: str | None,
) -> list[ColumnUse]:
    known_datasets = frozenset(columns)
    required: list[ColumnUse] = []
    for use in proposal.required_data:
        updates: dict[str, object] = {}
        dataset_id = use.dataset_id
        if dataset_id not in known_datasets and only_dataset is not None:
            dataset_id = only_dataset
            updates["dataset_id"] = dataset_id
        available = columns.get(dataset_id)
        if available is not None and use.column not in available:
            resolved = resolve_column(use.column, available, use.role)
            if resolved is not None and resolved != use.column:
                updates["column"] = resolved
        if use.role in {"signal", "control"} and use.min_lag_bars < 0:
            updates["min_lag_bars"] = 0
        required.append(use.model_copy(update=updates) if updates else use)
    return required


def _remap_short_evidence_ids(cited: tuple[str, ...], ordered: list[str]) -> tuple[str, ...]:
    remapped: list[str] = []
    for eid in cited:
        if _is_empty_evidence_sentinel(eid):
            continue
        match = re.fullmatch(r"ev[_-]?(\d+)", eid.strip(), flags=re.I)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(ordered):
                remapped.append(ordered[idx])
    return tuple(dict.fromkeys(remapped))


def normalize_proposal(
    proposal: HypothesisProposal, context: ResearchContext
) -> HypothesisProposal:
    """Mechanical repairs for common LLM slips (not a template hypothesis body)."""
    proposal = scrub_evidence_sentinels(proposal)
    evidence_ids = context.evidence_ids()
    columns = context.columns_by_dataset()
    only_dataset = next(iter(columns)) if len(columns) == 1 else None
    required = _normalize_required_data(proposal, columns, only_dataset)

    supporting = tuple(eid for eid in proposal.supporting_evidence_ids if eid in evidence_ids)
    contradictory = tuple(eid for eid in proposal.contradictory_evidence_ids if eid in evidence_ids)
    # If the model invented short ids, keep order by remapping ev_N → Nth context evidence.
    if not supporting and proposal.supporting_evidence_ids and context.evidence:
        supporting = _remap_short_evidence_ids(
            proposal.supporting_evidence_ids, [e.id for e in context.evidence]
        )

    instruments = tuple(i for i in proposal.instruments if i in frozenset(context.instruments))
    if not instruments:
        instruments = tuple(context.instruments)

    cost = proposal.execution_costs
    if cost.cost_model_version != context.cost_model_version:
        cost = cost.model_copy(update={"cost_model_version": context.cost_model_version})

    return proposal.model_copy(
        update={
            "domain": canonical_domain(proposal.domain),
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
    domain = canonical_domain(proposal.domain)
    if not domain:
        failures.append(CheckFailure(check="registered_mechanism", detail="domain is empty"))
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
