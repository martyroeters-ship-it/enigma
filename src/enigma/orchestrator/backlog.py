"""Ranked research backlog: mechanisms, instruments, and timeframes.

A direction is one mechanism on one instrument at one horizon. Completing it once is
enough unless a lesson recorded a positive gross edge that costs then destroyed.
A mechanism is set aside only after every seeded direction has a scientific result
and none passed. A method-gate or infrastructure failure is not such a result.
The ranker prefers information gain and novelty over running more copies of the same question.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from enigma.orchestrator.cio import (
    Agenda,
    Candidate,
    FamilyHistory,
    ResearchPrioritizer,
    ScoredCandidate,
)
from enigma.orchestrator.planning import (
    build_inputs,
    family_for_mechanism,
    proposal_candidates,
    read_record,
)
from enigma.orchestrator.quarantine import candidate_symbols, load_quarantine
from enigma.research.mechanisms import MECHANISMS
from enigma.schemas.cycle import NextExperimentProposal
from enigma.schemas.market import MarketScanReport
from enigma.schemas.research import Hypothesis, ResearchObjective
from enigma.schemas.validation import RedactedValidationSummary
from enigma.store import Stores

HORIZONS = ("1h", "8h", "1d")
SYMBOLS = ("BTCUSDT", "ETHUSDT")
REVISIT = "revisit:positive_gross_destroyed_by_costs"
_METHOD_GATES = frozenset({"G01", "G02", "G16"})


@dataclass(frozen=True)
class BacklogPolicy:
    diminishing_returns_min_cycles: int = 3
    novelty_floor: float = 0.05


def direction_key(mechanism_id: str, symbol: str, horizon: str) -> str:
    return f"{mechanism_id}|{symbol}|{horizon}"


def _completed(history: FamilyHistory | None) -> int:
    if history is None:
        return 0
    return history.passes + history.fails


def _scientific(stores: Stores, state: dict[str, object]) -> bool:
    """Infrastructure and split/implementation gates are not evidence about a mechanism."""
    summary_id = state.get("validation_summary_id")
    if not summary_id:
        return True
    summary = stores.documents.get(RedactedValidationSummary, str(summary_id))
    if summary is None:
        return True
    gates = {name.split(":", 1)[0] for name in summary.failed_gate_names}
    return not gates or not gates <= _METHOD_GATES


def direction_history(stores: Stores) -> dict[str, FamilyHistory]:
    """Completed validations only. An abort is not a scientific trial of the direction."""
    hypotheses = {h.id: h for h in stores.documents.list(Hypothesis, limit=10_000)}
    counts: dict[str, dict[str, int]] = {}
    for cp in stores.checkpoints.list("research_cycle"):
        state = cp["state"]
        hyp = hypotheses.get(state.get("hypothesis_id") or "")
        if hyp is None:
            continue
        overall = state.get("validation_overall")
        if overall not in {"PASS", "FAIL"} or not _scientific(stores, state):
            continue
        mechanism = hyp.family_id.removeprefix("fam_")
        horizon = hyp.horizon or "8h"
        for instrument in hyp.instruments:
            key = direction_key(mechanism, instrument.symbol, horizon)
            bucket = counts.setdefault(key, {"cycles": 0, "passes": 0, "fails": 0})
            bucket["cycles"] += 1
            if overall == "PASS":
                bucket["passes"] += 1
            else:
                bucket["fails"] += 1
    return {
        key: FamilyHistory(key, cycles=c["cycles"], passes=c["passes"], fails=c["fails"])
        for key, c in counts.items()
    }


def mechanism_history(directions: dict[str, FamilyHistory]) -> dict[str, FamilyHistory]:
    totals: dict[str, dict[str, int]] = {}
    for key, hist in directions.items():
        mechanism = key.split("|", 1)[0]
        bucket = totals.setdefault(mechanism, {"cycles": 0, "passes": 0, "fails": 0})
        bucket["cycles"] += hist.cycles
        bucket["passes"] += hist.passes
        bucket["fails"] += hist.fails
    return {
        mechanism: FamilyHistory(
            family_for_mechanism(mechanism),
            cycles=c["cycles"],
            passes=c["passes"],
            fails=c["fails"],
        )
        for mechanism, c in totals.items()
    }


def seeded_opportunities() -> list[Candidate]:
    out: list[Candidate] = []
    for mechanism_id, mechanism in MECHANISMS.items():
        for symbol in SYMBOLS:
            for horizon in HORIZONS:
                key = direction_key(mechanism_id, symbol, horizon)
                out.append(
                    Candidate(
                        id="opp_" + key.replace("|", "_"),
                        kind="opportunity",
                        family=family_for_mechanism(mechanism_id),
                        summary=(
                            f"Does {mechanism.name} show a gross edge on {symbol} "
                            f"perpetuals at a {horizon} horizon before trading costs? "
                            f"Direction {key}."
                        ),
                        refs=(mechanism_id, symbol, horizon),
                    )
                )
    return out


def _scan_candidates(stores: Stores) -> list[Candidate]:
    used = {
        cp["state"].get("objective_id")
        for cp in stores.checkpoints.list("research_cycle")
        if cp["state"].get("objective_id")
    }
    out: list[Candidate] = []
    for report in stores.documents.list(MarketScanReport, limit=8):
        if not report.objective_id or report.objective_id in used:
            continue
        objective = stores.documents.get(ResearchObjective, report.objective_id)
        if objective is None:
            continue
        mechanism = report.suggested_mechanism or "funding_crowding_reversal"
        out.append(
            Candidate(
                id=report.objective_id,
                kind="market_scan",
                family=family_for_mechanism(mechanism),
                summary=objective.statement,
                refs=(report.objective_id,),
            )
        )
    return out


def _mark(
    item: ScoredCandidate,
    *,
    viable: bool,
    reason: str,
) -> ScoredCandidate:
    reasons = item.reasons if viable else (*item.reasons, reason)
    return ScoredCandidate(
        item.candidate,
        item.score,
        item.components,
        reasons,
        viable and item.viable,
    )


def _justified(stores: Stores, candidate: Candidate) -> bool:
    if candidate.proposal_id is None:
        return False
    proposal = stores.documents.get(NextExperimentProposal, candidate.proposal_id)
    if proposal is None:
        return False
    return proposal.hypothesis_delta.get("revisit") == REVISIT.removeprefix("revisit:")


def apply_direction_policy(
    agenda: Agenda,
    stores: Stores,
    policy: BacklogPolicy,
) -> Agenda:
    directions = direction_history(stores)
    mechanisms = mechanism_history(directions)
    quarantine = load_quarantine(stores)
    ranked: list[ScoredCandidate] = []
    for item in agenda.ranked:
        candidate = item.candidate
        if quarantine.datasets or quarantine.symbols or quarantine.candidates:
            blocked = quarantine.reason(candidate, candidate_symbols(stores, candidate))
            if blocked:
                ranked.append(
                    _mark(
                        item,
                        viable=False,
                        reason=(
                            "quarantine: dataset inaccessible; the objective was not "
                            "opened and validation was not run"
                        ),
                    )
                )
                continue
        mechanism = candidate.family.removeprefix("fam_").split("|", 1)[0]
        mech_hist = mechanisms.get(mechanism)
        # A mechanism stays open until every seeded direction has a scientific
        # result and none passed. A handful of failures does not close the rest.
        grid = len(SYMBOLS) * len(HORIZONS) if mechanism in MECHANISMS else None
        abandoned = (
            grid is not None
            and mech_hist is not None
            and mech_hist.passes == 0
            and _completed(mech_hist) >= max(grid, policy.diminishing_returns_min_cycles)
        )
        exact: FamilyHistory | None = None
        if candidate.kind == "opportunity" and len(candidate.refs) == 3:
            mechanism_id, symbol, horizon = candidate.refs
            exact = directions.get(direction_key(mechanism_id, symbol, horizon))
        justified = _justified(stores, candidate)
        if abandoned and not justified:
            ranked.append(
                _mark(
                    item,
                    viable=False,
                    reason=(
                        "every seeded direction of this mechanism has a scientific result "
                        "and none passed"
                    ),
                )
            )
            continue
        if exact is not None and _completed(exact) >= 1 and not justified:
            ranked.append(
                _mark(
                    item,
                    viable=False,
                    reason=(
                        "this mechanism, instrument, and timeframe already has a "
                        "completed validation; repeating it needs a positive gross edge "
                        "that costs destroyed"
                    ),
                )
            )
            continue
        share = 0.0
        total = sum(_completed(h) for h in mechanisms.values())
        if total and mech_hist is not None:
            share = _completed(mech_hist) / total
        adjusted = round(item.score - 0.15 * share, 6)
        ranked.append(
            ScoredCandidate(candidate, adjusted, item.components, item.reasons, item.viable)
        )
    ranked.sort(key=lambda s: (not s.viable, -s.score, s.candidate.id))
    return Agenda(tuple(ranked), agenda.weights, agenda.notes)


def research_agenda(
    stores: Stores,
    *,
    inputs_budget: float,
    policy: BacklogPolicy | None = None,
) -> Agenda:
    """Proposals, scan objectives, and seeded directions, highest value first."""
    policy = policy or BacklogPolicy()
    record = read_record(stores)
    texts = record.prior_hypothesis_texts + tuple(
        o.statement for o in stores.documents.list(ResearchObjective, limit=2000)
    )
    record = replace(record, prior_hypothesis_texts=texts)
    candidates = [
        *proposal_candidates(stores),
        *_scan_candidates(stores),
        *seeded_opportunities(),
    ]
    agenda = ResearchPrioritizer().rank(
        candidates,
        build_inputs(record, remaining_budget=inputs_budget),
    )
    return apply_direction_policy(agenda, stores, policy)
