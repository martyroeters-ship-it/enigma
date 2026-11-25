"""Builds the CIO's prioritization inputs and candidates from institutional records."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from enigma.orchestrator.cio import Candidate, FamilyHistory, PrioritizationInputs
from enigma.schemas.common import AgentRole
from enigma.schemas.cycle import NextExperimentProposal, ProposalState
from enigma.schemas.research import DatasetVersion, Hypothesis
from enigma.store import Stores

CYCLE_ROLES = (
    AgentRole.INTELLIGENCE,
    AgentRole.DATA_ENGINEERING,
    AgentRole.MARKET_MICROSTRUCTURE,
    AgentRole.QUANT_RESEARCH,
    AgentRole.STRATEGY_ENGINEERING,
    AgentRole.INDEPENDENT_VALIDATION,
    AgentRole.META_LEARNING,
)


def family_for_mechanism(mechanism_id: str) -> str:
    return mechanism_id if mechanism_id.startswith("fam_") else f"fam_{mechanism_id}"


@dataclass(frozen=True)
class InstitutionalRecord:
    history: dict[str, FamilyHistory]
    prior_hypothesis_texts: tuple[str, ...]
    available_datasets: frozenset[str]
    agent_success_rates: dict[AgentRole, float]


def read_record(stores: Stores) -> InstitutionalRecord:
    docs = stores.documents
    hypotheses = {h.id: h for h in docs.list(Hypothesis, limit=10_000)}
    counts: dict[str, dict[str, int]] = {}
    for cp in stores.checkpoints.list("research_cycle"):
        state = cp["state"]
        hyp = hypotheses.get(state.get("hypothesis_id") or "")
        family = hyp.family_id if hyp else "unattributed"
        bucket = counts.setdefault(family, {"cycles": 0, "passes": 0, "fails": 0, "aborted": 0})
        bucket["cycles"] += 1
        overall = state.get("validation_overall")
        if overall == "PASS":
            bucket["passes"] += 1
        elif overall == "FAIL":
            bucket["fails"] += 1
        elif cp["status"] == "aborted":
            bucket["aborted"] += 1
    history = {
        family: FamilyHistory(
            family, cycles=c["cycles"], passes=c["passes"], fails=c["fails"], aborted=c["aborted"]
        )
        for family, c in counts.items()
    }
    texts = tuple(f"{h.title}. {h.mechanism}" for h in hypotheses.values())
    datasets = frozenset(
        d.name.split("_btcusdt")[0] for d in docs.list(DatasetVersion, limit=10_000)
    )
    rates: dict[AgentRole, float] = {}
    for role, statuses in stores.queue.outcomes_by_role().items():
        done = statuses.get("succeeded", 0)
        failed = statuses.get("failed", 0)
        if done + failed:
            try:
                rates[AgentRole(role)] = done / (done + failed)
            except ValueError:
                continue
    return InstitutionalRecord(history, texts, datasets, rates)


def proposal_candidates(stores: Stores) -> list[Candidate]:
    out = []
    for p in stores.documents.list(
        NextExperimentProposal, where={"state": ProposalState.PROPOSED.value}, limit=200
    ):
        delta = "; ".join(f"{k}={v}" for k, v in sorted(p.hypothesis_delta.items()))
        out.append(
            Candidate(
                id=p.id,
                kind="next_experiment",
                family=family_for_mechanism(p.mechanism_id),
                summary=f"{p.rationale} {delta}",
                roles=CYCLE_ROLES,
                proposal_id=p.id,
                refs=(p.parent_cycle_id, *p.lesson_ids),
            )
        )
    return out


def build_inputs(
    record: InstitutionalRecord,
    remaining_budget: float,
    guidance_penalty: Callable[[Candidate], tuple[float, tuple[str, ...]]] | None = None,
    holdout_available: bool = True,
) -> PrioritizationInputs:
    return PrioritizationInputs(
        history=record.history,
        prior_hypothesis_texts=record.prior_hypothesis_texts,
        available_datasets=record.available_datasets,
        agent_success_rates=record.agent_success_rates,
        remaining_budget=remaining_budget,
        guidance_penalty=guidance_penalty,
        holdout_available=holdout_available,
    )
