"""Bridge LLM `HypothesisProposal` / registration records into the cycle's legacy `Hypothesis`."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from enigma.core.hashing import sha256_hex
from enigma.schemas.common import Actor, Instrument
from enigma.schemas.quant import GenerationResult, HypothesisProposal, RegisteredHypothesis
from enigma.schemas.research import Hypothesis

METHODOLOGY_VERSION = "qm-llm-0.1"


def hypothesis_from_proposal(
    proposal: HypothesisProposal,
    *,
    objective_id: str,
    instruments: Sequence[Instrument],
    generation: GenerationResult,
    registered: RegisteredHypothesis | None = None,
    dataset_version_ids: tuple[str, ...] = (),
    lesson_ids: tuple[str, ...] = (),
    created_by: Actor,
    created_at: datetime,
    cycle_id: str = "",
    parent_hypothesis_id: str | None = None,
) -> Hypothesis:
    """Map a checked proposal into the document the research cycle already consumes."""
    family_id = registered.family_id if registered else f"fam_{proposal.domain}"
    hyp_id = (
        registered.hypothesis_id
        if registered
        else "hyp_"
        + sha256_hex(f"{cycle_id}|{proposal.id}|{proposal.signal_definition}")[:26].upper()
    )
    features = proposal.columns("signal") or proposal.columns()
    search_space = {k: tuple(v) for k, v in proposal.search_space().items()}
    return Hypothesis(
        id=hyp_id,
        objective_id=objective_id,
        family_id=family_id,
        title=proposal.title,
        mechanism=proposal.market_mechanism,
        prediction=proposal.alternative_hypothesis,
        formal_statement=proposal.signal_definition,
        falsification_criteria=tuple(proposal.failure_conditions),
        instruments=tuple(instruments),
        horizon=proposal.horizon,
        features=features,
        search_space=search_space,
        trial_budget=registered.trial_budget if registered else max(1, proposal.design.grid_size),
        cost_model_version=proposal.execution_costs.cost_model_version,
        dataset_version_ids=dataset_version_ids
        or (registered.dataset_version_ids if registered else ()),
        evidence_ids=tuple(proposal.supporting_evidence_ids),
        contradictory_evidence_ids=tuple(proposal.contradictory_evidence_ids),
        lesson_ids=lesson_ids,
        parent_hypothesis_id=parent_hypothesis_id,
        uncertainty_notes=(
            f"generation={generation.id}; mode={generation.execution_mode}; "
            f"prompt={generation.prompt_ref}; novelty={proposal.novelty_statement}"
        ),
        generation_method=generation.method,
        methodology_version=METHODOLOGY_VERSION,
        created_by=created_by,
        created_at=created_at,
    )
