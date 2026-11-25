"""Immutable, content-hashed hypothesis registration."""

from __future__ import annotations

from datetime import datetime

from enigma.core.errors import IntegrityError
from enigma.core.ids import new_id
from enigma.schemas.common import Actor
from enigma.schemas.quant import (
    GenerationResult,
    HypothesisProposal,
    RegisteredHypothesis,
    ResearchContext,
)
from enigma.store.documents import DocumentStore

_PLACEHOLDER_HASH = "sha256:" + ("0" * 64)


class HypothesisRegistry:
    """Pre-register proposals; amendments create new records linked via `amends`."""

    def __init__(self, documents: DocumentStore) -> None:
        self.documents = documents

    def register(
        self,
        *,
        hypothesis_id: str,
        proposal: HypothesisProposal,
        context: ResearchContext,
        generation: GenerationResult,
        trial_budget: int,
        family_id: str | None = None,
        family_members: tuple[str, ...] | None = None,
        amends: str | None = None,
        registered_by: Actor,
        registered_at: datetime,
        lineage: tuple[str, ...] = (),
        cumulative_trial_budget: int | None = None,
    ) -> RegisteredHypothesis:
        if trial_budget < 1:
            raise ValueError("trial_budget must be >= 1")
        fam = family_id or hypothesis_id
        members = family_members or (hypothesis_id,)
        if hypothesis_id not in members:
            members = (*members, hypothesis_id)
        cum = cumulative_trial_budget if cumulative_trial_budget is not None else trial_budget
        if amends:
            parent = self.documents.require(RegisteredHypothesis, amends)
            lineage = (*parent.lineage, parent.id) if parent.lineage else (parent.id,)
            fam = parent.family_id
            members = tuple(dict.fromkeys((*parent.family_members, hypothesis_id)))
            cum = parent.cumulative_trial_budget + trial_budget

        draft = RegisteredHypothesis.model_construct(
            id=new_id("rhs"),
            hypothesis_id=hypothesis_id,
            objective_id=context.objective_id,
            family_id=fam,
            proposal=proposal,
            trial_budget=trial_budget,
            cost_model_version=context.cost_model_version,
            dataset_version_ids=tuple(d.dataset_version_id for d in context.datasets),
            amends=amends,
            lineage=lineage,
            family_members=members,
            cumulative_trial_budget=cum,
            generation_id=generation.id,
            execution_mode=generation.execution_mode,
            prompt_ref=generation.prompt_ref,
            provenance=generation.provenance,
            research_cutoff=context.research_cutoff,
            registered_by=registered_by,
            registered_at=registered_at,
            content_hash=_PLACEHOLDER_HASH,
        )
        content_hash = RegisteredHypothesis.hash_fields(draft.model_dump(mode="json"))
        record = RegisteredHypothesis.model_validate(
            {**draft.model_dump(mode="json"), "content_hash": content_hash}
        )
        self.documents.put(record, str(registered_by))
        return record

    def require(self, spec_id: str) -> RegisteredHypothesis:
        return self.documents.require(RegisteredHypothesis, spec_id)

    def assert_unmodified(self, spec_id: str) -> RegisteredHypothesis:
        record = self.require(spec_id)
        expected = RegisteredHypothesis.hash_fields(record.model_dump(mode="json"))
        if expected != record.content_hash:
            raise IntegrityError(f"RegisteredHypothesis {spec_id} content hash mismatch")
        return record
