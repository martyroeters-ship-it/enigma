"""LLM-backed hypothesis generation with programmatic checks (no silent template fallback)."""

from __future__ import annotations

import json
from datetime import datetime

from enigma.core.ids import new_id
from enigma.gateway.errors import StructuredOutputError
from enigma.gateway.prompts import PromptRegistry, PromptTemplate
from enigma.gateway.protocol import StructuredGenerator, provenance_of
from enigma.research.checks import acquisition_needed, check_proposal, normalize_proposal
from enigma.research.mechanisms import canonical_domain
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import Budget, ExecutionMode
from enigma.schemas.quant import (
    DatasetAcquisitionRequest,
    GenerationResult,
    HypothesisProposal,
    RejectedDraft,
    ResearchContext,
)

PROMPT_ID = "quant.hypothesis"
METHOD = "llm:structured-hypothesis-v1"
TASK_KIND = "quant.hypothesis"


class HypothesisGenerator:
    """Produce checkable `HypothesisProposal`s via `StructuredLLM` + registered prompts.

    On LLM or check failure this returns an explicit `GenerationResult` with
    `failure_reason` / `rejected` — it never substitutes mechanism-library templates.
    """

    def __init__(
        self,
        llm: StructuredGenerator,
        prompts: PromptRegistry,
        *,
        prompt_id: str = PROMPT_ID,
        model: str | None = None,
        max_output_tokens: int = 1800,
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.prompt_id = prompt_id
        self.model = model
        self.max_output_tokens = max_output_tokens

    def generate(
        self,
        context: ResearchContext,
        *,
        actor: AgentRole = AgentRole.QUANT_RESEARCH,
        task_id: str | None,
        budget: Budget,
        now: datetime,
        n: int = 1,
    ) -> GenerationResult:
        prompt = self.prompts.get(self.prompt_id)
        provenance = []
        rejected: list[RejectedDraft] = []
        acquisitions: list[DatasetAcquisitionRequest] = []
        proposals: list[HypothesisProposal] = []
        repair_rounds = 0
        gen_id = new_id("hgen")

        try:
            draft, prov = self._call(prompt, context, actor, task_id, budget, critique=None)
            provenance.append(prov)
        except Exception as exc:
            return GenerationResult(
                id=gen_id,
                context_id=context.id,
                context_hash=context.content_hash(),
                execution_mode=ExecutionMode.LLM,
                method=METHOD,
                prompt_refs=(prompt.ref,),
                provenance=tuple(provenance),
                proposals=(),
                rejected=(),
                repair_rounds=0,
                failure_reason=f"llm_error: {exc}",
                created_by=actor,
                created_at=now,
            )

        draft = normalize_proposal(draft, context)
        failures = check_proposal(draft, context)
        if failures:
            rejected.append(
                RejectedDraft(
                    title=draft.title,
                    round=0,
                    reasons=failures,
                    draft=draft.model_dump(mode="json"),
                )
            )
            for desc, cols, rationale in acquisition_needed(draft, context):
                acquisitions.append(
                    DatasetAcquisitionRequest(
                        id=new_id("daq"),
                        requested_by=actor,
                        description=desc,
                        columns=cols,
                        rationale=rationale,
                        source_draft_title=draft.title,
                        generation_id=gen_id,
                        created_at=now,
                    )
                )
            critique = "; ".join(f"{f.check}: {f.detail}" for f in failures)
            repair_rounds = 1
            try:
                draft, prov = self._call(prompt, context, actor, task_id, budget, critique=critique)
                provenance.append(prov)
            except Exception as exc:
                return GenerationResult(
                    id=gen_id,
                    context_id=context.id,
                    context_hash=context.content_hash(),
                    execution_mode=ExecutionMode.LLM,
                    method=METHOD,
                    prompt_refs=(prompt.ref,),
                    provenance=tuple(provenance),
                    proposals=(),
                    rejected=tuple(rejected),
                    acquisition_requests=tuple(acquisitions),
                    repair_rounds=repair_rounds,
                    failure_reason=f"llm_repair_error: {exc}",
                    created_by=actor,
                    created_at=now,
                )
            draft = normalize_proposal(draft, context)
            failures = check_proposal(draft, context)
            if failures:
                rejected.append(
                    RejectedDraft(
                        title=draft.title,
                        round=1,
                        reasons=failures,
                        draft=draft.model_dump(mode="json"),
                    )
                )
            else:
                proposals.append(draft)
        else:
            proposals.append(draft)

        # Optional additional drafts when n > 1 (same prompt, no repair budget shared)
        for _ in range(1, max(1, n)):
            break  # single-proposal path for M2; n reserved for coordinator batching

        failure_reason = None if proposals else "no_proposal_survived_checks"
        return GenerationResult(
            id=gen_id,
            context_id=context.id,
            context_hash=context.content_hash(),
            execution_mode=ExecutionMode.LLM,
            method=METHOD,
            prompt_refs=(prompt.ref,),
            provenance=tuple(provenance),
            proposals=tuple(proposals),
            rejected=tuple(rejected),
            acquisition_requests=tuple(acquisitions),
            repair_rounds=repair_rounds,
            failure_reason=failure_reason,
            created_by=actor,
            created_at=now,
        )

    def _call(
        self,
        prompt: PromptTemplate,
        context: ResearchContext,
        actor: AgentRole,
        task_id: str | None,
        budget: Budget,
        *,
        critique: str | None,
    ) -> tuple[HypothesisProposal, object]:
        # Keep the prompt small for local CPUs: too much evidence makes 3b/7b models invent
        # columns and ignore allowlists.
        evidence_window = context.evidence[:8]
        evidence_compact = "\n".join(f"- {e.id}: {(e.text or '')[:160]}" for e in evidence_window)
        columns_by_dataset = "; ".join(
            f"{d.dataset_version_id}=[{', '.join(d.columns)}]" for d in context.datasets
        )
        # Omit bulky evidence/prior bodies from the dump; allowlists are passed explicitly.
        compact = context.model_dump(mode="json")
        compact["evidence"] = [
            {"id": e.id, "topics": list(e.topics), "text": (e.text or "")[:120]}
            for e in evidence_window
        ]
        compact["prior_hypotheses"] = compact.get("prior_hypotheses", [])[:4]
        system, user = prompt.render(
            objective=context.objective,
            research_cutoff=context.research_cutoff.isoformat(),
            instruments=", ".join(context.instruments),
            cost_model_version=context.cost_model_version,
            dataset_ids=", ".join(d.dataset_version_id for d in context.datasets),
            columns_by_dataset=columns_by_dataset,
            evidence_ids=", ".join(e.id for e in evidence_window) or "(none)",
            evidence_compact=evidence_compact or "(none)",
            context_json=json.dumps(compact, indent=2)[:8000],
            critique=critique or "(none)",
            tested_mechanisms=", ".join(
                dict.fromkeys(canonical_domain(p.family_id) for p in context.prior_hypotheses)
            )
            or "(none)",
        )
        traced = self.llm.generate_traced(
            HypothesisProposal,
            system,
            user,
            actor,
            task_id,
            budget,
            task_kind=TASK_KIND,
            model=self.model,
            max_output_tokens=self.max_output_tokens,
        )
        if not isinstance(traced.value, HypothesisProposal):
            raise StructuredOutputError("expected HypothesisProposal")
        return traced.value, provenance_of(traced.response, prompt.ref)
