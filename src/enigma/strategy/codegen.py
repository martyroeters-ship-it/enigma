"""LLM strategy codegen with static/sandbox repair loop (no silent template fallback)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from enigma.core.ids import new_id
from enigma.gateway.prompts import PromptRegistry, PromptTemplate
from enigma.gateway.protocol import StructuredGenerator, provenance_of
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import Budget, ExecutionMode, ModelProvenance
from enigma.schemas.quant import CodegenAttempt, StrategyGenerationRecord, StrategySpec
from enigma.schemas.research import CostModel
from enigma.strategy.package import build_package
from enigma.strategy.static_check import check_source

PROMPT_ID = "strategy.codegen"
METHOD = "llm:strategy-codegen-v1"
MAX_ATTEMPTS = 3


class GeneratedStrategySource(BaseModel):
    """Structured LLM output for one codegen attempt."""

    entrypoint: str = Field(min_length=1)
    source_code: str = Field(min_length=1)
    notes: str = ""


class StrategyCodeGenerator:
    """Generate a StrategyPackage from a StrategySpec via registered prompts.

    Failures are explicit (`ok=False` + `failure_reason`). Never substitutes canned source.
    """

    def __init__(
        self,
        llm: StructuredGenerator,
        prompts: PromptRegistry,
        *,
        prompt_id: str = PROMPT_ID,
        model: str | None = None,
        max_output_tokens: int = 2500,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.prompt_id = prompt_id
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.max_attempts = max_attempts

    def generate(
        self,
        spec: StrategySpec,
        *,
        cost_model: CostModel,
        actor: AgentRole = AgentRole.STRATEGY_ENGINEERING,
        task_id: str | None,
        budget: Budget,
        now: datetime,
        role: Literal["candidate", "baseline"] = "candidate",
        params: dict[str, float | int | str | bool] | None = None,
    ) -> StrategyGenerationRecord:
        prompt = self.prompts.get(self.prompt_id)
        attempts: list[CodegenAttempt] = []
        errors: list[str] = []
        package = None

        for index in range(1, self.max_attempts + 1):
            try:
                source, prov = self._call(prompt, spec, actor, task_id, budget, errors)
            except Exception as exc:
                attempts.append(
                    CodegenAttempt(
                        index=index,
                        entrypoint="",
                        source_hash="",
                        stage_reached="llm",
                        errors=(str(exc),),
                        provenance=None,
                    )
                )
                errors = [str(exc)]
                continue

            violations = check_source(source.source_code, source.entrypoint)
            if violations:
                detail = tuple(v.describe() for v in violations)
                attempts.append(
                    CodegenAttempt(
                        index=index,
                        entrypoint=source.entrypoint,
                        source_hash="",
                        stage_reached="static",
                        errors=detail,
                        provenance=prov if isinstance(prov, ModelProvenance) else None,
                    )
                )
                errors = list(detail)
                continue

            try:
                package = build_package(
                    source.source_code,
                    source.entrypoint,
                    params or {},
                    cost_model,
                    spec.hypothesis_id,
                    name=spec.name,
                    required_columns=spec.required_columns,
                    created_by=actor,
                    created_at=now,
                    label_horizon_bars=spec.label_horizon_bars,
                )
            except Exception as exc:
                attempts.append(
                    CodegenAttempt(
                        index=index,
                        entrypoint=source.entrypoint,
                        source_hash="",
                        stage_reached="static",
                        errors=(str(exc),),
                        provenance=prov if isinstance(prov, ModelProvenance) else None,
                    )
                )
                errors = [str(exc)]
                continue

            attempts.append(
                CodegenAttempt(
                    index=index,
                    entrypoint=source.entrypoint,
                    source_hash=package.content_hash,
                    stage_reached="passed",
                    errors=(),
                    provenance=prov if isinstance(prov, ModelProvenance) else None,
                )
            )
            break

        ok = package is not None
        return StrategyGenerationRecord(
            id=new_id("sgrec"),
            spec_id=spec.id,
            hypothesis_id=spec.hypothesis_id,
            role=role,
            execution_mode=ExecutionMode.LLM,
            prompt_ref=prompt.ref,
            attempts=tuple(attempts),
            ok=ok,
            package_id=package.id if package else None,
            package_hash=package.content_hash if package else None,
            checks={"static": ok},
            failure_reason=None if ok else "codegen_failed_after_repairs",
            created_by=actor,
            created_at=now,
        )

    def _call(
        self,
        prompt: PromptTemplate,
        spec: StrategySpec,
        actor: AgentRole,
        task_id: str | None,
        budget: Budget,
        errors: list[str],
    ) -> tuple[GeneratedStrategySource, ModelProvenance]:
        system, user = prompt.render(
            spec_json=json.dumps(spec.model_dump(mode="json"), indent=2)[:10000],
            prior_errors="; ".join(errors) if errors else "(none)",
        )
        traced = self.llm.generate_traced(
            GeneratedStrategySource,
            system,
            user,
            actor,
            task_id,
            budget,
            task_kind="strategy.codegen",
            model=self.model,
            max_output_tokens=self.max_output_tokens,
        )
        return traced.value, provenance_of(traced.response, prompt.ref)
