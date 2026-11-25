"""Structural interface of `StructuredLLM` for library callers, plus response provenance.

Library code that makes LLM calls types its dependency as `StructuredGenerator` so tests can
inject clearly labelled scripted test doubles while production passes a real `StructuredLLM`.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from enigma.gateway.structured import Traced
from enigma.gateway.types import ModelResponse, ReasoningEffort
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import Budget, ModelProvenance

T = TypeVar("T", bound=BaseModel)


class StructuredGenerator(Protocol):
    def generate_traced(
        self,
        model_cls: type[T],
        system: str,
        user: str,
        actor: AgentRole,
        task_id: str | None,
        budget: Budget,
        *,
        task_kind: str | None = None,
        model: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        max_output_tokens: int | None = None,
    ) -> Traced[T]: ...


def provenance_of(response: ModelResponse) -> ModelProvenance:
    return ModelProvenance(
        provider=response.provider,
        model_id=response.model_id,
        model_version=response.model_version,
        request_hash=response.request_hash,
        raw_response_hash=response.raw_response_hash,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=response.latency_ms,
        cost_usd=response.cost_usd,
        cached=response.cached,
        structured_mode=response.structured_mode,
        repaired=response.repaired,
        finish_reason=response.finish_reason.value,
    )
