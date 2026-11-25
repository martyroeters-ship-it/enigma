"""`StructuredLLM`: typed generation for other packages.

Callers get either a validated instance of their Pydantic model or an exception. When no
provider is configured it raises `ProviderUnavailableError` *before* doing anything, so callers
can fall back to deterministic logic explicitly; it never returns a fabricated object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from enigma.core.errors import ProviderUnavailableError
from enigma.gateway.errors import StructuredOutputError
from enigma.gateway.gateway import ModelGateway
from enigma.gateway.types import (
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ReasoningEffort,
    RequestMetadata,
)
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import Budget, ModelProvenance

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Traced[V: BaseModel]:
    """A validated structured value plus the gateway response that produced it."""

    value: V
    response: ModelResponse


def provenance_of(response: ModelResponse, prompt_ref: str | None = None) -> ModelProvenance:
    return ModelProvenance(
        provider=response.provider,
        model_id=response.model_id,
        model_version=response.model_version,
        request_hash=response.request_hash,
        raw_response_hash=response.raw_response_hash,
        prompt_ref=prompt_ref,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
        cached=response.cached,
    )


class StructuredLLM:
    def __init__(self, gateway: ModelGateway) -> None:
        self.gateway = gateway
        self.last_response: ModelResponse | None = None

    def is_available(self) -> bool:
        return self.gateway.has_available_provider()

    def generate(
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
    ) -> T:
        return self.generate_traced(
            model_cls,
            system,
            user,
            actor,
            task_id,
            budget,
            task_kind=task_kind,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
        ).value

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
    ) -> Traced[T]:
        """Return a validated `model_cls` instance with its response provenance.

        Raises `ProviderUnavailableError` (nothing configured / all candidates failed),
        `BudgetExceededError`, or `StructuredOutputError`.
        """
        if not self.is_available():
            missing = self.gateway.missing_credentials()
            hint = "; ".join(f"{p}: {' | '.join(v)}" for p, v in sorted(missing.items()))
            raise ProviderUnavailableError(
                "no model provider is configured or reachable"
                + (f" (set one of — {hint})" if hint else "")
            )
        request = ModelRequest(
            messages=(
                Message(role=MessageRole.SYSTEM, content=system),
                Message(role=MessageRole.USER, content=user),
            ),
            model=model,
            response_schema=model_cls,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            metadata=RequestMetadata(agent_role=actor, task_id=task_id, task_kind=task_kind),
        )
        response = self.gateway.complete(request, actor, task_id, budget)
        self.last_response = response
        if not isinstance(response.parsed, model_cls):
            raise StructuredOutputError(
                f"model returned no {model_cls.__name__} (finish={response.finish_reason})"
            )
        return Traced(response.parsed, response)
