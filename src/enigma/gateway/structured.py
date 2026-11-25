"""`StructuredLLM`: typed generation for other packages.

Callers get either a validated instance of their Pydantic model or an exception. When no
provider is configured it raises `ProviderUnavailableError` *before* doing anything, so callers
can fall back to deterministic logic explicitly; it never returns a fabricated object.
"""

from __future__ import annotations

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
from enigma.schemas.governance import Budget

T = TypeVar("T", bound=BaseModel)


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
        """Return a validated `model_cls` instance.

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
        return response.parsed
