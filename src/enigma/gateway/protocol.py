"""Structural interface of `StructuredLLM` for library callers.

Library code that makes LLM calls types its dependency as `StructuredGenerator` so tests can
inject clearly labelled scripted test doubles while production passes a real `StructuredLLM`.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from enigma.gateway.structured import Traced, provenance_of
from enigma.gateway.types import ReasoningEffort
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import Budget

T = TypeVar("T", bound=BaseModel)

__all__ = ["StructuredGenerator", "provenance_of"]


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
