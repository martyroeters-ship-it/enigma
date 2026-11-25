"""Decision providers. TypeSafe talks only to api.typesafe.ai and only with a key."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from enigma.core.errors import ProviderUnavailableError
from enigma.decisions.tasks import DecisionItem

OFFICIAL_HOST = "api.typesafe.ai"


@dataclass(frozen=True)
class DecisionResult:
    choice: str
    probability: float
    latency_ms: float
    cost_usd: float


class DeterministicDecisionProvider:
    provider_id = "deterministic"

    def decide(self, item: DecisionItem) -> DecisionResult:
        choice = item.text.split(maxsplit=1)[0]
        return DecisionResult(choice=choice, probability=0.9, latency_ms=0.1, cost_usd=0.0)


class LlmDecisionProvider:
    """Calls an injected completion. This class does not open a network connection."""

    provider_id = "llm"

    def __init__(
        self, complete: Callable[[str], str], *, latency_ms: float, cost_usd: float
    ) -> None:
        self._complete = complete
        self._latency_ms = latency_ms
        self._cost_usd = cost_usd

    def decide(self, item: DecisionItem) -> DecisionResult:
        choice = self._complete(item.text).strip().split(maxsplit=1)[0]
        return DecisionResult(
            choice=choice,
            probability=0.6,
            latency_ms=self._latency_ms,
            cost_usd=self._cost_usd,
        )


class TypeSafeDecisionProvider:
    provider_id = "typesafe"

    def __init__(
        self,
        *,
        api_key: str | None,
        host: str = OFFICIAL_HOST,
        transport: Callable[[str, str], str] | None = None,
    ) -> None:
        if host != OFFICIAL_HOST:
            raise ValueError("typesafe host must be api.typesafe.ai")
        self._api_key = api_key
        self._transport = transport

    def decide(self, item: DecisionItem) -> DecisionResult:
        if not self._api_key:
            raise ProviderUnavailableError("TYPESAFE_API_KEY is not set")
        if self._transport is None:
            raise ProviderUnavailableError("no typesafe transport configured")
        choice = self._transport(OFFICIAL_HOST, item.text)
        return DecisionResult(choice=choice, probability=0.5, latency_ms=0.0, cost_usd=0.0)
