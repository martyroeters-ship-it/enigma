"""Model, pricing, and routing registries (`registries/models/*.yaml`).

The registry is versioned by a content hash of all three files so every response and cost
record can be traced to the exact routing/pricing configuration that produced it.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, ValidationError, model_validator

from enigma.core.hashing import sha256_of
from enigma.gateway.errors import RegistryError
from enigma.gateway.types import ReasoningEffort
from enigma.schemas.common import AgentRole, NonEmptyStr, StrictModel


class ProviderKind(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENAI_COMPATIBLE = "openai_compatible"
    OLLAMA = "ollama"


class StructuredMode(StrEnum):
    JSON_SCHEMA = "json_schema"
    JSON_MODE = "json_mode"
    PROMPT = "prompt"


class ReasoningStyle(StrEnum):
    OPENAI_EFFORT = "openai_effort"
    OPENROUTER = "openrouter"
    ANTHROPIC_BUDGET = "anthropic_budget"
    GEMINI_BUDGET = "gemini_budget"
    OLLAMA_THINK = "ollama_think"


class ProviderSpec(StrictModel):
    name: NonEmptyStr
    kind: ProviderKind
    base_url: str | None = None
    base_url_env: str | None = Field(
        default=None, description="Env var overriding base_url; required if base_url is unset"
    )
    api_key_env: tuple[str, ...] = Field(
        default=(), description="Any one of these env vars supplies the key"
    )
    key_required: bool = True
    local: bool = Field(default=False, description="Runs on operator hardware; priced at zero")
    probe_path: str | None = Field(
        default=None, description="Local providers: GET path used to check reachability"
    )
    timeout_s: float = Field(default=120.0, gt=0)
    max_tokens_param: str = "max_tokens"
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> ProviderSpec:
        if self.base_url is None and self.base_url_env is None:
            raise ValueError(f"provider {self.name}: base_url or base_url_env required")
        if self.key_required and not self.api_key_env:
            raise ValueError(f"provider {self.name}: key_required but no api_key_env")
        return self

    def credential_env_vars(self) -> tuple[str, ...]:
        """Env vars an operator must set to enable this provider (empty if none)."""
        needed: list[str] = []
        if self.key_required:
            needed.extend(self.api_key_env)
        if self.base_url is None and self.base_url_env:
            needed.append(self.base_url_env)
        return tuple(needed)


class Capabilities(StrictModel):
    structured_output: StructuredMode = StructuredMode.PROMPT
    tools: bool = False
    reasoning: ReasoningStyle | None = None
    reasoning_efforts: tuple[str, ...] = Field(
        default=(), description="Provider-native effort values supported, lowest first"
    )
    temperature: bool = Field(default=True, description="False if the model rejects it")


class ModelSpec(StrictModel):
    alias: NonEmptyStr
    provider: NonEmptyStr
    model_id: NonEmptyStr
    context_window: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    capabilities: Capabilities = Field(default_factory=Capabilities)
    notes: str = ""

    @property
    def concrete(self) -> str:
        return f"{self.provider}/{self.model_id}"


class Price(StrictModel):
    input: float = Field(ge=0, description="USD per 1M input tokens")
    output: float = Field(ge=0, description="USD per 1M output tokens (incl. reasoning)")
    cached_input: float | None = Field(default=None, ge=0)
    source: NonEmptyStr


class PricingTable(StrictModel):
    as_of: date
    currency: str = "USD"
    estimate: bool = True
    notice: str = ""
    prices: dict[str, Price] = Field(description="Keyed by provider/model_id")


class Route(StrictModel):
    candidates: tuple[str, ...] | None = None
    reasoning_effort: ReasoningEffort | None = None
    max_cost_per_call_usd: float | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, gt=0)


class RoleRoutes(StrictModel):
    default: Route = Field(default_factory=Route)
    tasks: dict[str, Route] = Field(default_factory=dict)


class RoutingTable(StrictModel):
    defaults: Route
    roles: dict[AgentRole, RoleRoutes] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _defaults_complete(self) -> RoutingTable:
        d = self.defaults
        if (
            not d.candidates
            or d.max_cost_per_call_usd is None
            or d.max_output_tokens is None
            or d.reasoning_effort is None
        ):
            raise ValueError("routing defaults must set every field")
        return self


class ResolvedRoute(StrictModel):
    candidates: tuple[str, ...] = Field(min_length=1)
    reasoning_effort: ReasoningEffort
    max_cost_per_call_usd: float = Field(ge=0)
    max_output_tokens: int = Field(gt=0)
    source: str


def _load_yaml(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise RegistryError(f"registry file missing: {path}") from exc
    except yaml.YAMLError as exc:
        raise RegistryError(f"invalid YAML in {path}: {exc}") from exc


class ModelRegistry:
    """Validated, hash-versioned view of models.yaml + pricing.yaml + routing.yaml."""

    def __init__(
        self,
        providers: dict[str, ProviderSpec],
        models: dict[str, ModelSpec],
        pricing: PricingTable,
        routing: RoutingTable,
        version: str,
    ) -> None:
        self.providers = providers
        self.models = models
        self.pricing = pricing
        self.routing = routing
        self.version = version
        self._validate()

    @classmethod
    def from_documents(cls, models_doc: Any, pricing_doc: Any, routing_doc: Any) -> ModelRegistry:
        try:
            providers = {
                name: ProviderSpec.model_validate({"name": name, **body})
                for name, body in (models_doc.get("providers") or {}).items()
            }
            models = {
                alias: ModelSpec.model_validate({"alias": alias, **body})
                for alias, body in (models_doc.get("models") or {}).items()
            }
            pricing = PricingTable.model_validate(pricing_doc)
            routing = RoutingTable.model_validate(routing_doc)
        except (ValidationError, AttributeError, TypeError) as exc:
            raise RegistryError(f"invalid model registry: {exc}") from exc
        version = sha256_of({"models": models_doc, "pricing": pricing_doc, "routing": routing_doc})
        return cls(providers, models, pricing, routing, version)

    @classmethod
    def load(cls, directory: Path) -> ModelRegistry:
        return cls.from_documents(
            _load_yaml(directory / "models.yaml"),
            _load_yaml(directory / "pricing.yaml"),
            _load_yaml(directory / "routing.yaml"),
        )

    @classmethod
    def load_default(cls) -> ModelRegistry:
        from enigma.core.config import Settings  # noqa: PLC0415 - avoid env read at import

        return cls.load(Settings.from_env().registries_dir / "models")

    def _validate(self) -> None:
        for alias, spec in self.models.items():
            if spec.provider not in self.providers:
                raise RegistryError(f"model {alias!r} references unknown provider {spec.provider}")
            if not spec.capabilities.reasoning_efforts and spec.capabilities.reasoning in (
                ReasoningStyle.OPENAI_EFFORT,
                ReasoningStyle.OPENROUTER,
            ):
                raise RegistryError(f"model {alias!r}: reasoning_efforts required")
        for key in self.pricing.prices:
            provider, sep, _ = key.partition("/")
            if not sep or provider not in self.providers:
                raise RegistryError(f"pricing key {key!r} must be <known provider>/<model_id>")
        routes: list[tuple[str, Route]] = [("defaults", self.routing.defaults)]
        for role, rr in self.routing.roles.items():
            routes.append((f"{role}.default", rr.default))
            routes.extend((f"{role}.{kind}", r) for kind, r in rr.tasks.items())
        for where, route in routes:
            for alias in route.candidates or ():
                if alias not in self.models:
                    raise RegistryError(f"route {where} references unknown alias {alias!r}")

    def price_for(self, spec: ModelSpec) -> Price | None:
        if self.providers[spec.provider].local:
            return Price(input=0.0, output=0.0, source="local")
        return self.pricing.prices.get(spec.concrete)

    def resolve_model(self, ref: str) -> ModelSpec:
        """Resolve an alias or a concrete ``provider/model_id`` reference."""
        if ref in self.models:
            return self.models[ref]
        for spec in self.models.values():
            if spec.concrete == ref:
                return spec
        provider, sep, model_id = ref.partition("/")
        if sep and provider in self.providers and model_id:
            return ModelSpec(
                alias=ref,
                provider=provider,
                model_id=model_id,
                context_window=8192,
                max_output_tokens=4096,
                notes="unregistered concrete pin; conservative capabilities",
            )
        raise RegistryError(f"unknown model reference {ref!r}")

    def route(self, role: AgentRole | None, task_kind: str | None) -> ResolvedRoute:
        layers: list[tuple[str, Route]] = []
        if role is not None and role in self.routing.roles:
            rr = self.routing.roles[role]
            if task_kind and task_kind in rr.tasks:
                layers.append((f"{role}:{task_kind}", rr.tasks[task_kind]))
            layers.append((f"{role}:default", rr.default))
        layers.append(("defaults", self.routing.defaults))

        def pick(attr: str) -> Any:
            for _, layer in layers:
                value = getattr(layer, attr)
                if value is not None:
                    return value
            return None

        return ResolvedRoute(
            candidates=pick("candidates"),
            reasoning_effort=pick("reasoning_effort"),
            max_cost_per_call_usd=pick("max_cost_per_call_usd"),
            max_output_tokens=pick("max_output_tokens"),
            source=layers[0][0],
        )
