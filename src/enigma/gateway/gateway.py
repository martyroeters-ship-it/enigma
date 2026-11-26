"""`ModelGateway`: the single entry point for model calls.

Fail-closed guarantees:
- No credentials / unreachable local server -> `ProviderUnavailableError` (never a fake reply).
- Estimated cost or tokens beyond the caller's `Budget`, the route's per-call ceiling, or the
  global daily ceiling (`ENIGMA_MAX_DAILY_API_SPEND_USD`, default 0 = no paid calls) ->
  `BudgetExceededError` before any HTTP request is made.
- Paid models without a pricing entry cannot be budget-checked and are refused.
- Every provider invocation (success, failure, cache hit) writes a `CostRecord` (category "llm").
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from enigma.core.clock import Clock
from enigma.core.errors import BudgetExceededError, ProviderUnavailableError
from enigma.core.ids import new_id
from enigma.gateway import jsonschema
from enigma.gateway.cache import ResponseCache, cache_key
from enigma.gateway.credentials import first_present, redact, secret_values
from enigma.gateway.errors import ProviderCallError, StructuredOutputError
from enigma.gateway.providers import (
    ProviderAdapter,
    ProviderCall,
    billable_output_cap,
    make_adapter,
)
from enigma.gateway.providers.base import ProviderResult
from enigma.gateway.registry import ModelRegistry, ModelSpec, Price, ProviderSpec
from enigma.gateway.types import (
    FinishReason,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ReasoningEffort,
)
from enigma.schemas.common import AgentRole, StrictModel
from enigma.schemas.governance import Budget, CostRecord
from enigma.store.ledger import CostLedger

DAILY_CAP_ENV = "ENIGMA_MAX_DAILY_API_SPEND_USD"
LLM_CATEGORY = "llm"
USER_AGENT = "Enigma-Research/0.1 (+https://github.com/martyroeters-ship-it/enigma)"

_TOKEN_SAFETY = 1.25
_BYTES_PER_TOKEN = 3.0
_PER_MESSAGE_OVERHEAD = 8
_PROBE_TTL_S = 60.0
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _provider_seconds(result: ProviderResult) -> float:
    """Ollama reports ``total_duration`` in nanoseconds. Other providers leave this at 0."""
    raw = result.extra.get("provider_seconds") if result.extra else 0
    try:
        value = float(raw or 0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or value < 0:
        return 0.0
    return value


def _empty_native_json(result: ProviderResult, errors: list[str]) -> bool:
    """Native grammar returned nothing the JSON parser can read."""
    if result.structured_mode != "native":
        return False
    if not (result.text or "").strip():
        return True
    return any("Expecting value" in err or "not valid JSON" in err for err in errors)


def _combine_results(first: ProviderResult, second: ProviderResult) -> ProviderResult:
    return replace(
        second,
        usage=second.usage.model_copy(
            update={
                "input_tokens": first.usage.input_tokens + second.usage.input_tokens,
                "output_tokens": first.usage.output_tokens + second.usage.output_tokens,
                "reasoning_tokens": first.usage.reasoning_tokens + second.usage.reasoning_tokens,
            }
        ),
    )


class FailureKind(StrEnum):
    MISSING_CREDENTIALS = "missing_credentials"
    UNREACHABLE = "unreachable"
    CAPABILITY = "capability"
    BUDGET = "budget"
    PROVIDER_ERROR = "provider_error"
    STRUCTURED_OUTPUT = "structured_output"


class CandidateFailure(StrictModel):
    candidate: str
    provider: str
    kind: FailureKind
    detail: str


class ProviderStatus(StrictModel):
    provider: str
    kind: str
    available: bool
    local: bool
    env_vars: tuple[str, ...] = ()
    missing_env_vars: tuple[str, ...] = ()
    detail: str = ""


class GatewayUnavailableError(ProviderUnavailableError):
    """No candidate could serve the request. Carries per-candidate reasons."""

    def __init__(self, message: str, failures: list[CandidateFailure]) -> None:
        super().__init__(message)
        self.failures = failures


class _Refusal(Exception):  # noqa: N818 - internal control flow, never escapes the gateway
    def __init__(self, kind: FailureKind, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


@dataclass(frozen=True)
class _Estimate:
    input_tokens: int
    output_tokens: int
    usd: float


def estimate_input_tokens(request: ModelRequest) -> int:
    """Conservative token estimate: UTF-8 bytes / 3, plus 25% and per-message overhead."""
    size = sum(len(m.content.encode()) for m in request.messages)
    size += sum(len(tc.raw_arguments.encode()) for m in request.messages for tc in m.tool_calls)
    schema = request.json_schema()
    if schema is not None:
        size += 2 * len(json.dumps(schema).encode())
    size += sum(len(t.model_dump_json().encode()) for t in request.tools)
    tokens = size / _BYTES_PER_TOKEN * _TOKEN_SAFETY
    return math.ceil(tokens) + _PER_MESSAGE_OVERHEAD * (len(request.messages) + 2)


def cost_usd(price: Price, input_tokens: int, output_tokens: int, cached_input: int = 0) -> float:
    cached_rate = price.cached_input if price.cached_input is not None else price.input
    fresh = max(0, input_tokens - cached_input)
    return (
        fresh * price.input + cached_input * cached_rate + output_tokens * price.output
    ) / 1_000_000


def _extract_json(text: str) -> Any:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        stripped = match.group(1)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start != -1 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def validate_structured(
    text: str, schema: type[BaseModel] | dict[str, Any]
) -> tuple[Any, list[str]]:
    """Parse and validate model output. Returns (parsed, errors)."""
    try:
        data = _extract_json(text)
    except json.JSONDecodeError as exc:
        return None, [f"output is not valid JSON: {exc.msg} at char {exc.pos}"]
    if isinstance(schema, dict):
        errors = jsonschema.validate(data, schema)
        return (data if not errors else None), errors
    try:
        return schema.model_validate(data), []
    except ValidationError as exc:
        return None, [
            f"{'.'.join(str(p) for p in e['loc']) or '$'}: {e['msg']}" for e in exc.errors()
        ]


def _normalize_base_url(url: str) -> str:
    return url if "://" in url else f"http://{url}"


class ModelGateway:
    def __init__(
        self,
        registry: ModelRegistry,
        ledger: CostLedger,
        cache_db_path: Path | str | None,
        clock: Clock,
        *,
        environ: Mapping[str, str] | None = None,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
    ) -> None:
        self.registry = registry
        self.ledger = ledger
        self.clock = clock
        self._environ = environ
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            headers={"user-agent": USER_AGENT}, follow_redirects=False
        )
        self._sleep = sleep
        self._rng = rng or random.Random()  # noqa: S311 - jitter only, not security
        self._monotonic = monotonic
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.cache = ResponseCache(cache_db_path) if cache_db_path is not None else None
        self._adapters: dict[str, ProviderAdapter] = {}
        self._probe_cache: dict[str, tuple[float, bool, str]] = {}

    @property
    def environ(self) -> Mapping[str, str]:
        return self._environ if self._environ is not None else os.environ

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
        if self.cache is not None:
            self.cache.close()

    # ------------------------------------------------------------------ credentials

    def _base_url(self, provider: ProviderSpec) -> str | None:
        if provider.base_url_env:
            override = self.environ.get(provider.base_url_env, "").strip()
            if override:
                return _normalize_base_url(override)
        return provider.base_url

    def _probe(self, provider: ProviderSpec, base_url: str) -> tuple[bool, str]:
        if provider.probe_path is None:
            return True, ""
        now = self._monotonic()
        cached = self._probe_cache.get(provider.name)
        if cached is not None and now - cached[0] < _PROBE_TTL_S:
            return cached[1], cached[2]
        url = f"{base_url.rstrip('/')}{provider.probe_path}"
        try:
            resp = self._client.get(url, timeout=2.0)
            ok, detail = resp.status_code < 400, f"probe {url} -> HTTP {resp.status_code}"
        except httpx.HTTPError as exc:
            ok, detail = False, f"not reachable at {base_url} ({type(exc).__name__})"
        self._probe_cache[provider.name] = (now, ok, detail)
        return ok, detail

    def provider_status(self, name: str) -> ProviderStatus:
        provider = self.registry.providers[name]
        needed = provider.credential_env_vars()
        missing: list[str] = []
        if provider.key_required and not first_present(self.environ, provider.api_key_env):
            missing.extend(provider.api_key_env)
        base_url = self._base_url(provider)
        if base_url is None and provider.base_url_env:
            missing.append(provider.base_url_env)
        if missing:
            return ProviderStatus(
                provider=name,
                kind=provider.kind.value,
                available=False,
                local=provider.local,
                env_vars=needed,
                missing_env_vars=tuple(missing),
                detail="set " + " or ".join(missing),
            )
        assert base_url is not None
        ok, detail = self._probe(provider, base_url)
        return ProviderStatus(
            provider=name,
            kind=provider.kind.value,
            available=ok,
            local=provider.local,
            env_vars=needed,
            detail=detail or "configured",
        )

    def credential_report(self) -> list[ProviderStatus]:
        return [self.provider_status(n) for n in sorted(self.registry.providers)]

    def available_providers(self) -> list[str]:
        return [s.provider for s in self.credential_report() if s.available]

    def has_available_provider(self) -> bool:
        """Cheap check: key-based providers first, local servers probed only if needed."""
        names = sorted(self.registry.providers, key=lambda n: self.registry.providers[n].local)
        return any(self.provider_status(n).available for n in names)

    def missing_credentials(self) -> dict[str, tuple[str, ...]]:
        """Provider -> env vars that would enable it (any one key var suffices)."""
        return {
            s.provider: s.missing_env_vars for s in self.credential_report() if s.missing_env_vars
        }

    def available_models(self) -> list[ModelSpec]:
        available = set(self.available_providers())
        return [m for m in self.registry.models.values() if m.provider in available]

    # ------------------------------------------------------------------ budget

    def daily_cap_usd(self) -> float:
        raw = self.environ.get(DAILY_CAP_ENV, "0").strip() or "0"
        try:
            value = float(raw)
        except ValueError as exc:
            raise BudgetExceededError(
                f"{DAILY_CAP_ENV} is not a number; refusing paid calls"
            ) from exc
        if not math.isfinite(value) or value < 0:
            raise BudgetExceededError(f"{DAILY_CAP_ENV} must be a finite number >= 0")
        return value

    def _day_start(self) -> datetime:
        now = self.clock.now().astimezone(UTC)
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    def _spent(self, actor: AgentRole, task_id: str | None) -> dict[str, float]:
        """Committed spend plus open reservations (ceilings cover both)."""
        self.ledger.expire_stale_reservations(now=self.clock.now())
        if task_id is not None:
            return self.ledger.exposure(actor=str(actor), task_id=task_id)
        return self.ledger.exposure(actor=str(actor), since=self._day_start())

    def _check_budget(
        self,
        estimate: _Estimate,
        actor: AgentRole,
        task_id: str | None,
        budget: Budget,
        max_cost_per_call: float,
    ) -> int:
        """Raise `_Refusal` if the call may not proceed; return remaining HTTP allowance."""
        spent = self._spent(actor, task_id)
        tokens = estimate.input_tokens + estimate.output_tokens
        remaining_tokens = budget.max_tokens - spent["tokens"]
        if tokens > remaining_tokens:
            raise _Refusal(
                FailureKind.BUDGET,
                f"estimated {tokens} tokens exceeds remaining token budget {remaining_tokens:.0f}",
            )
        http_left = int(budget.max_http_requests - spent["http_requests"])
        if http_left < 1:
            raise _Refusal(FailureKind.BUDGET, "HTTP request budget exhausted")
        if estimate.usd > 0:
            if estimate.usd > max_cost_per_call:
                raise _Refusal(
                    FailureKind.BUDGET,
                    f"estimated ${estimate.usd:.6f} exceeds route max per call "
                    f"${max_cost_per_call:.6f}",
                )
            remaining_usd = budget.max_usd - spent["usd"]
            if estimate.usd > remaining_usd:
                raise _Refusal(
                    FailureKind.BUDGET,
                    f"estimated ${estimate.usd:.6f} exceeds remaining budget ${remaining_usd:.6f}",
                )
            cap = self.daily_cap_usd()
            day_spent = self.ledger.exposure(since=self._day_start())["usd"]
            if day_spent + estimate.usd > cap:
                raise _Refusal(
                    FailureKind.BUDGET,
                    f"daily API ceiling: spent ${day_spent:.6f} + estimated ${estimate.usd:.6f}"
                    f" > {DAILY_CAP_ENV}=${cap:.2f}",
                )
        return http_left

    # ------------------------------------------------------------------ ledger

    def _cost_record(
        self,
        actor: AgentRole,
        task_id: str | None,
        provider: str,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        usd: float = 0.0,
        http_requests: int = 0,
        cpu_seconds: float = 0.0,
        cached: bool = False,
    ) -> CostRecord:
        return CostRecord(
            id=new_id("cst"),
            actor=actor,
            task_id=task_id,
            category=LLM_CATEGORY,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=usd,
            http_requests=http_requests,
            cpu_seconds=cpu_seconds,
            cached=cached,
            created_at=self.clock.now(),
        )

    def _record(
        self,
        actor: AgentRole,
        task_id: str | None,
        provider: str,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        usd: float = 0.0,
        http_requests: int = 0,
        cpu_seconds: float = 0.0,
        cached: bool = False,
        reservation_id: str | None = None,
    ) -> None:
        cost = self._cost_record(
            actor,
            task_id,
            provider,
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=usd,
            http_requests=http_requests,
            cpu_seconds=cpu_seconds,
            cached=cached,
        )
        if reservation_id is not None:
            self.ledger.commit_reservation(reservation_id, cost)
        else:
            self.ledger.record(cost)

    def _reserve_call(
        self,
        estimate: _Estimate,
        actor: AgentRole,
        task_id: str | None,
        budget: Budget,
        provider: str,
        model: str,
        http_left: int,
    ) -> str:
        """Atomically reserve worst-case spend before any provider HTTP."""
        reservation_id = new_id("rsv")
        try:
            self.ledger.reserve(
                reservation_id=reservation_id,
                actor=str(actor),
                task_id=task_id,
                category=LLM_CATEGORY,
                provider=provider,
                model=model,
                usd=estimate.usd,
                tokens=estimate.input_tokens + estimate.output_tokens,
                http_requests=min(self.max_retries + 1, max(1, http_left)),
                max_usd=budget.max_usd,
                max_tokens=budget.max_tokens,
                max_http=budget.max_http_requests,
                daily_cap_usd=self.daily_cap_usd() if estimate.usd > 0 else None,
                day_start=self._day_start(),
                now=self.clock.now(),
            )
        except BudgetExceededError as exc:
            raise _Refusal(FailureKind.BUDGET, str(exc)) from exc
        return reservation_id

    # ------------------------------------------------------------------ invocation

    def _adapter(self, provider: ProviderSpec) -> ProviderAdapter:
        if provider.name not in self._adapters:
            self._adapters[provider.name] = make_adapter(provider.kind, self._client)
        return self._adapters[provider.name]

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        ceiling = min(self.backoff_max_s, self.backoff_base_s * 2.0 ** (attempt - 1))
        delay = ceiling / 2 + self._rng.uniform(0, ceiling / 2)
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.backoff_max_s))
        return delay

    def _invoke(
        self, adapter: ProviderAdapter, call: ProviderCall, http_left: int
    ) -> tuple[ProviderResult, int]:
        attempt = 0
        while True:
            attempt += 1
            try:
                return adapter.invoke(call), attempt
            except ProviderCallError as exc:
                exc.attempts = attempt
                if not exc.retryable or attempt > self.max_retries or attempt >= http_left:
                    raise
                self._sleep(self._backoff(attempt, exc.retry_after_s))

    def _estimate(self, call: ProviderCall, price: Price) -> _Estimate:
        inp = estimate_input_tokens(call.request)
        out = billable_output_cap(call)
        return _Estimate(inp, out, cost_usd(price, inp, out))

    def _run_call(
        self,
        call: ProviderCall,
        price: Price,
        actor: AgentRole,
        task_id: str | None,
        budget: Budget,
        max_cost_per_call: float,
    ) -> tuple[ProviderResult, int, float, float]:
        estimate = self._estimate(call, price)
        if estimate.input_tokens > call.model.context_window:
            raise _Refusal(
                FailureKind.CAPABILITY,
                f"estimated {estimate.input_tokens} input tokens exceed context window "
                f"{call.model.context_window}",
            )
        http_left = self._check_budget(estimate, actor, task_id, budget, max_cost_per_call)
        reservation_id = self._reserve_call(
            estimate,
            actor,
            task_id,
            budget,
            call.provider.name,
            call.model.model_id,
            http_left,
        )
        adapter = self._adapter(call.provider)
        started = self._monotonic()
        try:
            result, attempts = self._invoke(adapter, call, http_left)
        except ProviderCallError as exc:
            self._record(
                actor,
                task_id,
                call.provider.name,
                call.model.model_id,
                http_requests=exc.attempts,
                reservation_id=reservation_id,
            )
            raise
        except BaseException:
            self.ledger.release_reservation(reservation_id, now=self.clock.now())
            raise
        latency_ms = (self._monotonic() - started) * 1000.0
        usd = cost_usd(
            price,
            result.usage.input_tokens,
            result.usage.output_tokens,
            result.usage.cached_input_tokens,
        )
        self._record(
            actor,
            task_id,
            call.provider.name,
            result.model_version,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            usd=usd,
            http_requests=attempts,
            cpu_seconds=_provider_seconds(result),
            reservation_id=reservation_id,
        )
        return result, attempts, latency_ms, usd

    def complete(
        self, request: ModelRequest, actor: AgentRole, task_id: str | None, budget: Budget
    ) -> ModelResponse:
        """Route, budget-check, call, validate, cache, and record one completion."""
        route = self.registry.route(actor, request.metadata.task_kind)
        candidates = (request.model,) if request.model else route.candidates
        failures: list[CandidateFailure] = []
        for ref in candidates:
            spec = self.registry.resolve_model(ref)
            provider = self.registry.providers[spec.provider]
            try:
                return self._complete_with(
                    request, spec, provider, route.max_cost_per_call_usd,
                    route.max_output_tokens, route.reasoning_effort, actor, task_id, budget,
                )  # fmt: skip
            except _Refusal as refusal:
                failures.append(
                    CandidateFailure(
                        candidate=ref,
                        provider=provider.name,
                        kind=refusal.kind,
                        detail=refusal.detail,
                    )
                )
            except ProviderCallError as exc:
                failures.append(
                    CandidateFailure(
                        candidate=ref,
                        provider=provider.name,
                        kind=FailureKind.PROVIDER_ERROR,
                        detail=str(exc),
                    )
                )
        raise self._aggregate(failures, route.source if not request.model else "pinned")

    def _aggregate(self, failures: list[CandidateFailure], source: str) -> Exception:
        summary = "; ".join(f"{f.candidate} [{f.kind}]: {f.detail}" for f in failures)
        message = f"no model candidate succeeded (route {source}): {summary}"
        kinds = {f.kind for f in failures}
        if FailureKind.STRUCTURED_OUTPUT in kinds:
            return StructuredOutputError(message)
        if FailureKind.BUDGET in kinds:
            return BudgetExceededError(message)
        return GatewayUnavailableError(message, failures)

    def _complete_with(
        self,
        request: ModelRequest,
        spec: ModelSpec,
        provider: ProviderSpec,
        max_cost_per_call: float,
        default_max_output: int,
        default_effort: ReasoningEffort,
        actor: AgentRole,
        task_id: str | None,
        budget: Budget,
    ) -> ModelResponse:
        if request.tools and not spec.capabilities.tools:
            raise _Refusal(FailureKind.CAPABILITY, "model does not support tool calling")
        schema = request.json_schema()
        req_hash = request.request_hash()

        key: str | None = None
        if self.cache is not None and request.is_cacheable:
            key = cache_key(req_hash, provider.name, spec.model_id, self.registry.version)
            hit = self._cache_hit(key, request, actor, task_id)
            if hit is not None:
                return hit

        status = self.provider_status(provider.name)
        if not status.available:
            kind = (
                FailureKind.MISSING_CREDENTIALS
                if status.missing_env_vars
                else FailureKind.UNREACHABLE
            )
            raise _Refusal(kind, status.detail)
        price = self.registry.price_for(spec)
        if price is None:
            raise _Refusal(
                FailureKind.BUDGET,
                f"no pricing entry for paid model {spec.concrete}; cannot enforce budget",
            )
        base_url = self._base_url(provider)
        assert base_url is not None
        call = ProviderCall(
            request=request,
            model=spec,
            provider=provider,
            base_url=base_url,
            api_key=first_present(self.environ, provider.api_key_env),
            output_token_cap=min(
                request.max_output_tokens or default_max_output, spec.max_output_tokens
            ),
            reasoning_effort=request.reasoning_effort or default_effort,
            json_schema=schema,
            secrets=tuple(secret_values(self.environ, provider.api_key_env)),
        )
        result, attempts, latency_ms, usd = self._run_call(
            call, price, actor, task_id, budget, max_cost_per_call
        )
        parsed: Any = None
        repaired = False
        if request.response_schema is not None and not result.tool_calls:
            parsed, errors = self._validate_result(result, request.response_schema)
            if errors and _empty_native_json(result, errors):
                json_call = replace(call, structured_format="json")
                result2, attempts2, latency2, usd2 = self._run_call(
                    json_call, price, actor, task_id, budget, max_cost_per_call
                )
                parsed, errors = self._validate_result(result2, request.response_schema)
                result = _combine_results(result, result2)
                attempts += attempts2
                latency_ms += latency2
                usd += usd2
            if errors:
                repair_format = (
                    "json"
                    if result.structured_mode in {"native", "json"}
                    or call.structured_format == "json"
                    else call.structured_format
                )
                repair_call = replace(
                    call,
                    request=self._repair_request(request, result.text, errors),
                    structured_format=repair_format,
                )
                result2, attempts2, latency2, usd2 = self._run_call(
                    repair_call, price, actor, task_id, budget, max_cost_per_call
                )
                parsed, errors2 = self._validate_result(result2, request.response_schema)
                if errors2:
                    raise _Refusal(
                        FailureKind.STRUCTURED_OUTPUT,
                        redact("; ".join(errors2[:10]), call.secrets),
                    )
                result = _combine_results(result, result2)
                attempts += attempts2
                latency_ms += latency2
                usd += usd2
                repaired = True

        response = ModelResponse(
            text=result.text,
            parsed=parsed,
            tool_calls=result.tool_calls,
            usage=result.usage,
            provider=provider.name,
            alias=spec.alias,
            model_id=spec.model_id,
            model_version=result.model_version,
            latency_ms=latency_ms,
            cost_usd=usd,
            cost_is_estimate=self.registry.pricing.estimate and not provider.local,
            cached=False,
            finish_reason=result.finish_reason,
            raw_finish_reason=result.raw_finish_reason,
            raw_response_hash=result.raw_hash,
            request_hash=req_hash,
            reasoning_effort_applied=result.reasoning_applied,
            ignored_parameters=result.ignored,
            structured_mode=result.structured_mode,
            repaired=repaired,
            attempts=attempts,
            registry_version=self.registry.version,
        )
        if (
            key is not None
            and self.cache is not None
            and response.finish_reason
            in (
                FinishReason.STOP,
                FinishReason.TOOL_CALLS,
            )
        ):
            self.cache.put(key, response, self.clock.now())
        return response

    def _validate_result(
        self, result: ProviderResult, schema: type[BaseModel] | dict[str, Any]
    ) -> tuple[Any, list[str]]:
        if result.finish_reason is FinishReason.LENGTH:
            return None, ["output truncated (finish_reason=length); raise max_output_tokens"]
        if result.finish_reason is FinishReason.CONTENT_FILTER:
            return None, ["output blocked by provider content filter / refusal"]
        return validate_structured(result.text, schema)

    @staticmethod
    def _repair_request(request: ModelRequest, bad_text: str, errors: list[str]) -> ModelRequest:
        feedback = (
            "Your previous reply did not satisfy the required JSON Schema. Problems:\n- "
            + "\n- ".join(errors[:10])
            + "\nReply again with only a corrected JSON object that satisfies the schema."
        )
        messages = (
            *request.messages,
            Message(role=MessageRole.ASSISTANT, content=bad_text[:20_000]),
            Message(role=MessageRole.USER, content=feedback),
        )
        return request.model_copy(update={"messages": messages, "cacheable": False})

    def _cache_hit(
        self, key: str, request: ModelRequest, actor: AgentRole, task_id: str | None
    ) -> ModelResponse | None:
        assert self.cache is not None
        cached = self.cache.get(key)
        if cached is None:
            return None
        parsed: Any = None
        if request.response_schema is not None and not cached.tool_calls:
            parsed, errors = validate_structured(cached.text, request.response_schema)
            if errors:
                return None
        self._record(actor, task_id, cached.provider, cached.model_version, cached=True)
        return cached.model_copy(
            update={"parsed": parsed, "cached": True, "cost_usd": 0.0, "latency_ms": 0.0}
        )
