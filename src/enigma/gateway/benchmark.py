"""Deterministic model benchmark for the Technology & Innovation agent.

Runs a fixed, versioned task set against every *available* model (credentials present or local
server reachable) through the real `ModelGateway` (budgets, ledger, retries all apply) and scores
outputs with deterministic checks only (schema validity, exact-match fields). With no available
provider it reports ``no_providers_available`` and produces no scores.

    uv run python -m enigma.gateway.benchmark --out bench.json --max-usd 0.50
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from enigma.core.errors import BudgetExceededError, ProviderUnavailableError
from enigma.core.hashing import sha256_of
from enigma.core.ids import new_id
from enigma.gateway.errors import StructuredOutputError
from enigma.gateway.gateway import ModelGateway, ProviderStatus
from enigma.gateway.types import (
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    RequestMetadata,
    ToolChoice,
    ToolSpec,
)
from enigma.schemas.common import AgentRole, StrictModel
from enigma.schemas.governance import Budget

BENCHMARK_VERSION = "1.0.0"
NO_PROVIDERS = "no_providers_available"


# --------------------------------------------------------------------------- task schemas


class ExtractedHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=10, description="One-sentence falsifiable hypothesis")
    asset: Literal["BTC", "ETH"]
    predictor: str = Field(description="The explanatory variable")
    direction: Literal["positive", "negative"]
    horizon_hours: int = Field(ge=1)
    falsification_test: str = Field(min_length=10)


class NumericAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: float


_ABSTRACT = (
    "We study BTCUSDT perpetual futures on Binance between January 2020 and December 2024. "
    "When the 8-hour funding rate exceeds 0.05%, BTC returns over the subsequent 24 hours are "
    "negative on average (-0.31%, t = -2.9, n = 412 events). The effect is not present for "
    "funding rates below 0.01%. After deducting 10 bps round-trip costs the effect remains "
    "marginally significant. We do not examine ETH."
)

_INSTRUMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["symbol", "venue", "kind", "max_leverage", "tags"],
    "properties": {
        "symbol": {"type": "string", "pattern": "^[A-Z]{3,10}USDT$"},
        "venue": {"type": "string", "enum": ["binance", "bybit", "okx"]},
        "kind": {"type": "string", "enum": ["spot", "perpetual"]},
        "max_leverage": {"type": "integer", "minimum": 1, "maximum": 5},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
    },
}

_TOOLS = (
    ToolSpec(
        name="get_ohlcv",
        description="Fetch OHLCV bars for a symbol.",
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["symbol", "interval", "start_date"],
            "properties": {
                "symbol": {"type": "string", "description": "Exchange symbol, e.g. BTCUSDT"},
                "interval": {"type": "string", "enum": ["1h", "4h", "1d"]},
                "start_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
            },
        },
    ),
    ToolSpec(
        name="get_funding_rate",
        description="Fetch the perpetual funding-rate history for a symbol.",
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["symbol"],
            "properties": {"symbol": {"type": "string"}},
        },
    ),
)


class TaskKind(StrEnum):
    EXTRACTION = "structured_extraction"
    SCHEMA = "schema_adherence"
    TOOL = "tool_call"
    ARITHMETIC = "arithmetic_reasoning"


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    kind: TaskKind
    system: str
    user: str
    response_schema: type[BaseModel] | dict[str, Any] | None
    expected: dict[str, Any]
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: ToolChoice = "auto"

    def request(self, model: str, max_output_tokens: int) -> ModelRequest:
        return ModelRequest(
            messages=(
                Message(role=MessageRole.SYSTEM, content=self.system),
                Message(role=MessageRole.USER, content=self.user),
            ),
            model=model,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            response_schema=self.response_schema,
            schema_name=f"bench_{self.id}" if self.response_schema is not None else None,
            tools=self.tools,
            tool_choice=self.tool_choice,
            cacheable=False,
            metadata=RequestMetadata(
                agent_role=AgentRole.TECHNOLOGY_INNOVATION, task_kind=f"benchmark.{self.kind}"
            ),
        )

    def fingerprint(self) -> dict[str, Any]:
        schema = self.response_schema
        return {
            "id": self.id,
            "kind": self.kind,
            "system": self.system,
            "user": self.user,
            "schema": schema.model_json_schema() if isinstance(schema, type) else schema,
            "expected": self.expected,
            "tools": [t.model_dump(mode="json") for t in self.tools],
            "tool_choice": self.tool_choice
            if isinstance(self.tool_choice, str)
            else self.tool_choice.model_dump(),
        }


TASKS: tuple[BenchmarkTask, ...] = (
    BenchmarkTask(
        id="hypothesis_extraction_v1",
        kind=TaskKind.EXTRACTION,
        system="You extract falsifiable trading hypotheses from research abstracts.",
        user=f"Extract the main hypothesis from this abstract.\n\nAbstract: {_ABSTRACT}",
        response_schema=ExtractedHypothesis,
        expected={
            "asset": "BTC",
            "direction": "negative",
            "horizon_hours": 24,
            "predictor_contains": "funding",
        },
    ),
    BenchmarkTask(
        id="schema_adherence_v1",
        kind=TaskKind.SCHEMA,
        system="You produce configuration objects that exactly follow the given schema.",
        user=(
            "Create the instrument config for the ETH/USDT perpetual on Binance with a maximum "
            "leverage of 3. Use the tags 'paper' and 'research' in that order."
        ),
        response_schema=_INSTRUMENT_SCHEMA,
        expected={
            "symbol": "ETHUSDT",
            "venue": "binance",
            "kind": "perpetual",
            "max_leverage": 3,
            "tags": ["paper", "research"],
        },
    ),
    BenchmarkTask(
        id="tool_call_v1",
        kind=TaskKind.TOOL,
        system="You are a data assistant. Use the provided tools to fetch data; do not guess.",
        user="Fetch daily OHLCV bars for ETHUSDT starting 2024-03-01.",
        response_schema=None,
        tools=_TOOLS,
        tool_choice="required",
        expected={
            "name": "get_ohlcv",
            "arguments": {"symbol": "ETHUSDT", "interval": "1d", "start_date": "2024-03-01"},
        },
    ),
    BenchmarkTask(
        id="arithmetic_allocation_v1",
        kind=TaskKind.ARITHMETIC,
        system="You solve arithmetic word problems. Report only the final number.",
        user=(
            "A portfolio of 1,250,000 USD allocates 12% to BTC and 8% to ETH. BTC then rises 30% "
            "and ETH falls 10%. What is the combined USD value of the BTC and ETH positions?"
        ),
        response_schema=NumericAnswer,
        expected={"answer": 285000.0},
    ),
    BenchmarkTask(
        id="arithmetic_compounding_v1",
        kind=TaskKind.ARITHMETIC,
        system="You solve arithmetic word problems. Report only the final number.",
        user=(
            "An account starts at 1000 USD and has three consecutive period returns of +10%, "
            "-10% and +5%. What is the final account value in USD?"
        ),
        response_schema=NumericAnswer,
        expected={"answer": 1039.5},
    ),
)

TASK_SET_HASH = sha256_of({"version": BENCHMARK_VERSION, "tasks": [t.fingerprint() for t in TASKS]})


# --------------------------------------------------------------------------- scoring


def _close(a: Any, b: float) -> bool:
    return isinstance(a, int | float) and abs(float(a) - b) <= 1e-6 * max(1.0, abs(b))


def score(task: BenchmarkTask, response: ModelResponse | None) -> dict[str, bool]:
    """Deterministic named checks. `None` response means the output never validated."""
    exp = task.expected
    if task.kind is TaskKind.TOOL:
        calls = response.tool_calls if response is not None else ()
        first = calls[0] if calls else None
        checks = {
            "single_tool_call": len(calls) == 1,
            "tool_name": first is not None and first.name == exp["name"],
            "arguments_valid": first is not None and first.arguments_valid,
        }
        for key, value in exp["arguments"].items():
            checks[f"arg_{key}"] = first is not None and first.arguments.get(key) == value
        return checks

    parsed = response.parsed if response is not None else None
    data: dict[str, Any] = parsed.model_dump() if isinstance(parsed, BaseModel) else (parsed or {})
    checks = {"schema_valid": parsed is not None}
    if task.kind is TaskKind.EXTRACTION:
        checks["asset"] = data.get("asset") == exp["asset"]
        checks["direction"] = data.get("direction") == exp["direction"]
        checks["horizon_hours"] = data.get("horizon_hours") == exp["horizon_hours"]
        checks["predictor"] = exp["predictor_contains"] in str(data.get("predictor", "")).lower()
    elif task.kind is TaskKind.SCHEMA:
        for key, value in exp.items():
            checks[key] = data.get(key) == value
    elif task.kind is TaskKind.ARITHMETIC:
        checks["answer"] = _close(data.get("answer"), exp["answer"])
    return checks


# --------------------------------------------------------------------------- results


class ResultStatus(StrEnum):
    SCORED = "scored"
    SKIPPED_BUDGET = "skipped_budget"
    PROVIDER_ERROR = "provider_error"


class BenchmarkResult(StrictModel):
    model_alias: str
    provider: str
    model_version: str | None
    task_id: str
    task_kind: str
    status: ResultStatus
    score: float | None = Field(default=None, ge=0, le=1)
    checks: dict[str, bool] = Field(default_factory=dict)
    latency_ms: float | None = None
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    repaired: bool = False
    error: str | None = None


class ModelSummary(StrictModel):
    model_alias: str
    provider: str
    tasks_scored: int
    mean_score: float | None
    total_cost_usd: float
    mean_latency_ms: float | None


class BenchmarkReport(StrictModel):
    benchmark_version: str
    task_set_hash: str
    registry_version: str
    started_at: datetime
    finished_at: datetime
    status: Literal["completed", "no_providers_available"]
    available_providers: tuple[str, ...]
    missing_credentials: dict[str, tuple[str, ...]]
    provider_status: tuple[ProviderStatus, ...] = ()
    results: tuple[BenchmarkResult, ...] = ()
    summary: tuple[ModelSummary, ...] = ()

    def to_table(self) -> str:
        if self.status == NO_PROVIDERS:
            lines = ["no providers available — no scores produced.", "Enable one of:"]
            lines += [
                f"  {p}: {' | '.join(v)}" for p, v in sorted(self.missing_credentials.items())
            ]
            lines += [
                f"  {st.provider}: {st.detail}"
                for st in self.provider_status
                if st.local and not st.missing_env_vars and not st.available
            ]
            return "\n".join(lines)
        rows = ["| model | provider | scored | mean score | cost USD | mean latency ms |"]
        rows.append("|---|---|---|---|---|---|")
        for s in self.summary:
            mean = "n/a" if s.mean_score is None else f"{s.mean_score:.3f}"
            lat = "n/a" if s.mean_latency_ms is None else f"{s.mean_latency_ms:.0f}"
            rows.append(
                f"| {s.model_alias} | {s.provider} | {s.tasks_scored} | {mean} | "
                f"{s.total_cost_usd:.6f} | {lat} |"
            )
        return "\n".join(rows)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _summarize(results: Sequence[BenchmarkResult]) -> tuple[ModelSummary, ...]:
    out: list[ModelSummary] = []
    for alias in dict.fromkeys(r.model_alias for r in results):
        rs = [r for r in results if r.model_alias == alias]
        scored = [r for r in rs if r.status is ResultStatus.SCORED and r.score is not None]
        lat = [r.latency_ms for r in scored if r.latency_ms is not None]
        out.append(
            ModelSummary(
                model_alias=alias,
                provider=rs[0].provider,
                tasks_scored=len(scored),
                mean_score=(sum(r.score or 0.0 for r in scored) / len(scored)) if scored else None,
                total_cost_usd=sum(r.cost_usd for r in rs),
                mean_latency_ms=(sum(lat) / len(lat)) if lat else None,
            )
        )
    return tuple(out)


class BenchmarkRunner:
    def __init__(
        self,
        gateway: ModelGateway,
        budget: Budget,
        *,
        tasks: Sequence[BenchmarkTask] = TASKS,
        max_output_tokens: int = 1024,
        actor: AgentRole = AgentRole.TECHNOLOGY_INNOVATION,
    ) -> None:
        self.gateway = gateway
        self.budget = budget
        self.tasks = tuple(tasks)
        self.max_output_tokens = max_output_tokens
        self.actor = actor

    def run(self, models: Sequence[str] | None = None) -> BenchmarkReport:
        started = self.gateway.clock.now()
        status = tuple(self.gateway.credential_report())
        available = tuple(st.provider for st in status if st.available)
        missing = {st.provider: st.missing_env_vars for st in status if st.missing_env_vars}
        if models is None:
            specs = self.gateway.available_models()
        else:
            specs = [self.gateway.registry.resolve_model(m) for m in models]
            specs = [s for s in specs if s.provider in available]
        if not specs:
            return BenchmarkReport(
                benchmark_version=BENCHMARK_VERSION,
                task_set_hash=TASK_SET_HASH,
                registry_version=self.gateway.registry.version,
                started_at=started,
                finished_at=self.gateway.clock.now(),
                status=NO_PROVIDERS,
                available_providers=available,
                missing_credentials=missing,
                provider_status=status,
            )
        task_id = new_id("bench")
        results: list[BenchmarkResult] = []
        for spec in specs:
            for task in self.tasks:
                results.append(self._run_one(spec.alias, spec.provider, task, task_id))
        return BenchmarkReport(
            benchmark_version=BENCHMARK_VERSION,
            task_set_hash=TASK_SET_HASH,
            registry_version=self.gateway.registry.version,
            started_at=started,
            finished_at=self.gateway.clock.now(),
            status="completed",
            available_providers=available,
            missing_credentials=missing,
            provider_status=status,
            results=tuple(results),
            summary=_summarize(results),
        )

    def _run_one(
        self, alias: str, provider: str, task: BenchmarkTask, task_id: str
    ) -> BenchmarkResult:
        base: dict[str, Any] = {
            "model_alias": alias,
            "provider": provider,
            "task_id": task.id,
            "task_kind": task.kind.value,
        }
        if task.tools and not self.gateway.registry.models[alias].capabilities.tools:
            return BenchmarkResult(
                **base,
                model_version=None,
                status=ResultStatus.PROVIDER_ERROR,
                error="model has no tool-calling capability",
            )
        request = task.request(alias, self.max_output_tokens)
        response: ModelResponse | None
        error: str | None = None
        try:
            response = self.gateway.complete(request, self.actor, task_id, self.budget)
        except StructuredOutputError as exc:
            response, error = None, str(exc)
        except BudgetExceededError as exc:
            return BenchmarkResult(
                **base, model_version=None, status=ResultStatus.SKIPPED_BUDGET, error=str(exc)
            )
        except ProviderUnavailableError as exc:
            return BenchmarkResult(
                **base, model_version=None, status=ResultStatus.PROVIDER_ERROR, error=str(exc)
            )
        checks = score(task, response)
        return BenchmarkResult(
            **base,
            model_version=response.model_version if response else None,
            status=ResultStatus.SCORED,
            score=sum(checks.values()) / len(checks),
            checks=checks,
            latency_ms=response.latency_ms if response else None,
            cost_usd=response.cost_usd if response else 0.0,
            input_tokens=response.usage.input_tokens if response else 0,
            output_tokens=response.usage.output_tokens if response else 0,
            repaired=response.repaired if response else False,
            error=error,
        )


# --------------------------------------------------------------------------- CLI


def main(
    argv: Sequence[str] | None = None,
    gateway_factory: Callable[[], ModelGateway] | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="enigma.gateway.benchmark", description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Write JSON report here")
    parser.add_argument("--models", nargs="*", default=None, help="Aliases (default: available)")
    parser.add_argument("--max-usd", type=float, default=0.0, help="Benchmark USD budget")
    parser.add_argument("--max-tokens", type=int, default=200_000, help="Benchmark token budget")
    args = parser.parse_args(argv)

    if gateway_factory is None:
        gateway_factory = _default_gateway
    gateway = gateway_factory()
    try:
        budget = Budget(max_usd=args.max_usd, max_tokens=args.max_tokens, max_http_requests=1000)
        report = BenchmarkRunner(gateway, budget).run(args.models)
    finally:
        gateway.close()
    sys.stdout.write(report.to_table() + "\n")
    if args.out is not None:
        report.write(args.out)
        sys.stdout.write(f"report written to {args.out}\n")
    return 2 if report.status == NO_PROVIDERS else 0


def _default_gateway() -> ModelGateway:
    from enigma.core.clock import SystemClock  # noqa: PLC0415
    from enigma.core.config import Settings  # noqa: PLC0415
    from enigma.gateway.registry import ModelRegistry  # noqa: PLC0415
    from enigma.store import Stores  # noqa: PLC0415

    settings = Settings.from_env()
    settings.ensure_dirs()
    stores = Stores.open(settings.db_path)
    registry = ModelRegistry.load(settings.registries_dir / "models")
    return ModelGateway(registry, stores.costs, None, SystemClock())


if __name__ == "__main__":
    raise SystemExit(main())
