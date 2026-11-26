"""Operational service wiring: real data catalogs, sandbox + engine, validation, model gateway.

Each plane gets its own objects. The research catalog writes under ENIGMA_RESEARCH_DIR and is
capped at the research cutoff; the validation plane acquires its own copy of the data under
ENIGMA_VALIDATION_DIR, which only the validation service reads.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from pydantic import BaseModel

from enigma.agents.budget import BudgetTracker
from enigma.backtest import BarBacktestEngine
from enigma.core.clock import Clock
from enigma.core.config import Settings
from enigma.core.errors import PermissionDeniedError
from enigma.core.hashing import sha256_of
from enigma.data import DataCatalog
from enigma.fsm import StateMachineStore
from enigma.gateway import ModelGateway, ModelRegistry, StructuredLLM
from enigma.gateway.benchmark import BenchmarkRunner
from enigma.intelligence import (
    HttpBudget,
    HttpClient,
    RawStore,
    ResearchEngine,
    ResearchQuery,
    default_providers,
    persist_result,
)
from enigma.knowledge import KnowledgeBase
from enigma.orchestrator.bindings import Services
from enigma.orchestrator.cycle import HoldoutPolicy
from enigma.schemas.common import AgentRole, Plane
from enigma.schemas.governance import Budget
from enigma.schemas.knowledge import EvidenceItem
from enigma.schemas.research import CostModel, DatasetVersion, StrategyPackage
from enigma.schemas.validation import RedactedValidationSummary, ValidationRequest
from enigma.store import Stores
from enigma.store.documents import PlaneView
from enigma.strategy import SandboxJob, SandboxRunner, build_package, symbol_arrays
from enigma.strategy.interface import INTERFACE_VERSION
from enigma.strategy.package import package_content_hash
from enigma.validation import ValidationService, summary_id_for

BACKTEST_NOTIONAL_USD = 1_000_000.0
SEARCH_CREDENTIALS = {
    "semantic_scholar": ("SEMANTIC_SCHOLAR_API_KEY",),
    "brave_search": ("BRAVE_SEARCH_API_KEY",),
    "tavily": ("TAVILY_API_KEY",),
    "serpapi": ("SERPAPI_API_KEY",),
    "github": ("GITHUB_TOKEN",),
}


def load_cost_model(registries: Path, version: str) -> CostModel:
    path = registries / "costs" / f"{version}.yaml"
    if not path.exists():
        raise KeyError(f"cost model {version} is not registered")
    return CostModel.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class CatalogHoldoutSource:
    """HoldoutSource backed by a validation-plane DataCatalog (independent acquisition)."""

    def __init__(self, catalog: DataCatalog, data_start: datetime, clock: Clock) -> None:
        self.catalog = catalog
        self.data_start = data_start
        self.clock = clock
        self._versions: dict[tuple[tuple[str, ...], datetime, datetime], DatasetVersion] = {}

    def available_range(self, symbols: Sequence[str]) -> tuple[datetime, datetime]:
        today = self.clock.now().astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        return self.data_start, today - timedelta(days=1)

    def _version(self, symbols: Sequence[str], start: datetime, end: datetime) -> DatasetVersion:
        key = (tuple(sorted(symbols)), start, end)
        if key not in self._versions:
            self._versions[key] = self.catalog.acquire_perp_funding_panel(
                list(symbols), start, end, "8h"
            )
        return self._versions[key]

    def load_holdout(self, symbols: Sequence[str], start: datetime, end: datetime) -> pl.DataFrame:
        return self.catalog.load(self._version(symbols, start, end))

    def dataset_ids(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> tuple[str, ...]:
        return (self._version(symbols, start, end).id,)


class OperationalServices:
    def __init__(self, settings: Settings, stores: Stores, policy: HoldoutPolicy, clock: Clock):
        self.settings = settings
        self.stores = stores
        self.policy = policy
        self.clock = clock
        self.fsm = StateMachineStore(stores.db, stores.audit)
        self.research_catalog = DataCatalog(
            settings.research_dir / "data",
            stores.documents,
            Plane.RESEARCH,
            AgentRole.DATA_ENGINEERING,
            okx_cross_check=True,
            costs=stores.costs,
            clock=clock,
        )
        validation_catalog = DataCatalog(
            settings.validation_dir / "data",
            None,
            Plane.VALIDATION,
            AgentRole.INDEPENDENT_VALIDATION,
            costs=stores.costs,
            clock=clock,
        )
        self.holdout_source = CatalogHoldoutSource(validation_catalog, policy.research_start, clock)
        self.sandbox = SandboxRunner(audit=stores.audit, actor="strategy_sandbox")
        self.engine = BarBacktestEngine()
        self._validation: ValidationService | None = None
        self._cutoffs: dict[tuple[str, ...], datetime] = {}
        self.gateway = ModelGateway(
            ModelRegistry.load(settings.registries_dir / "models"),
            stores.costs,
            settings.home / "cache" / "model_cache.sqlite3",
            clock,
        )

    # ---------------------------------------------------------------- validation plane
    @property
    def validation(self) -> ValidationService:
        if self._validation is None:
            view = PlaneView(
                self.stores.documents,
                readable=frozenset({Plane.CONTROL, Plane.RESEARCH, Plane.VALIDATION}),
                writable=frozenset({Plane.CONTROL, Plane.VALIDATION}),
                actor="validation_service",
            )
            self._validation = ValidationService(
                self.settings.validation_dir,
                view,
                self.holdout_source,
                self.stores.audit,
                self.fsm,
                clock=self.clock,
            )
        return self._validation

    def research_cutoff(self, symbols: Sequence[str]) -> datetime:
        key = tuple(sorted(symbols))
        if key not in self._cutoffs:
            self._cutoffs[key] = self.validation.seal_holdout(key)
        return self._cutoffs[key]

    def validation_run(self, request_id: str) -> dict[str, Any]:
        request = self.stores.documents.require(ValidationRequest, request_id)
        report = self.validation.validate(request)
        summary = self.stores.documents.require(
            RedactedValidationSummary, summary_id_for(request.id)
        )
        return {
            "summary_id": summary.id,
            "report_id": report.id,
            "overall": summary.overall.value,
            "protocol_version": summary.protocol_version,
            "failed_gates": list(summary.failed_gate_names),
        }

    # ---------------------------------------------------------------- research plane
    def data_acquire_research(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        frequency: str,
        budget: BudgetTracker | None = None,
    ) -> dict[str, Any]:
        cutoff = self.research_cutoff(symbols)
        if end > cutoff:
            raise PermissionDeniedError(
                f"research acquisition must end at or before the sealed cutoff {cutoff.isoformat()}"
            )
        ds = self.research_catalog.acquire_perp_funding_panel(symbols, start, end, frequency)
        return {
            "dataset_version_id": ds.id,
            "content_hash": ds.content_hash,
            "pit_certified": ds.pit_certified,
            "rows": ds.row_count,
            "failed_checks": [c.name for c in ds.quality.checks if not c.passed],
        }

    def data_load_research(self, dataset_version_id: str) -> pl.DataFrame:
        ds = self.research_catalog.find(dataset_version_id)
        if ds.plane != Plane.RESEARCH.value:
            raise PermissionDeniedError(f"{dataset_version_id} is not a research-plane dataset")
        return self.research_catalog.load(ds)

    def strategy_build(
        self,
        source_code: str,
        entrypoint: str,
        params: dict[str, Any],
        hypothesis_id: str,
        name: str,
        required_columns: Sequence[str],
        cost_model_version: str,
    ) -> StrategyPackage:
        cost = load_cost_model(self.settings.registries_dir, cost_model_version)
        columns = tuple(required_columns)
        digest = package_content_hash(
            source_code, entrypoint, params, INTERFACE_VERSION, cost, columns, 1
        )
        return build_package(
            source_code,
            entrypoint,
            params,
            cost,
            hypothesis_id,
            name=name,
            required_columns=columns,
            created_by=AgentRole.STRATEGY_ENGINEERING,
            created_at=self.clock.now(),
            package_id="pkg_"
            + sha256_of([digest, hypothesis_id]).removeprefix("sha256:")[:26].upper(),
        )

    def backtest_run(self, package_id: str, dataset_version_id: str, seed: int) -> dict[str, Any]:
        package = self.stores.documents.require(StrategyPackage, package_id)
        frame = self.data_load_research(dataset_version_id)
        symbols = sorted(frame.get_column("symbol").unique().to_list())
        datasets = {s: symbol_arrays(frame, s, package.required_columns) for s in symbols}
        jobs = [SandboxJob(job_id=s, dataset=s) for s in symbols]
        run = self.sandbox.run(package, datasets, jobs).require_ok()
        result = self.engine.run(
            run.positions, frame, package.cost_model, seed, notional_usd=BACKTEST_NOTIONAL_USD
        )
        out_dir = self.settings.research_dir / "backtests"
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact = out_dir / f"{result.result_hash.removeprefix('sha256:')[:32]}.npz"
        if not artifact.exists():
            arrays: dict[str, Any] = {"ts": result.ts, "equity": result.equity, **result.series}
            np.savez_compressed(artifact, **arrays)
        return {
            "metrics": {k: float(v) for k, v in result.metrics.items()},
            "resources": {"sandbox_cpu_seconds": float(run.cpu_seconds)},
            "result_hash": result.result_hash,
            "artifact_uri": str(artifact),
            "isolation": run.isolation,
        }

    def intel_research(
        self,
        topic: str,
        keywords: Sequence[str],
        max_sources: int,
        depth: int,
        budget: BudgetTracker,
    ) -> dict[str, Any]:
        view = PlaneView(
            self.stores.documents,
            readable=frozenset({Plane.RESEARCH, Plane.CONTROL}),
            writable=frozenset({Plane.RESEARCH}),
            actor=AgentRole.INTELLIGENCE.value,
        )
        kb = KnowledgeBase(view, self.clock)
        client = HttpClient(
            HttpBudget(budget.budget.max_http_requests),
            RawStore(self.settings.research_dir / "raw" / "intelligence"),
            user_agent=self.settings.http_user_agent,
        )
        query = ResearchQuery(
            topic=topic,
            keywords=tuple(keywords),
            max_sources=max_sources,
            max_depth=depth,
            full_text_fetches=2,
            budget=budget.budget,
        )
        try:
            result = ResearchEngine(
                default_providers(client, self.clock), client, self.clock, knowledge=kb
            ).run(query)
        finally:
            budget.charge("http", "intelligence", http_requests=client.budget.used)
        report = persist_result(result, kb)
        source_ids = list(dict.fromkeys(report.source_ids[s.id] for s in result.sources))
        evidence_ids = [
            e.id
            for sid in source_ids
            for e in self.stores.documents.list(EvidenceItem, where={"source_id": sid}, limit=100)
        ]
        return {
            "source_ids": source_ids,
            "evidence_ids": evidence_ids,
            "providers": {
                name: stats.sources_kept for name, stats in result.provider_stats.items()
            },
            "provider_errors": {
                name: list(stats.errors)
                for name, stats in result.provider_stats.items()
                if stats.errors
            },
            "queries": list(result.queries_issued),
            "duplicates": report.duplicates + report.sources_merged,
            "http_requests": result.http_requests_used,
            "budget_exhausted": result.budget_exhausted,
        }

    # ---------------------------------------------------------------- model gateway
    def gateway_status(self) -> dict[str, Any]:
        missing = [
            f"{provider}: {' | '.join(names)}"
            for provider, names in sorted(self.gateway.missing_credentials().items())
        ]
        missing += [
            f"{provider}: {' | '.join(names)}"
            for provider, names in SEARCH_CREDENTIALS.items()
            if not any(os.environ.get(n) for n in names)
        ]
        return {
            "available_providers": self.gateway.available_providers(),
            "missing_credentials": missing,
            "daily_cap_usd": self.gateway.daily_cap_usd(),
        }

    def llm_structured(
        self,
        model_cls: type[BaseModel],
        system: str,
        user: str,
        task_id: str,
        budget: BudgetTracker,
        actor: str,
    ) -> BaseModel:
        return StructuredLLM(self.gateway).generate(
            model_cls, system, user, AgentRole(actor), task_id, budget.budget
        )

    def gateway_benchmark(self) -> dict[str, Any]:
        budget = Budget(max_usd=0.25, max_tokens=100_000, max_wall_seconds=600)
        report = BenchmarkRunner(self.gateway, budget).run()
        return report.model_dump(mode="json")

    def services(self) -> Services:
        return Services(
            intel_research=self.intel_research,
            data_acquire_research=self.data_acquire_research,
            data_load_research=self.data_load_research,
            strategy_build=self.strategy_build,
            backtest_run=self.backtest_run,
            validation_run=self.validation_run,
            gateway_status=self.gateway_status,
            llm_structured=self.llm_structured,
            gateway_benchmark=self.gateway_benchmark,
            research_cutoff=self.research_cutoff,
        )


def default_services(
    settings: Settings, stores: Stores, policy: HoldoutPolicy, clock: Clock
) -> Services:
    return OperationalServices(settings, stores, policy, clock).services()


def credential_report() -> dict[str, Any]:
    """Optional credentials. Enigma runs without any of them; values are never printed."""
    registry = ModelRegistry.load(Settings.from_env().registries_dir / "models")
    models = {
        name: {
            "env": list(spec.api_key_env or ((spec.base_url_env,) if spec.base_url_env else ())),
            "configured": any(
                os.environ.get(v) for v in (*spec.api_key_env, spec.base_url_env or "") if v
            ),
        }
        for name, spec in registry.providers.items()
    }
    search = {
        name: {"env": list(env), "configured": any(os.environ.get(v) for v in env)}
        for name, env in SEARCH_CREDENTIALS.items()
    }
    return {
        "model_providers": models,
        "search_providers": search,
        "spend_cap": {
            "env": "ENIGMA_MAX_DAILY_API_SPEND_USD",
            "value": os.environ.get("ENIGMA_MAX_DAILY_API_SPEND_USD", "0 (no paid model calls)"),
        },
    }
