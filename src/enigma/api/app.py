"""Operational dashboard (read-only) + JSON API. Shows only recorded institutional activity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from enigma.api.readmodels import ReadModels
from enigma.core.config import Settings
from enigma.fsm import RESEARCH_CYCLE_STEPS
from enigma.store import Stores

TEMPLATES = Path(__file__).parent / "templates"


def create_app(settings: Settings | None = None, stores: Stores | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    stores = stores or Stores.open(settings.db_path)
    rm = ReadModels(stores)
    app = FastAPI(title="Enigma", docs_url="/api/docs", redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATES))
    templates.env.filters["tojson_pretty"] = lambda v: json.dumps(v, indent=2, default=str)

    def page(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        return templates.TemplateResponse(request, name, {"health": rm.health(), **ctx})

    @app.get("/", response_class=HTMLResponse)
    def overview(request: Request) -> HTMLResponse:
        return page(
            request,
            "overview.html",
            cycles=rm.cycles(),
            knowledge=rm.knowledge(),
            machines=rm.machines(),
            costs=rm.costs(),
            decisions=rm.decisions(15),
            operations=rm.operations(),
        )

    @app.get("/cycles/{cycle_id}", response_class=HTMLResponse)
    def cycle(request: Request, cycle_id: str) -> HTMLResponse:
        try:
            detail = rm.cycle_detail(cycle_id)
        except KeyError as exc:
            raise HTTPException(404, "cycle not found") from exc
        return page(request, "cycle.html", c=detail, steps=RESEARCH_CYCLE_STEPS)

    @app.get("/agents", response_class=HTMLResponse)
    def agents(request: Request) -> HTMLResponse:
        return page(request, "agents.html", agents=rm.agents(), decisions=rm.decisions(60))

    @app.get("/knowledge", response_class=HTMLResponse)
    def knowledge(request: Request) -> HTMLResponse:
        return page(request, "knowledge.html", k=rm.knowledge())

    @app.get("/strategies", response_class=HTMLResponse)
    def strategies(request: Request) -> HTMLResponse:
        return page(request, "strategies.html", s=rm.strategies())

    @app.get("/execution", response_class=HTMLResponse)
    def execution(request: Request) -> HTMLResponse:
        return page(request, "execution.html", e=rm.execution())

    @app.get("/learning", response_class=HTMLResponse)
    def learning(request: Request) -> HTMLResponse:
        return page(request, "learning.html", lr=rm.learning())

    @app.get("/messages", response_class=HTMLResponse)
    def messages(request: Request) -> HTMLResponse:
        return page(request, "messages.html", messages=rm.messages())

    @app.get("/models", response_class=HTMLResponse)
    def models(request: Request) -> HTMLResponse:
        return page(request, "models.html", m=rm.model_usage())

    @app.get("/audit", response_class=HTMLResponse)
    def audit(request: Request) -> HTMLResponse:
        return page(request, "audit.html", entries=rm.audit(), costs=rm.costs())

    @app.get("/api/operations")
    def api_operations() -> dict[str, Any]:
        return rm.operations()

    @app.get("/api/health")
    def api_health() -> dict[str, Any]:
        return rm.health()

    @app.get("/api/cycles")
    def api_cycles() -> list[dict[str, Any]]:
        return rm.cycles()

    @app.get("/api/cycles/{cycle_id}")
    def api_cycle(cycle_id: str) -> dict[str, Any]:
        try:
            return rm.cycle_detail(cycle_id)
        except KeyError as exc:
            raise HTTPException(404, "cycle not found") from exc

    @app.get("/api/agents")
    def api_agents() -> list[dict[str, Any]]:
        return rm.agents()

    @app.get("/api/knowledge")
    def api_knowledge() -> dict[str, Any]:
        return rm.knowledge()

    @app.get("/api/strategies")
    def api_strategies() -> dict[str, Any]:
        return rm.strategies()

    @app.get("/api/execution")
    def api_execution() -> dict[str, Any]:
        return rm.execution()

    @app.get("/api/learning")
    def api_learning() -> dict[str, Any]:
        return rm.learning()

    @app.get("/api/messages")
    def api_messages() -> list[dict[str, Any]]:
        return rm.messages()

    @app.get("/api/costs")
    def api_costs() -> dict[str, Any]:
        return rm.costs()

    @app.get("/api/models")
    def api_models() -> dict[str, Any]:
        return rm.model_usage()

    @app.get("/api/audit")
    def api_audit() -> list[dict[str, Any]]:
        return rm.audit()

    return app
