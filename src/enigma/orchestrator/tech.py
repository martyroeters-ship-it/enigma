"""Technology Innovation role: scheduled scans of external tooling/papers and model benchmarks."""

from __future__ import annotations

from typing import Any

from enigma.agents.base import Agent, AgentContext, DecisionRecord, TaskOutcome
from enigma.schemas.common import AgentRole
from enigma.schemas.knowledge import Source, SourceKind

WATCH_KINDS = (SourceKind.CODE_REPOSITORY, SourceKind.PREPRINT, SourceKind.ACADEMIC_PAPER)


class TechnologyInnovationAgent(Agent):
    role = AgentRole.TECHNOLOGY_INNOVATION

    def execute(self, ctx: AgentContext) -> TaskOutcome:
        if ctx.task.kind == "tech.benchmark_models":
            return self._benchmark(ctx)
        topics = list(ctx.task.inputs["topics"])
        found: list[str] = []
        providers: dict[str, int] = {}
        for topic in topics:
            result: dict[str, Any] = ctx.tools.call(
                "intel.research",
                topic=topic,
                keywords=[],
                max_sources=int(ctx.task.inputs.get("max_sources", 10)),
                depth=0,
                budget=ctx.budget,
            )
            found.extend(result["source_ids"])
            for name, n in result.get("providers", {}).items():
                providers[name] = providers.get(name, 0) + int(n)
        sources = [s for s in (ctx.view.get(Source, x) for x in dict.fromkeys(found)) if s]
        watch = [s for s in sources if s.kind in WATCH_KINDS]
        return TaskOutcome(
            outputs={
                "source_ids": [s.id for s in sources],
                "watchlist": [{"source_id": s.id, "title": s.title, "url": s.url} for s in watch],
                "providers": providers,
            },
            decisions=[
                DecisionRecord(
                    "technology_scan",
                    "watchlist_updated" if watch else "nothing_new",
                    "scheduled_scan",
                    f"Scanned {len(topics)} topics across {sorted(providers)}; "
                    f"{len(sources)} sources, {len(watch)} code/paper items placed on the "
                    "watchlist. Adoption requires a spike, benchmark and ADR (human approval).",
                    output_refs=tuple(s.id for s in watch[:50]),
                )
            ],
            metrics={"sources": float(len(sources)), "watchlist": float(len(watch))},
            what_worked=f"{len(sources)} sources retrieved" if sources else "",
            what_failed="" if sources else "no sources retrieved",
        )

    def _benchmark(self, ctx: AgentContext) -> TaskOutcome:
        report: dict[str, Any] = ctx.tools.call("gateway.benchmark")
        status = str(report.get("status", "unknown"))
        return TaskOutcome(
            outputs={"status": status, "summary": report.get("summary", [])},
            decisions=[
                DecisionRecord(
                    "model_benchmark",
                    status,
                    "scheduled_benchmark",
                    f"Benchmark status {status}; providers "
                    f"{report.get('available_providers', [])}; missing credentials for "
                    f"{sorted(report.get('missing_credentials', {}))}.",
                )
            ],
        )
