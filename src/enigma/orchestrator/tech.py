"""Technology Innovation role: scheduled scans of external tooling/papers and model benchmarks."""

from __future__ import annotations

from typing import Any

from enigma.agents.base import Agent, AgentContext, DecisionRecord, TaskOutcome
from enigma.gateway.benchmark import ModelSummary, choose_model
from enigma.learning.adoption import may_roll_out
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
        choice = selection_from_benchmark(report)
        return TaskOutcome(
            outputs={
                "status": status,
                "summary": report.get("summary", []),
                **choice,
            },
            decisions=[
                DecisionRecord(
                    "model_benchmark",
                    status,
                    "scheduled_benchmark",
                    f"Benchmark status {status}; providers "
                    f"{report.get('available_providers', [])}; missing credentials for "
                    f"{sorted(report.get('missing_credentials', {}))}. "
                    f"Selected {choice['selected_model']}. "
                    "Adoption stays closed without a positive measured lift and a postmortem.",
                )
            ],
        )


def selection_from_benchmark(report: dict[str, Any]) -> dict[str, Any]:
    """Pick a measured model. A benchmark score is not an adoption lift."""
    summaries = [ModelSummary.model_validate(row) for row in report.get("summary") or []]
    chosen = choose_model(summaries)
    return {
        "selected_model": None if chosen is None else chosen.model_alias,
        "adoption_allowed": may_roll_out(
            0.0,
            postmortem_ids=(),
            benchmark_id=str(report.get("task_set_hash") or "") or None,
        ),
    }
