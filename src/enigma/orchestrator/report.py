"""Human-readable account of what the institution actually did."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from enigma.orchestrator.backlog import research_agenda
from enigma.schemas.governance import ImprovementProposal, ProposalStatus
from enigma.schemas.knowledge import Lesson
from enigma.schemas.research import (
    DiscoveredMechanism,
    Experiment,
    ExperimentStatus,
    Hypothesis,
    HypothesisStatus,
)
from enigma.schemas.validation import RedactedValidationSummary
from enigma.store import Stores


def build_report(stores: Stores, now: datetime | None = None) -> dict[str, Any]:
    """Read-only snapshot. It does not open a cycle or adopt an improvement."""
    moment = now or datetime.now(UTC)
    experiments = stores.documents.list(Experiment, limit=2000)
    completed = [item for item in experiments if item.status is ExperimentStatus.SUCCEEDED]
    hypotheses = stores.documents.list(Hypothesis, limit=2000)
    rejected = [
        item
        for item in hypotheses
        if item.status
        in {
            HypothesisStatus.REJECTED_IN_RESEARCH,
            HypothesisStatus.INVALIDATED,
        }
    ]
    summaries = stores.documents.list(RedactedValidationSummary, limit=500)
    discovered = stores.documents.list(DiscoveredMechanism, limit=200)
    lessons = stores.documents.list(Lesson, limit=200)
    improvements = stores.documents.list(ImprovementProposal, limit=500)
    adopted = [
        item
        for item in improvements
        if item.status in {ProposalStatus.APPROVED, ProposalStatus.ROLLED_OUT}
    ]
    ranked = research_agenda(stores, inputs_budget=1.0).ranked
    viable = [item for item in ranked if item.viable]
    cycles = stores.checkpoints.list("research_cycle")
    finished = [row for row in cycles if row["status"] in {"completed", "aborted"}]
    active = [row for row in cycles if row["status"] == "running"]
    costs = stores.costs.totals()
    inference = stores.db.query_one(
        "SELECT created_at, provider, model, usd FROM costs WHERE category = ? "
        "ORDER BY created_at DESC LIMIT 1",
        ("llm",),
    )
    last_cycle = cycles[0]["updated_at"] if cycles else None
    idle_hours = None
    if not active and isinstance(last_cycle, str):
        idle_hours = (moment - datetime.fromisoformat(last_cycle)).total_seconds() / 3600
    return {
        "as_of": moment.isoformat(),
        "completed_experiments": len(completed),
        "best_experiments": _best(completed),
        "rejected_hypotheses": [
            {"id": item.id, "title": item.title, "status": item.status.value}
            for item in rejected[:20]
        ],
        "validation": [
            {
                "id": item.id,
                "overall": item.overall.value,
                "failed_gates": list(item.failed_gate_names),
                "strategy_package_id": item.strategy_package_id,
            }
            for item in summaries[:20]
        ],
        "discovered_mechanisms": [
            {
                "id": item.id,
                "name": item.name,
                "origin": item.origin,
                "horizon": item.horizon,
                "features": list(item.features),
                "objective_id": item.objective_id,
            }
            for item in discovered
        ],
        "lessons": [
            {"id": item.id, "outcome": item.outcome, "statement": item.statement}
            for item in lessons[:20]
        ],
        "improvements_proposed": len(improvements),
        "improvements_adopted": [
            {
                "id": item.id,
                "status": item.status.value,
                "summary": item.change_summary,
                "lift": item.measured_lift,
            }
            for item in adopted
        ],
        "cycles_recorded": len(cycles),
        "cycles_finished": len(finished),
        "cycles_running": len(active),
        "last_cycle_at": last_cycle,
        "idle": not active,
        "idle_hours": idle_hours,
        "inference": None
        if inference is None
        else {
            "at": inference["created_at"],
            "provider": inference["provider"],
            "model": inference["model"],
            "usd": inference["usd"],
        },
        "costs": costs,
        "next_priorities": [
            {"id": item.candidate.id, "summary": item.candidate.summary, "score": item.score}
            for item in viable[:8]
        ],
        "backlog_viable": len(viable),
        "backlog_nonviable": len(ranked) - len(viable),
    }


def render_report(stores: Stores, now: datetime | None = None) -> str:
    body = build_report(stores, now)
    lines = [
        f"Enigma research report as of {body['as_of']}",
        f"Cycles recorded {body['cycles_recorded']}, finished {body['cycles_finished']}, "
        f"running {body['cycles_running']}.",
        f"Completed experiments {body['completed_experiments']}.",
        f"Improvements proposed {body['improvements_proposed']}, "
        f"adopted {len(body['improvements_adopted'])}.",
        f"Viable backlog items {body['backlog_viable']}; nonviable {body['backlog_nonviable']}.",
        f"Last cycle checkpoint {body['last_cycle_at'] or 'none'}. "
        f"Idle hours {body['idle_hours'] if body['idle_hours'] is not None else 'n/a'}.",
    ]
    inference = body["inference"]
    if inference is None:
        lines.append("No model inference cost has been recorded.")
    else:
        lines.append(
            f"Last inference {inference['at']} via {inference['provider']} "
            f"{inference['model'] or ''} at ${float(inference['usd']):.4f}."
        )
    costs = body["costs"]
    lines.append(
        f"Costs: ${float(costs['usd']):.4f}, {int(costs['tokens'])} tokens, "
        f"{int(costs['http_requests'])} HTTP requests."
    )
    lines.append("Best completed experiments:")
    lines.extend(
        f"  {item['id']} net Sharpe {item['net_sharpe']} gross Sharpe {item['gross_sharpe']}"
        for item in body["best_experiments"]
    )
    if not body["best_experiments"]:
        lines.append("  none")
    lines.append("Validation results:")
    lines.extend(
        f"  {item['id']} {item['overall']} failed {', '.join(item['failed_gates']) or 'none'}"
        for item in body["validation"]
    )
    if not body["validation"]:
        lines.append("  none")
    lines.append("Discovered mechanisms:")
    lines.extend(
        f"  {item['id']} {item['name']} ({item['origin']}, {item['horizon']}, "
        f"{', '.join(item['features'])})"
        for item in body["discovered_mechanisms"]
    )
    if not body["discovered_mechanisms"]:
        lines.append("  none")
    lines.append("Rejected hypotheses:")
    lines.extend(
        f"  {item['id']} {item['title']} ({item['status']})" for item in body["rejected_hypotheses"]
    )
    if not body["rejected_hypotheses"]:
        lines.append("  none")
    lines.append("Recent lessons:")
    lines.extend(f"  {item['outcome']}: {item['statement']}" for item in body["lessons"])
    if not body["lessons"]:
        lines.append("  none")
    lines.append("Adopted improvements:")
    lines.extend(
        f"  {item['id']} lift {item['lift']} {item['summary']}"
        for item in body["improvements_adopted"]
    )
    if not body["improvements_adopted"]:
        lines.append("  none")
    lines.append("Next research priorities:")
    lines.extend(f"  {item['id']}: {item['summary']}" for item in body["next_priorities"])
    if not body["next_priorities"]:
        lines.append("  none. Discovery runs only when a verbatim claim justifies a new question.")
    return "\n".join(lines)


def _best(experiments: list[Experiment]) -> list[dict[str, Any]]:
    ranked = sorted(
        experiments,
        key=lambda item: item.metrics.get("net_sharpe", float("-inf")),
        reverse=True,
    )
    return [
        {
            "id": item.id,
            "net_sharpe": item.metrics.get("net_sharpe"),
            "gross_sharpe": item.metrics.get("gross_sharpe"),
            "hypothesis_id": item.hypothesis_id,
        }
        for item in ranked[:5]
    ]
