"""Institutional report: what the research time was spent on, and what was learned.

Counts of cycles are not the result. The report separates scientific failures from
technical ones and names mechanisms that were not in the seed library.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from enigma.research.mechanisms import MECHANISMS
from enigma.schemas.governance import ImprovementProposal, ProposalStatus
from enigma.schemas.inquiry import ScientificCase
from enigma.schemas.knowledge import MarketMechanism
from enigma.schemas.learning import Postmortem
from enigma.store import Stores

_METHOD = frozenset({"G01", "G02", "G16"})
_TECHNICAL = frozenset({"infrastructure", "model", "data", "implementation"})


def build_report(stores: Stores) -> dict[str, Any]:
    cycles = stores.checkpoints.list("research_cycle")
    productive = 0.0
    technical_failures = 0
    scientific_failures = 0
    method_failures = 0
    passes = 0
    designs: Counter[str] = Counter()
    frequencies: Counter[str] = Counter()
    for cp in cycles:
        state = cp["state"]
        productive += _hours(state.get("started_at"), cp.get("updated_at"))
        overall = state.get("validation_overall")
        failure_class = str(state.get("failure_class") or "")
        if overall == "PASS":
            passes += 1
        elif failure_class in _TECHNICAL or cp["status"] == "aborted":
            technical_failures += 1
        elif overall == "FAIL":
            failed = set(state.get("failed_gates") or [])
            if failed and failed <= _METHOD:
                method_failures += 1
            else:
                scientific_failures += 1
        for topic in _topics(stores, state):
            if topic in {"1h", "8h", "1d"}:
                frequencies[topic] += 1
            elif topic not in {"discovered", "continuous-backlog"}:
                designs[topic] += 1
    seed_names = {item.name for item in MECHANISMS.values()}
    mechanisms = stores.documents.list(MarketMechanism, limit=500)
    novel = [item.name for item in mechanisms if item.name not in seed_names]
    cases = stores.documents.list(ScientificCase, limit=500)
    proposals = stores.documents.list(ImprovementProposal, limit=500)
    adopted = [item for item in proposals if item.status is ProposalStatus.ROLLED_OUT]
    rejected = [item for item in proposals if item.status is ProposalStatus.REJECTED]
    costs = stores.costs.totals()
    return {
        "productive_research_hours": round(productive, 3),
        "passes": passes,
        "scientific_failures": scientific_failures,
        "method_failures": method_failures,
        "technical_failures": technical_failures,
        "inference": {
            "usd": costs.get("usd", 0.0),
            "tokens": costs.get("tokens", 0),
            "http_requests": costs.get("http_requests", 0),
        },
        "idle_note": (
            "Idle wall-clock is not inferred. A tick that opens no cycle is idle. "
            "Backlog exhaustion is the count of seed directions that are not viable."
        ),
        "backlog_exhausted_seed_directions": _exhausted(stores),
        "novel_mechanisms": novel,
        "scientific_cases": len(cases),
        "design_counts": dict(designs),
        "frequency_counts": dict(frequencies),
        "improvements_adopted": len(adopted),
        "improvements_rejected": len(rejected),
        "postmortems": len(stores.documents.list(Postmortem, limit=500)),
        "diversity": {
            "mechanisms": len({item.name for item in mechanisms}),
            "designs": len(designs),
            "frequencies": len(frequencies),
        },
    }


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"Productive research hours (cycle spans): {report['productive_research_hours']}",
        f"Passes: {report['passes']}",
        f"Scientific failures: {report['scientific_failures']}",
        f"Method failures (not evidence against a mechanism): {report['method_failures']}",
        f"Technical failures: {report['technical_failures']}",
        f"Inference USD: {report['inference']['usd']}",
        f"Novel mechanisms: {', '.join(report['novel_mechanisms']) or 'none'}",
        f"Scientific cases: {report['scientific_cases']}",
        f"Improvements adopted: {report['improvements_adopted']}",
        f"Improvements rejected: {report['improvements_rejected']}",
        f"Designs: {report['design_counts']}",
        f"Frequencies: {report['frequency_counts']}",
        str(report["idle_note"]),
    ]
    return "\n".join(lines)


def _hours(started: object, updated: object) -> float:
    if not isinstance(started, str) or not isinstance(updated, str):
        return 0.0
    try:
        begin = datetime.fromisoformat(started)
        end = datetime.fromisoformat(updated)
    except ValueError:
        return 0.0
    return max((end - begin).total_seconds(), 0.0) / 3600.0


def _topics(stores: Stores, state: dict[str, Any]) -> tuple[str, ...]:
    objective_id = state.get("objective_id")
    if not isinstance(objective_id, str):
        return ()
    from enigma.schemas.research import ResearchObjective  # noqa: PLC0415

    objective = stores.documents.get(ResearchObjective, objective_id)
    if objective is None:
        return ()
    return objective.topics


def _exhausted(stores: Stores) -> int:
    from enigma.orchestrator.backlog import research_agenda  # noqa: PLC0415

    agenda = research_agenda(stores, inputs_budget=24)
    return sum(1 for item in agenda.ranked if not item.viable)
