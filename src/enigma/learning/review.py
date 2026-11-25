"""Institutional review: KPIs from reflections and tasks -> improvement proposals (WS8).

Proposals are only PROPOSED here. Adoption goes through the IMPROVEMENT_PROPOSAL state machine
with benchmark evidence; protected targets always require human approval.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from enigma.core.hashing import sha256_hex
from enigma.schemas.common import AgentRole, UtcDatetime
from enigma.schemas.governance import (
    PROTECTED_TARGETS,
    ImprovementProposal,
    ProposalTarget,
    Reflection,
    ReflectionOutcome,
)
from enigma.schemas.knowledge import Lesson


def agent_kpis(reflections: Sequence[Reflection]) -> dict[str, dict[str, float]]:
    by_agent: dict[str, list[Reflection]] = defaultdict(list)
    for r in reflections:
        by_agent[r.agent.value].append(r)
    out: dict[str, dict[str, float]] = {}
    for agent, items in sorted(by_agent.items()):
        failures = sum(1 for r in items if r.outcome is not ReflectionOutcome.SUCCESS)
        walls = [r.metrics.get("wall_seconds", 0.0) for r in items]
        out[agent] = {
            "tasks": float(len(items)),
            "failure_rate": failures / len(items),
            "mean_wall_seconds": sum(walls) / len(walls),
            "http_requests": sum(r.metrics.get("http_requests", 0.0) for r in items),
            "usd": sum(r.metrics.get("usd", 0.0) for r in items),
        }
    return out


def _proposal(
    key: str,
    target: ProposalTarget,
    target_ref: str,
    summary: str,
    rationale: str,
    metric: str,
    baseline: float | None,
    evidence: Sequence[str],
    now: UtcDatetime,
) -> ImprovementProposal:
    return ImprovementProposal(
        id="imp_" + sha256_hex(key)[:26].upper(),
        target=target,
        target_ref=target_ref,
        change_summary=summary,
        rationale=rationale,
        success_metric=metric,
        baseline_metric_value=baseline,
        evidence_ids=tuple(evidence)[:50],
        proposed_by=AgentRole.META_LEARNING,
        requires_human_approval=target in PROTECTED_TARGETS,
        created_at=now,
    )


def review_institution(
    reflections: Sequence[Reflection],
    lessons: Sequence[Lesson],
    missing_credentials: Sequence[str],
    now: UtcDatetime,
    window_key: str,
) -> list[ImprovementProposal]:
    proposals: list[ImprovementProposal] = []
    for agent, kpi in agent_kpis(reflections).items():
        if kpi["tasks"] >= 3 and kpi["failure_rate"] > 0.3:
            evidence = [
                r.id
                for r in reflections
                if r.agent.value == agent and r.outcome is not ReflectionOutcome.SUCCESS
            ]
            failures = Counter(
                r.what_failed.split(":")[0]
                for r in reflections
                if r.agent.value == agent and r.what_failed
            )
            proposals.append(
                _proposal(
                    f"{window_key}|failure|{agent}",
                    ProposalTarget.WORKFLOW,
                    agent,
                    f"Reduce {agent} task failures (top cause: {failures.most_common(1)[0][0]})"
                    if failures
                    else f"Reduce {agent} task failures",
                    f"Failure rate {kpi['failure_rate']:.0%} over {int(kpi['tasks'])} tasks.",
                    f"{agent}.failure_rate",
                    kpi["failure_rate"],
                    evidence,
                    now,
                )
            )
    failure_topics = Counter(t for les in lessons if les.outcome == "failure" for t in les.topics)
    for topic, count in failure_topics.items():
        if topic in {"cost_fragility", "overfit", "no_signal", "multiple_testing"} and count >= 2:
            proposals.append(
                _proposal(
                    f"{window_key}|lesson|{topic}",
                    ProposalTarget.METHODOLOGY,
                    "quant_research",
                    f"Address recurring failure mode '{topic}' in hypothesis generation",
                    f"{count} cycles failed with '{topic}'. Adjust mechanism scoring or default "
                    "parameter grids so new hypotheses avoid this failure mode.",
                    f"share_of_cycles_failing_{topic}",
                    None,
                    [les.id for les in lessons if topic in les.topics],
                    now,
                )
            )
    if missing_credentials:
        proposals.append(
            _proposal(
                f"{window_key}|credentials|{','.join(sorted(missing_credentials))}",
                ProposalTarget.TOOL,
                "gateway",
                "Configure model/search provider credentials to enable LLM-assisted research",
                "Deterministic fallbacks are active; credentials would enable query expansion, "
                "structured extraction and model benchmarking: " + ", ".join(missing_credentials),
                "llm_assisted_tasks_share",
                0.0,
                (),
                now,
            )
        )
    return proposals
