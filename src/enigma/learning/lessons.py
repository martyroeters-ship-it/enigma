"""Lesson derivation and next-experiment proposals (M1 steps 10-12).

Rules are deterministic and cite their inputs. Every cycle produces at least one lesson, whether
the strategy passed or failed (ADR-007).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from enigma.core.hashing import sha256_hex
from enigma.research.mechanisms import MECHANISMS
from enigma.schemas.common import AgentRole, ClaimLevel, UtcDatetime
from enigma.schemas.cycle import NextExperimentProposal
from enigma.schemas.knowledge import Lesson
from enigma.schemas.research import Experiment, ExperimentStatus, Hypothesis
from enigma.schemas.validation import GateOutcome, RedactedValidationSummary


@dataclass(frozen=True)
class CycleEvidence:
    cycle_id: str
    hypothesis: Hypothesis
    experiments: tuple[Experiment, ...]
    summary: RedactedValidationSummary | None
    mechanism_id: str


def _lesson_id(cycle_id: str, tag: str) -> str:
    return "les_" + sha256_hex(f"{cycle_id}|{tag}")[:26].upper()


def _metric(exps: Sequence[Experiment], key: str) -> list[float]:
    return [e.metrics[key] for e in exps if key in e.metrics]


def derive_lessons(ev: CycleEvidence, now: UtcDatetime) -> list[Lesson]:
    done = [e for e in ev.experiments if e.status is ExperimentStatus.SUCCEEDED]
    net = _metric(done, "net_sharpe")
    gross = _metric(done, "gross_sharpe")
    turnover = _metric(done, "annual_turnover")
    symbols = sorted({i.symbol for i in ev.hypothesis.instruments})
    base_topics = (ev.mechanism_id, ev.hypothesis.family_id, *symbols)
    refs = (ev.hypothesis.id, *(e.id for e in ev.experiments))
    if ev.summary is not None:
        refs = (*refs, ev.summary.report_id)
    failed = set(ev.summary.failed_gate_names) if ev.summary else set()
    passed = ev.summary is not None and ev.summary.overall is GateOutcome.PASS
    outcome = "success" if passed else "failure"
    lessons: list[Lesson] = []

    def add(tag: str, statement: str, topics: tuple[str, ...], result: str = outcome) -> None:
        lessons.append(
            Lesson(
                id=_lesson_id(ev.cycle_id, tag),
                statement=statement,
                topics=tuple(dict.fromkeys((*base_topics, *topics))),
                derived_from_ids=refs,
                outcome=result,
                claim_level=ClaimLevel.L2_HISTORICAL,
                created_by=AgentRole.META_LEARNING,
                created_at=now,
            )
        )

    best = max(net) if net else float("nan")
    median = statistics.median(net) if net else float("nan")
    verdict = (
        f"independent validation {ev.summary.overall.value}"
        + (f" (failed: {', '.join(sorted(failed))})" if failed else "")
        if ev.summary
        else "no validation decision"
    )
    add(
        "outcome",
        f"{MECHANISMS[ev.mechanism_id].name} on {', '.join(symbols)}: {len(ev.experiments)} "
        f"registered trials; research-window net Sharpe best {best:.2f}, median {median:.2f}; "
        f"{verdict}.",
        ("validation_outcome",),
    )

    if net and max(net) <= 0:
        add(
            "no_signal",
            f"No trial of {ev.mechanism_id} had positive research-window net Sharpe "
            f"(best {best:.2f}); the mechanism is unsupported on this sample at these "
            "parameters.",
            ("no_signal",),
        )
    if gross and net and statistics.median(gross) > 0 >= median:
        add(
            "cost_fragility",
            f"Median gross Sharpe {statistics.median(gross):.2f} turns to {median:.2f} net: "
            f"costs dominate at median turnover {statistics.median(turnover or [0]):.0f}x/yr.",
            ("cost_fragility",),
        )
    elif "costs" in failed:
        add(
            "cost_fragility",
            "The strategy did not survive doubled costs on the holdout (G12).",
            ("cost_fragility",),
        )
    if net and best > 0.5 and ({"out_of_sample", "deflated_sharpe"} & failed):
        add(
            "decay",
            f"Research-window best net Sharpe {best:.2f} did not hold on the sealed holdout: "
            "selection on the research window, regime change, or both.",
            ("overfit",),
        )
    if "multiple_testing" in failed or "deflated_sharpe" in failed:
        add(
            "multiple_testing",
            f"With {len(ev.experiments)} trials in family {ev.hypothesis.family_id}, the effect "
            "was not large enough to survive multiple-testing deflation.",
            ("multiple_testing",),
        )
    if passed:
        add(
            "candidate",
            "Passed protocol 0.1.0-slice. Eligible for robustness extension; not deployable "
            "until the full protocol (M2) and paper congruence pass.",
            ("candidate",),
            "success",
        )
    return lessons


def propose_next(
    ev: CycleEvidence,
    lessons: Sequence[Lesson],
    tested_mechanisms: dict[str, int],
    now: UtcDatetime,
) -> NextExperimentProposal:
    tags = {t for les in lessons for t in les.topics}
    mech = MECHANISMS[ev.mechanism_id]
    grid = {k: list(v) for k, v in ev.hypothesis.search_space.items()}
    delta: dict[str, object]
    if "candidate" in tags:
        mechanism_id = ev.mechanism_id
        delta = {"param_grid": grid, "extend": "robustness: later data and second venue"}
        rationale = "Validated under the slice protocol; test robustness before any paper step."
    elif "cost_fragility" in tags:
        mechanism_id = ev.mechanism_id
        delta = {"param_grid": _slower(grid)}
        rationale = (
            "Gross edge was eroded by costs; re-test with slower signals (longer "
            "lookbacks, higher entry thresholds) to cut turnover."
        )
    elif "overfit" in tags or "multiple_testing" in tags:
        mechanism_id = ev.mechanism_id
        delta = {"param_grid": {k: [v[len(v) // 2]] for k, v in grid.items()}}
        rationale = (
            "Research-window performance did not survive deflation or the holdout; "
            "re-test a single pre-registered central parameter set to cut trials."
        )
    else:
        untested = [
            m for m in MECHANISMS if tested_mechanisms.get(m, 0) == 0 and m != ev.mechanism_id
        ]
        mechanism_id = (
            untested[0]
            if untested
            else min(MECHANISMS, key=lambda m: (tested_mechanisms.get(m, 0), m))
        )
        delta = {
            "mechanism": mechanism_id,
            "param_grid": {k: list(v) for k, v in MECHANISMS[mechanism_id].param_grid.items()},
        }
        rationale = (
            f"{mech.name} showed no supported edge; switch to the least-tested "
            f"mechanism {MECHANISMS[mechanism_id].name}."
        )
    return NextExperimentProposal(
        id="nxp_" + sha256_hex(f"{ev.cycle_id}|next")[:26].upper(),
        parent_cycle_id=ev.cycle_id,
        parent_hypothesis_id=ev.hypothesis.id,
        mechanism_id=mechanism_id,
        hypothesis_delta=delta,
        rationale=rationale,
        lesson_ids=tuple(les.id for les in lessons),
        validation_summary_id=ev.summary.report_id if ev.summary else None,
        proposed_by=AgentRole.META_LEARNING,
        created_at=now,
    )


def _slower(grid: dict[str, list[float | int | str]]) -> dict[str, list[float | int | str]]:
    out: dict[str, list[float | int | str]] = {}
    for key, values in grid.items():
        scaled: list[float | int | str] = []
        for v in values:
            if isinstance(v, int) and not isinstance(v, bool):
                scaled.append(int(v * 2))
            elif isinstance(v, float):
                scaled.append(round(v * 1.5, 6))
            else:
                scaled.append(v)
        out[key] = scaled
    return out
