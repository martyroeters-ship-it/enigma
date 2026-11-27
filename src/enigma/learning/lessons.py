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


def _mechanism_label(mechanism_id: str, hypothesis: Hypothesis) -> str:
    """LLM domains are not in the M1 mechanism library; fall back to hypothesis text."""
    known = MECHANISMS.get(mechanism_id)
    if known is not None:
        return known.name
    for candidate in (hypothesis.mechanism, hypothesis.title, mechanism_id):
        text = (candidate or "").strip()
        if text:
            return text
    return mechanism_id or "unknown_mechanism"


def _scope(ev: CycleEvidence, gross_state: str, net: list[float], median: float) -> str:
    """How far a failed result is allowed to travel. Method gates are not a mechanism verdict."""
    names = ev.summary.failed_gate_names if ev.summary else ()
    gate_ids = {name.split(":", 1)[0] for name in names}
    if ev.summary is None:
        return "verdict:no_decision"
    if gate_ids and gate_ids <= {"G01", "G02", "G16"}:
        return "verdict:method"
    if gross_state == "positive" and ({"G03", "G12"} & gate_ids or (net and median <= 0)):
        return "verdict:cost"
    if gross_state == "absent":
        return "verdict:hypothesis"
    return "verdict:direction"


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
    failed = (
        {name.split(":", 1)[-1] for name in ev.summary.failed_gate_names} if ev.summary else set()
    )
    passed = ev.summary is not None and ev.summary.overall is GateOutcome.PASS
    outcome = "success" if passed else "failure"
    lessons: list[Lesson] = []
    mechanism_name = _mechanism_label(ev.mechanism_id, ev.hypothesis)

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
    best_gross = max(gross) if gross else float("nan")
    median_gross = statistics.median(gross) if gross else float("nan")
    gross_state = "unknown" if not gross else ("positive" if best_gross > 0 else "absent")
    verdict = (
        f"independent validation {ev.summary.overall.value}"
        + (f" (failed: {', '.join(sorted(failed))})" if failed else "")
        if ev.summary
        else "no validation decision"
    )
    scope = _scope(ev, gross_state, net, median)
    add(
        "outcome",
        f"{mechanism_name} on {', '.join(symbols)}: {len(ev.experiments)} "
        f"registered trials; research-window net Sharpe best {best:.2f}, median {median:.2f}; "
        + (
            f"gross Sharpe best {best_gross:.2f}, median {median_gross:.2f}; "
            if gross
            else "gross Sharpe was not recorded; "
        )
        + f"{verdict}. Scope {scope}.",
        ("validation_outcome", scope),
    )

    if net and max(net) <= 0:
        add(
            "no_signal",
            f"No trial of {ev.mechanism_id} had positive research-window net Sharpe "
            f"(best {best:.2f}); the mechanism is unsupported on this sample at these "
            "parameters.",
            ("no_signal",),
        )
    # A failed cost gate is not evidence that costs removed an edge. That claim
    # requires a positive gross Sharpe that the net result then loses.
    if gross_state == "positive" and net and median <= 0:
        add(
            "cost_fragility",
            f"Median gross Sharpe {median_gross:.2f} turns to {median:.2f} net: "
            f"costs dominate at median turnover {statistics.median(turnover or [0]):.0f}x/yr.",
            ("cost_fragility",),
        )
    elif gross_state == "positive" and "costs" in failed:
        add(
            "cost_fragility",
            "A positive gross Sharpe did not survive doubled costs on the holdout (G12).",
            ("cost_fragility",),
        )
    elif gross_state == "absent":
        add(
            "absent_gross_edge",
            f"No trial had a positive research-window gross Sharpe (best {best_gross:.2f}, "
            f"median {median_gross:.2f}). Trading costs did not remove an edge; the gross "
            "result was already negative"
            + (" and the cost-stress gate also failed." if "costs" in failed else "."),
            ("absent_gross_edge",),
        )
    elif "costs" in failed:
        add(
            "cost_stress_inconclusive",
            "The cost-stress gate failed and gross Sharpe was not recorded, so this is not "
            "evidence that transaction costs destroyed a profitable edge.",
            ("cost_stress_inconclusive",),
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


def _least_tested_mechanism(
    tested_mechanisms: dict[str, int], *, exclude: str | None = None
) -> str:
    untested = [m for m in MECHANISMS if tested_mechanisms.get(m, 0) == 0 and m != exclude]
    if untested:
        return untested[0]
    return min(MECHANISMS, key=lambda m: (tested_mechanisms.get(m, 0), m))


def propose_next(
    ev: CycleEvidence,
    lessons: Sequence[Lesson],
    tested_mechanisms: dict[str, int],
    now: UtcDatetime,
) -> NextExperimentProposal:
    tags = {t for les in lessons for t in les.topics}
    mech_name = _mechanism_label(ev.mechanism_id, ev.hypothesis)
    known = MECHANISMS.get(ev.mechanism_id)
    grid = {k: list(v) for k, v in ev.hypothesis.search_space.items()}
    delta: dict[str, object]
    # Unknown LLM domains (e.g. fam_cryptocurrency → cryptocurrency) are not in
    # MECHANISMS. Never invent strategy params for them: only label lessons with
    # hypothesis text, and route next trials to a registered library mechanism.
    if known is None:
        mechanism_id = _least_tested_mechanism(tested_mechanisms, exclude=ev.mechanism_id)
        delta = {
            "mechanism": mechanism_id,
            "param_grid": {k: list(v) for k, v in MECHANISMS[mechanism_id].param_grid.items()},
            "prior_domain": ev.mechanism_id,
        }
        rationale = (
            f"Domain {ev.mechanism_id!r} is not a registered M1 mechanism "
            f"({mech_name}); switch to least-tested library mechanism "
            f"{MECHANISMS[mechanism_id].name}."
        )
    elif "candidate" in tags:
        mechanism_id = ev.mechanism_id
        delta = {"param_grid": grid, "extend": "robustness: later data and second venue"}
        rationale = "Validated under the slice protocol; test robustness before any paper step."
    elif "cost_fragility" in tags:
        mechanism_id = ev.mechanism_id
        delta = {
            "param_grid": _slower(grid),
            "revisit": "positive_gross_destroyed_by_costs",
        }
        rationale = (
            "A positive gross Sharpe was lost after costs; re-test with slower signals "
            "(longer lookbacks, higher entry thresholds) to cut turnover."
        )
    elif "absent_gross_edge" in tags:
        mechanism_id = _least_tested_mechanism(tested_mechanisms, exclude=ev.mechanism_id)
        delta = {
            "mechanism": mechanism_id,
            "param_grid": {k: list(v) for k, v in MECHANISMS[mechanism_id].param_grid.items()},
        }
        rationale = (
            f"{mech_name} had no positive gross Sharpe, so trading costs did not remove an "
            f"edge and slowing the signal would not create one. Switch to the least-tested "
            f"mechanism {MECHANISMS[mechanism_id].name}."
        )
    elif "overfit" in tags or "multiple_testing" in tags:
        mechanism_id = ev.mechanism_id
        delta = {"param_grid": {k: [v[len(v) // 2]] for k, v in grid.items()}}
        rationale = (
            "Research-window performance did not survive deflation or the holdout; "
            "re-test a single pre-registered central parameter set to cut trials."
        )
    else:
        mechanism_id = _least_tested_mechanism(tested_mechanisms, exclude=ev.mechanism_id)
        delta = {
            "mechanism": mechanism_id,
            "param_grid": {k: list(v) for k, v in MECHANISMS[mechanism_id].param_grid.items()},
        }
        rationale = (
            f"{mech_name} showed no supported edge; switch to the least-tested "
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
        validation_summary_id=ev.summary.id if ev.summary else None,
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
