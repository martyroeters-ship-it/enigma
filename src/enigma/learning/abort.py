"""Structured failure lessons from aborted research cycles.

Aborted cycles never reach reflection or validation, so without this path the institution
learns nothing from them. Lessons here are L1 operational knowledge: they cite the abort
reason and the step where the cycle stopped, and they are never treated as validated edge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from enigma.core.hashing import sha256_hex
from enigma.schemas.common import AgentRole, ClaimLevel, UtcDatetime
from enigma.schemas.knowledge import Lesson


class AbortClass(StrEnum):
    INFRASTRUCTURE = "infrastructure"
    DATA = "data"
    RESEARCH = "research"
    STRATEGY_RUNTIME = "strategy_runtime"
    BUDGET = "budget"
    POLICY = "policy"


@dataclass(frozen=True)
class AbortDiagnosis:
    abort_class: AbortClass
    step: str
    root_cause: str
    prevention: str
    topics: tuple[str, ...]


_RULES: tuple[tuple[re.Pattern[str], AbortClass, str, str, tuple[str, ...]], ...] = (
    (
        re.compile(r"holdout rows changed|IntegrityError|reseal", re.I),
        AbortClass.INFRASTRUCTURE,
        "holdout_integrity",
        "Reload sealed holdouts with the availability-lag window so the last bar is preserved; "
        "verify the content hash before evaluating.",
        ("holdout", "integrity"),
    ),
    (
        re.compile(r"InvalidTransitionError|RESEARCH_REJECTED|cannot go .* -> DRAFT", re.I),
        AbortClass.POLICY,
        "lifecycle_collision",
        "Scope strategy package ids to the hypothesis so re-tested code gets a fresh lifecycle; "
        "never reopen a REJECTED subject.",
        ("lifecycle", "package_id"),
    ),
    (
        re.compile(
            r"twin backtests were not reproducible|twin_reproducible|not reproducible", re.I
        ),
        AbortClass.STRATEGY_RUNTIME,
        "non_reproducible_backtest",
        "Keep resource usage (CPU seconds) out of the result hash and the twin comparison; "
        "compare only economic metrics and the deterministic result hash.",
        ("reproducibility", "backtest"),
    ),
    (
        re.compile(
            r"no strategy trial ran successfully|AttributeError|SandboxExecutionError|"
            r"StrategyRejectedError|metadata\(\)",
            re.I,
        ),
        AbortClass.STRATEGY_RUNTIME,
        "strategy_runtime_failure",
        "Run every template and generated package through the real sandbox and truncation "
        "invariance checks before registering trials.",
        ("sandbox", "strategy"),
    ),
    (
        re.compile(r"BudgetExceeded|spend ceiling|daily cap|max_cycles", re.I),
        AbortClass.BUDGET,
        "budget_exhausted",
        "Authorize the planned spend before the call and stop scheduling when the remaining "
        "cycle or USD budget cannot cover the next attempt.",
        ("budget",),
    ),
    (
        re.compile(
            r"dataset|PIT certified|data task failed|ProviderUnavailableError.*binance", re.I
        ),
        AbortClass.DATA,
        "data_acquisition_failure",
        "Fail closed on quality checks; retry only after provenance and content-hash verification.",
        ("data", "pit"),
    ),
    (
        re.compile(
            r"not valid JSON|Expecting value|llm_error:|structured_output|StructuredOutputError",
            re.I,
        ),
        AbortClass.INFRASTRUCTURE,
        "invalid_model_json",
        "Retry local structured output in JSON mode, then the next local model on the task "
        "route. An empty model body is not a missing-evidence failure.",
        ("model", "json"),
    ),
    (
        re.compile(r"codegen_failed_after_repairs", re.I),
        AbortClass.INFRASTRUCTURE,
        "strategy_codegen_failed",
        "Compile a registered mechanism when the hypothesis names one. Keep unrestricted "
        "code generation for domain novel, and keep the static or sandbox error from the "
        "last attempt on the abort reason.",
        ("codegen", "strategy"),
    ),
    (
        re.compile(
            r"no external sources|no source-backed evidence|intelligence task failed|"
            r"hypothesis task failed",
            re.I,
        ),
        AbortClass.RESEARCH,
        "research_pipeline_failure",
        "Require at least one real source and one evidence excerpt before hypothesis generation; "
        "surface provider failures explicitly rather than inventing content.",
        ("research", "evidence"),
    ),
)


def diagnose_abort(reason: str, step: str | None) -> AbortDiagnosis:
    stopped = step or "unknown"
    for pattern, abort_class, root, prevention, topics in _RULES:
        if pattern.search(reason):
            return AbortDiagnosis(
                abort_class=abort_class,
                step=stopped,
                root_cause=root,
                prevention=prevention,
                topics=topics,
            )
    return AbortDiagnosis(
        abort_class=AbortClass.INFRASTRUCTURE,
        step=stopped,
        root_cause="unclassified_abort",
        prevention="Record the abort reason and step, then add a classifier rule once the "
        "failure mode is understood.",
        topics=("abort",),
    )


def lessons_from_aborted_cycle(
    cycle_id: str,
    abort_reason: str,
    *,
    step: str | None,
    hypothesis_id: str | None = None,
    now: UtcDatetime,
) -> list[Lesson]:
    """Return one structured failure lesson for an aborted cycle."""
    diagnosis = diagnose_abort(abort_reason, step)
    refs: tuple[str, ...] = (cycle_id, hypothesis_id) if hypothesis_id else (cycle_id,)
    statement = (
        f"Cycle aborted at {diagnosis.step} ({diagnosis.abort_class.value}/"
        f"{diagnosis.root_cause}): {abort_reason[:400].rstrip()}. "
        f"Prevention: {diagnosis.prevention}"
    )
    return [
        Lesson(
            id="les_" + sha256_hex(f"{cycle_id}|abort|{diagnosis.root_cause}")[:26].upper(),
            statement=statement,
            topics=tuple(
                dict.fromkeys(
                    ("abort", diagnosis.abort_class.value, diagnosis.root_cause, *diagnosis.topics)
                )
            ),
            derived_from_ids=refs,
            outcome="aborted",
            claim_level=ClaimLevel.L1_HYPOTHESIS,
            created_by=AgentRole.META_LEARNING,
            created_at=now,
        )
    ]
