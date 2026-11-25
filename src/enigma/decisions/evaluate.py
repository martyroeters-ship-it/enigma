"""Score measured decision providers. An unmeasured provider cannot be adopted."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from enigma.core.errors import ProviderUnavailableError
from enigma.decisions.providers import DecisionResult
from enigma.decisions.tasks import DecisionItem, frozen_items, label_set_hash


@dataclass(frozen=True)
class ProviderScore:
    provider_id: str
    measured: bool
    macro_f1: float | None
    latency_ms: float | None
    cost_usd: float
    errors: int


@dataclass(frozen=True)
class EvalReport:
    label_set_hash: str
    n_items: int
    scores: tuple[ProviderScore, ...]
    selected: str | None


@dataclass(frozen=True)
class Adoption:
    adopt: bool
    reason: str


class _Provider(Protocol):
    provider_id: str

    def decide(self, item: DecisionItem) -> DecisionResult: ...


def evaluate(
    providers: tuple[_Provider, ...], items: tuple[DecisionItem, ...] | None = None
) -> EvalReport:
    rows = items if items is not None else frozen_items()
    digest = label_set_hash(rows)
    scores = tuple(_score(provider, rows) for provider in providers)
    return EvalReport(
        label_set_hash=digest,
        n_items=len(rows),
        scores=scores,
        selected=_select(scores),
    )


def adoption_decision(report: EvalReport, thresholds_path: Path) -> Adoption:
    raw = yaml.safe_load(thresholds_path.read_text(encoding="utf-8"))
    if report.label_set_hash != raw["label_set_hash"]:
        return Adoption(False, "label_set_hash_mismatch")
    if report.n_items < int(raw["label_set_size_min"]):
        return Adoption(False, "label_set_too_small")
    typesafe = next((row for row in report.scores if row.provider_id == "typesafe"), None)
    if typesafe is None or not typesafe.measured or typesafe.macro_f1 is None:
        return Adoption(False, "typesafe_not_measured")
    best = max(
        (
            row.macro_f1
            for row in report.scores
            if row.provider_id != "typesafe" and row.macro_f1 is not None
        ),
        default=None,
    )
    if best is None or typesafe.macro_f1 < best:
        return Adoption(False, "quality_below_baseline")
    limit = float(raw["max_p50_latency_ms"])
    if typesafe.latency_ms is None or typesafe.latency_ms > limit:
        return Adoption(False, "latency_above_threshold")
    return Adoption(True, "thresholds_met")


def _score(provider: _Provider, items: tuple[DecisionItem, ...]) -> ProviderScore:
    correct = 0
    errors = 0
    latency = 0.0
    cost = 0.0
    labels = sorted({item.label for item in items})
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    for item in items:
        try:
            result = provider.decide(item)
        except ProviderUnavailableError:
            errors += 1
            continue
        latency += result.latency_ms
        cost += result.cost_usd
        hit = result.choice == item.label
        correct += int(hit)
        for label in labels:
            predicted = result.choice == label
            actual = item.label == label
            if predicted and actual:
                counts[label]["tp"] += 1
            elif predicted:
                counts[label]["fp"] += 1
            elif actual:
                counts[label]["fn"] += 1
    measured = errors < len(items) and correct + errors > 0 and errors == 0
    if not measured:
        return ProviderScore(provider.provider_id, False, None, None, cost, errors or len(items))
    f1s = []
    for label in labels:
        tp, fp, fn = counts[label]["tp"], counts[label]["fp"], counts[label]["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        )
    answered = len(items) - errors
    return ProviderScore(
        provider.provider_id,
        True,
        sum(f1s) / len(f1s),
        latency / answered,
        cost,
        errors,
    )


def _select(scores: tuple[ProviderScore, ...]) -> str | None:
    measured = [row for row in scores if row.measured and row.macro_f1 is not None]
    if not measured:
        return None
    winner = min(
        measured,
        key=lambda row: (
            -(row.macro_f1 or 0.0),
            row.latency_ms if row.latency_ms is not None else float("inf"),
            row.cost_usd,
            row.provider_id,
        ),
    )
    return winner.provider_id
