"""Capability routing while paid comparison calls are unauthorized.

Routine extraction and scientific case construction both stay on local models
at a zero per-call ceiling. Hosted frontier models are listed with a price
estimate and are not called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from enigma.gateway.registry import ModelRegistry
from enigma.schemas.common import AgentRole

SCIENTIFIC_TASKS = (
    "quant_research.design_experiment",
    "research.scientific_case",
)
ROUTINE_TASKS = (
    "quant.hypothesis",
    "strategy.codegen",
    "intelligence.extract_hypothesis",
    "intelligence.summarize_source",
)
LOCAL_CANDIDATES = ("local-qwen", "local-qwen-3b")
ASSUMED_INPUT_TOKENS = 1200
ASSUMED_OUTPUT_TOKENS = 600

_FRONTIER = ("claude-sonnet", "gpt-5", "gemini-pro", "deepseek-reasoner")
_ROUTINE_HOSTED = ("claude-haiku", "gemini-flash", "gpt-4.1-mini")


def comparison_path() -> Path:
    return (
        Path(__file__).resolve().parents[3] / "registries" / "research" / "model_comparison_v1.yaml"
    )


def decide(task_kind: str) -> dict[str, Any]:
    """Local, zero-cost route. Hosted candidates are withheld in this build."""
    if task_kind in SCIENTIFIC_TASKS:
        capability = "scientific"
    elif task_kind in ROUTINE_TASKS:
        capability = "routine"
    else:
        capability = "unspecified"
    return {
        "task_kind": task_kind,
        "capability": capability,
        "candidates": list(LOCAL_CANDIDATES),
        "max_cost_per_call_usd": 0.0,
        "hosted_called": False,
        "reason": "Hosted models stay in the comparison plan until a paid run is authorized.",
    }


def role_for(task_kind: str) -> AgentRole:
    if task_kind.startswith("intelligence."):
        return AgentRole.INTELLIGENCE
    if task_kind.startswith("strategy."):
        return AgentRole.STRATEGY_ENGINEERING
    return AgentRole.QUANT_RESEARCH


def hosted_plan(
    n_tasks: int = 32,
    registry: ModelRegistry | None = None,
) -> dict[str, Any]:
    """Price one frozen-suite pass. Does not call a provider."""
    reg = registry if registry is not None else ModelRegistry.load(_registry_dir())
    document = load_comparison_document()
    frontier = _aliases(document.get("frontier_scientific"), _FRONTIER)
    routine = _aliases(document.get("routine_stays_local"), _ROUTINE_HOSTED)
    return {
        "paid_calls": 0,
        "status": "not_called",
        "authorization": document.get("authorization", "not granted"),
        "assumed_input_tokens": ASSUMED_INPUT_TOKENS,
        "assumed_output_tokens": ASSUMED_OUTPUT_TOKENS,
        "cost_per_acceptable_case_usd": None,
        "notice": (
            "Prices are the estimates in pricing.yaml. "
            "Cost per scientifically acceptable case is undefined until that model is scored."
        ),
        "frontier_scientific": [_estimate(reg, alias, n_tasks) for alias in frontier],
        "routine_not_routed": [_estimate(reg, alias, n_tasks) for alias in routine],
    }


def _aliases(raw: Any, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return fallback
    names: list[str] = []
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("alias"), str):
            names.append(item["alias"])
        elif isinstance(item, str):
            names.append(item)
    return tuple(names) or fallback


def _estimate(registry: ModelRegistry, alias: str, n_tasks: int) -> dict[str, Any]:
    spec = registry.resolve_model(alias)
    price = registry.price_for(spec)
    per_call = None
    if price is not None:
        per_call = (
            price.input * ASSUMED_INPUT_TOKENS + price.output * ASSUMED_OUTPUT_TOKENS
        ) / 1_000_000
    per_suite = None if per_call is None else per_call * n_tasks
    return {
        "alias": alias,
        "concrete": spec.concrete,
        "status": "not_called",
        "estimated_usd_per_call": per_call,
        "estimated_usd_per_suite": per_suite,
        "cost_per_acceptable_case_usd": None,
    }


def _registry_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "registries" / "models"


def load_comparison_document() -> dict[str, Any]:
    loaded = yaml.safe_load(comparison_path().read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}
