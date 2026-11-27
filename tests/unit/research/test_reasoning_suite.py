"""The frozen reasoning suite accepts a complete case and rejects a slogan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from enigma.gateway.registry import ModelRegistry
from enigma.orchestrator.acceptance import verdict
from enigma.research.model_policy import (
    ROUTINE_TASKS,
    SCIENTIFIC_TASKS,
    decide,
    hosted_plan,
    role_for,
)
from enigma.research.reasoning_suite import load_suite, run_frozen_suite, score_task

_MECHANISM = (
    "A hedger who finances inventory across a settlement clock earns a payoff "
    "that is not the cashflow printed at one timestamp, because capacity is scarce."
)
_PERSISTENCE = "The settlement clock and the financing limit both stay in force across later bars."
_COMPETING = "A delayed index calculation could open the same gap without any financing limit."


def _gold(task: dict[str, Any]) -> dict[str, Any]:
    allowed = [str(item) for item in task["allowed_observables"]]
    diagnosis = " ".join(group[0] for group in task.get("diagnosis_groups") or [])
    method = " ".join(group[0] for group in task.get("methodology_groups") or [])
    expect = task.get("data_expect") or {}
    if "observable" in expect:
        needs = [
            {
                "observable": expect["observable"],
                "why": "unmet; the archive does not include it",
            }
        ]
    elif "observables_all" in expect:
        needs = [
            {"observable": name, "why": "required column"} for name in expect["observables_all"]
        ]
    elif "text_groups" in expect:
        phrase = " ".join(group[0] for group in expect["text_groups"])
        needs = [{"observable": allowed[0], "why": phrase}]
    elif "status_any" in expect:
        needs = [
            {
                "observable": "liquidation_prints",
                "why": "unmet; not among the acquired columns",
            }
        ]
    else:
        needs = [{"observable": allowed[0], "why": "required to compute the prediction"}]
    return {
        "mechanism": _MECHANISM,
        "persistence": _PERSISTENCE,
        "participants": [
            {
                "name": "hedger",
                "incentive": "earn the periodic cashflow",
                "constraint": "balance-sheet capacity is finite",
            }
        ],
        "competing_explanations": [_COMPETING],
        "supporting_evidence_ids": [item["id"] for item in task["evidence"]],
        "contradictory_evidence_ids": list(task["contradictory_ids"]),
        "predictions": [
            {
                "statement": "The next holding window changes sign when both series are extreme.",
                "observable": allowed[0],
                "discriminates": "a cashflow-only account",
            }
        ],
        "data_needs": needs,
        "program": {
            "design_class": "nonlinear",
            "computations": [
                {"op": "zscore", "inputs": [allowed[0]]},
                {"op": "product", "inputs": [allowed[0], allowed[min(1, len(allowed) - 1)]]},
            ],
        },
        "methodology": {"estimator": "deflated Sharpe", "error_control": method or "holdout"},
        "failure_diagnosis": diagnosis or "not applicable",
    }


def test_suite_is_frozen_and_a_complete_case_passes() -> None:
    suite = load_suite()
    assert suite["version"] == "reasoning-suite-v1"
    assert len(suite["tasks"]) >= 30
    facets = {task["facet"] for task in suite["tasks"]}
    assert "economic_mechanism" in facets
    assert "methodological_diagnosis" in facets
    for task in suite["tasks"]:
        scored = score_task(task, _gold(task))
        assert scored["acceptable"], (task["id"], scored["dimensions"])


def test_a_slogan_is_not_a_scientific_case() -> None:
    task = load_suite()["tasks"][0]
    scored = score_task(
        task,
        {
            "mechanism": "funding crowding reversal",
            "program": "use a z-score",
            "predictions": ["it goes up"],
            "data_needs": ["prices"],
            "supporting_evidence_ids": "ev",
            "contradictory_evidence_ids": [],
        },
    )
    assert scored["acceptable"] is False
    assert scored["dimensions"]["structured_output"] is False


def test_runner_refuses_a_hosted_alias_and_prices_it() -> None:
    def complete(model: str, prompt: str) -> str:
        assert model == "qwen2.5:7b-instruct"
        assert "Evidence:" in prompt
        task_id = prompt.split("\n", 1)[0]
        task = next(item for item in load_suite()["tasks"] if item["prompt"] == task_id)
        return json.dumps(_gold(task))

    result = run_frozen_suite(
        complete=complete,
        models=("qwen2.5:7b-instruct", "claude-sonnet"),
    )
    assert result["paid_calls"] == 0
    assert result["hosted_refused"] == ["claude-sonnet"]
    assert result["models"][0]["acceptable"] == result["tasks"]
    assert result["models"][0]["usd"] == 0.0
    plan = result["hosted_plan"]
    assert plan["paid_calls"] == 0
    assert plan["cost_per_acceptable_case_usd"] is None
    assert plan["frontier_scientific"][0]["status"] == "not_called"
    assert plan["frontier_scientific"][0]["estimated_usd_per_call"] > 0


def test_scientific_routes_are_local_and_free() -> None:
    registry = ModelRegistry.load(Path(__file__).resolve().parents[3] / "registries" / "models")
    for task_kind in (*SCIENTIFIC_TASKS, *ROUTINE_TASKS):
        decision = decide(task_kind)
        assert decision["hosted_called"] is False
        assert decision["max_cost_per_call_usd"] == 0.0
        route = registry.route(role_for(task_kind), task_kind)
        assert route.max_cost_per_call_usd == 0.0
        for alias in route.candidates:
            spec = registry.resolve_model(alias)
            assert registry.providers[spec.provider].local
    plan = hosted_plan(n_tasks=32, registry=registry)
    assert plan["paid_calls"] == 0
    names = {row["alias"] for row in plan["frontier_scientific"]}
    assert {"claude-sonnet", "gpt-5", "gemini-pro"} <= names


def test_blocked_criteria_are_not_acceptance() -> None:
    blocked = verdict([{"status": "pass"}, {"status": "blocked"}])
    assert blocked["label"] == "NOT ACCEPTED"
    assert blocked["code"] == 3
    failed = verdict([{"status": "fail"}])
    assert failed["code"] == 2
    assert failed["label"] == "NOT ACCEPTED"
    accepted = verdict([{"status": "pass"}, {"status": "pass"}])
    assert accepted["code"] == 0
    assert accepted["label"] == "ACCEPTED"
