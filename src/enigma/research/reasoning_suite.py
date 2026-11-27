"""Frozen scientific-reasoning suite.

The task file is content-hashed. A model is scientifically acceptable on a task
only when the structured reply meets that task's facet. Hosted models are priced
and not called.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from enigma.core.hashing import sha256_hex

SUITE_SHA256 = "edd0183869d7c32e399f6ff418084e0e8501b263c1b603472f8df46d561217cc"
_OLLAMA = "http://127.0.0.1:11434"
_OPS = frozenset({"identity", "zscore", "rank", "product", "difference", "threshold", "regime"})
_SEED_NAMES = frozenset(
    {
        "funding crowding reversal",
        "funding carry",
        "basis reversion",
        "funding momentum",
    }
)
_LOCAL_TAGS = {
    "qwen2.5:7b-instruct": "local-qwen",
    "qwen2.5:3b": "local-qwen-3b",
    "qwen3:8b": "local-qwen3",
    "gpt-oss:20b": "local-gpt-oss",
}


class SuiteError(ValueError):
    """The frozen suite file does not match its pin."""


def suite_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "registries" / "research" / "reasoning_suite_v1.json"


def load_suite() -> dict[str, Any]:
    raw = suite_path().read_bytes()
    digest = sha256_hex(raw)
    if digest != SUITE_SHA256:
        raise SuiteError(f"reasoning suite hash {digest} does not match the pin")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise SuiteError("reasoning suite is not an object")
    tasks = parsed.get("tasks")
    if not isinstance(tasks, list) or len(tasks) < 30:
        raise SuiteError("reasoning suite does not contain 30 tasks")
    return parsed


def prompt_for(task: dict[str, Any]) -> str:
    lines = [str(task.get("prompt") or ""), "Evidence:"]
    evidence = task.get("evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                lines.append(f"- {item.get('id')}: {item.get('excerpt')}")
    allowed = task.get("allowed_observables")
    names = ", ".join(str(item) for item in allowed) if isinstance(allowed, list) else ""
    lines.append(
        "Return one JSON object with mechanism, persistence, participants, "
        "competing_explanations, supporting_evidence_ids, contradictory_evidence_ids, "
        "predictions, data_needs, program, methodology, and failure_diagnosis. "
        "participants is an array of objects with name, incentive, and constraint. "
        "predictions is an array of objects with statement, observable, and discriminates. "
        "data_needs is an array of objects with observable and why. "
        "program is an object with design_class and computations. "
        "Each computation has op "
        "(identity, zscore, rank, product, difference, threshold, or regime) "
        "and inputs, an array of column names. "
        "Cite only the evidence ids above. "
        "If none of them contradicts the claim, contradictory_evidence_ids is empty. "
        "methodology is an object with estimator and error_control. "
        "Do not claim the JSON is empirical proof. "
        f"Allowed observables: {names}."
    )
    return "\n".join(lines)


def score_task(task: dict[str, Any], body: Any) -> dict[str, Any]:
    mapping = body if isinstance(body, dict) else {}
    facet = str(task.get("facet") or "")
    dims: dict[str, bool] = {
        "structured_output": _structured(mapping),
        "evidence_grounding": _grounding(task, mapping),
    }
    if facet == "economic_mechanism":
        dims["scientific_reasoning"] = _reasoning(mapping)
        dims["originality"] = _originality(task, mapping)
        dims["economic_plausibility"] = _plausibility(mapping)
    elif facet == "competing_explanations":
        dims["economic_plausibility"] = _plausibility(mapping)
    elif facet in {"falsifiable_prediction", "experimental_design"}:
        dims["experimental_validity"] = _experiment(task, mapping)
    elif facet == "statistical_methodology":
        dims["statistical_methodology"] = _groups(_blob(mapping), task.get("methodology_groups"))
    elif facet == "data_requirements":
        dims["data_requirements"] = _data(task, mapping)
    elif facet == "methodological_diagnosis":
        dims["methodological_diagnosis"] = _groups(_blob(mapping), task.get("diagnosis_groups"))
    return {
        "task_id": task.get("id"),
        "facet": facet,
        "dimensions": dims,
        "acceptable": all(dims.values()),
    }


def run_frozen_suite(
    complete: Callable[[str, str], str] | None = None,
    models: tuple[str, ...] | None = None,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    """Score local models. `complete` is injected in tests. Paid aliases are refused."""
    from enigma.research.model_policy import hosted_plan  # noqa: PLC0415

    suite = load_suite()
    tasks = [item for item in suite["tasks"] if isinstance(item, dict)]
    local, refused = _select_models(models, complete is not None)
    caller = complete or _local_complete
    summaries: list[dict[str, Any]] = []
    for model in local:
        rows = _score_model(model, tasks, caller, progress_path, summaries)
        summaries.append(_summarize(model, rows, len(tasks)))
    missing = [tag for tag in _LOCAL_TAGS if tag not in local and (models is None or tag in models)]
    return {
        "suite_version": suite.get("version"),
        "suite_sha256": SUITE_SHA256,
        "tasks": len(tasks),
        "paid_calls": 0,
        "status": "scored" if summaries else "model_unavailable",
        "models": summaries,
        "local_not_installed": missing,
        "hosted_refused": refused,
        "hosted_plan": hosted_plan(n_tasks=len(tasks)),
    }


def installed_local_tags(timeout: float = 2.0) -> tuple[str, ...]:
    try:
        with urlopen(f"{_OLLAMA}/api/tags", timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode())
    except Exception:
        return ()
    rows = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return ()
    names: list[str] = []
    for item in rows:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"].removesuffix(":latest"))
    return tuple(names)


def _score_model(
    model: str,
    tasks: list[dict[str, Any]],
    complete: Callable[[str, str], str],
    progress_path: Path | None,
    done: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        started = time.perf_counter()
        excerpt = ""
        try:
            raw = complete(model, prompt_for(task))
            excerpt = raw[:400]
            parsed: Any = json.loads(raw)
        except Exception as exc:
            parsed = {}
            excerpt = excerpt or str(exc)[:400]
        scored = score_task(task, parsed if isinstance(parsed, dict) else {})
        scored["latency_seconds"] = round(time.perf_counter() - started, 3)
        if not scored["acceptable"]:
            scored["excerpt"] = excerpt
        rows.append(scored)
        _write_progress(progress_path, done, model, rows, len(tasks))
    return rows


def _write_progress(
    path: Path | None,
    done: list[dict[str, Any]],
    model: str,
    rows: list[dict[str, Any]],
    n_tasks: int,
) -> None:
    if path is None:
        return
    payload = {"paid_calls": 0, "models": [*done, _summarize(model, rows, n_tasks)]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summarize(model: str, rows: list[dict[str, Any]], n_tasks: int) -> dict[str, Any]:
    dims: dict[str, list[bool]] = {}
    facets: dict[str, list[bool]] = {}
    latencies: list[float] = []
    for row in rows:
        facets.setdefault(str(row["facet"]), []).append(bool(row["acceptable"]))
        latencies.append(float(row.get("latency_seconds") or 0.0))
        dimensions = row.get("dimensions")
        if isinstance(dimensions, dict):
            for name, passed in dimensions.items():
                dims.setdefault(str(name), []).append(bool(passed))
    acceptable = sum(1 for row in rows if row["acceptable"])
    total = sum(latencies)
    return {
        "model": model,
        "alias": _LOCAL_TAGS.get(model, ""),
        "tasks": n_tasks,
        "acceptable": acceptable,
        "usd": 0.0,
        "paid_calls": 0,
        "cost_per_acceptable_case_usd": 0.0 if acceptable else None,
        "latency_seconds_total": round(total, 3),
        "latency_seconds_mean": round(total / len(rows), 3) if rows else None,
        "dimensions": {
            name: {"passed": sum(values), "applicable": len(values)}
            for name, values in dims.items()
        },
        "by_facet": {
            name: {"acceptable": sum(values), "tasks": len(values)}
            for name, values in facets.items()
        },
        "tasks_detail": [_detail(row) for row in rows],
    }


def _detail(row: dict[str, Any]) -> dict[str, Any]:
    dimensions = row.get("dimensions")
    failed: list[str] = []
    if isinstance(dimensions, dict):
        failed = [str(name) for name, ok in dimensions.items() if not ok]
    detail: dict[str, Any] = {
        "task_id": row.get("task_id"),
        "facet": row.get("facet"),
        "acceptable": row.get("acceptable"),
        "failed": failed,
        "latency_seconds": row.get("latency_seconds"),
    }
    if "excerpt" in row:
        detail["excerpt"] = row["excerpt"]
    return detail


def _select_models(models: tuple[str, ...] | None, injected: bool) -> tuple[list[str], list[str]]:
    if injected:
        chosen = list(models or ("qwen2.5:7b-instruct",))
        return (
            [item for item in chosen if item in _LOCAL_TAGS],
            [item for item in chosen if item not in _LOCAL_TAGS],
        )
    present = {name for name in installed_local_tags() if name in _LOCAL_TAGS}
    if models is None:
        return [tag for tag in _LOCAL_TAGS if tag in present], []
    local = [item for item in models if item in present]
    refused = [item for item in models if item not in _LOCAL_TAGS]
    return local, refused


def _local_complete(model: str, prompt: str) -> str:
    if model not in _LOCAL_TAGS:
        raise RuntimeError(f"refusing non-local model {model}")
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 700},
        }
    ).encode()
    request = Request(  # noqa: S310
        f"{_OLLAMA}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:  # noqa: S310
        body = json.loads(response.read().decode())
    return str(body.get("response") or "")


def _structured(body: dict[str, Any]) -> bool:
    keys = (
        "mechanism",
        "participants",
        "supporting_evidence_ids",
        "contradictory_evidence_ids",
        "predictions",
        "data_needs",
        "program",
    )
    if any(key not in body for key in keys):
        return False
    list_keys = (
        "participants",
        "supporting_evidence_ids",
        "contradictory_evidence_ids",
        "predictions",
        "data_needs",
    )
    return all(isinstance(body[key], list) for key in list_keys) and isinstance(
        body["program"], dict
    )


def _grounding(task: dict[str, Any], body: dict[str, Any]) -> bool:
    evidence_ids = _evidence_ids(task)
    expected = {str(item) for item in task.get("contradictory_ids") or []}
    supporting = body.get("supporting_evidence_ids")
    cited = body.get("contradictory_evidence_ids")
    if not isinstance(supporting, list) or not isinstance(cited, list):
        return False
    if not all(isinstance(item, str) for item in supporting):
        return False
    if not all(isinstance(item, str) for item in cited):
        return False
    if not set(supporting) <= evidence_ids or not set(cited) <= evidence_ids:
        return False
    if expected and not expected <= set(cited):
        return False
    if not expected and cited:
        return False
    if not [item for item in supporting if item not in expected]:
        return False
    if task.get("facet") == "evidence_grounding":
        return evidence_ids - expected <= set(supporting)
    return True


def _reasoning(body: dict[str, Any]) -> bool:
    if len(_text(body.get("mechanism"))) < 40 or len(_text(body.get("persistence"))) < 20:
        return False
    return _participant(body)


def _participant(body: dict[str, Any]) -> bool:
    for item in _as_list(body.get("participants")):
        if not isinstance(item, dict):
            continue
        incentive = _text(item.get("incentive"))
        constraint = _text(item.get("constraint"))
        if incentive and constraint and incentive.casefold() != constraint.casefold():
            return True
    return False


def _originality(task: dict[str, Any], body: dict[str, Any]) -> bool:
    mechanism = _text(body.get("mechanism"))
    if len(mechanism) < 40 or mechanism.casefold() in _SEED_NAMES:
        return False
    if _overlap(mechanism, _excerpts(task)) >= 0.72:
        return False
    return _overlap(mechanism, str(task.get("prompt") or "")) < 0.85


def _plausibility(body: dict[str, Any]) -> bool:
    mechanism = _text(body.get("mechanism"))
    persistence = _text(body.get("persistence"))
    if persistence and mechanism and persistence.casefold() == mechanism.casefold():
        return False
    for item in _as_list(body.get("competing_explanations")):
        text = item.strip() if isinstance(item, str) else ""
        if len(text) >= 20 and _overlap(text, mechanism) < 0.9:
            return True
    return False


def _experiment(task: dict[str, Any], body: dict[str, Any]) -> bool:
    if task.get("facet") == "falsifiable_prediction":
        return _prediction(task, body)
    return _program(body)


def _prediction(task: dict[str, Any], body: dict[str, Any]) -> bool:
    allowed = {_norm(str(item)) for item in task.get("allowed_observables") or []}
    for item in _as_list(body.get("predictions")):
        if not isinstance(item, dict):
            continue
        observable = _norm(_text(item.get("observable")))
        statement = _text(item.get("statement"))
        discriminates = _text(item.get("discriminates"))
        if statement and observable in allowed and discriminates:
            return True
    return False


def _program(body: dict[str, Any]) -> bool:
    program = body.get("program")
    if not isinstance(program, dict):
        return False
    valid = [item for item in _as_list(program.get("computations")) if _computation(item)]
    return bool(valid) and not (len(valid) == 1 and _lone_zscore(valid[0]))


def _computation(item: Any) -> bool:
    if not isinstance(item, dict) or _text(item.get("op")) not in _OPS:
        return False
    inputs = item.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        return False
    return all(isinstance(name, str) and name.strip() for name in inputs)


def _lone_zscore(item: dict[str, Any]) -> bool:
    inputs = item.get("inputs")
    lone = item.get("op") in {"zscore", "identity"}
    return lone and isinstance(inputs, list) and len(inputs) == 1


def _data(task: dict[str, Any], body: dict[str, Any]) -> bool:
    expect = task.get("data_expect")
    needs = body.get("data_needs")
    if not isinstance(expect, dict) or not isinstance(needs, list):
        return False
    observables, blob = _needs(needs)
    whole = _blob(body)
    required = expect.get("observable")
    if isinstance(required, str) and _norm(required) not in observables:
        return False
    status_any = expect.get("status_any")
    if isinstance(status_any, list) and not _any_phrase(status_any, blob, whole):
        return False
    if not _all_observables(expect.get("observables_all"), observables, blob):
        return False
    groups = expect.get("text_groups")
    if isinstance(groups, list):
        return _groups(whole, groups)
    return True


def _needs(needs: list[Any]) -> tuple[list[str], str]:
    observables: list[str] = []
    parts: list[str] = []
    for need in needs:
        if isinstance(need, dict):
            observable = _text(need.get("observable"))
            observables.append(_norm(observable))
            parts.append(f"{observable} {_text(need.get('why'))}")
    return observables, " ".join(parts).lower()


def _all_observables(required: Any, observables: list[str], blob: str) -> bool:
    if not isinstance(required, list):
        return True
    haystack = " ".join(observables) + _norm(blob)
    return all(_norm(str(item)) in haystack for item in required)


def _any_phrase(phrases: list[Any], blob: str, whole: str) -> bool:
    return any(str(item).lower() in blob or str(item).lower() in whole for item in phrases)


def _groups(text: str, groups: Any) -> bool:
    if not isinstance(groups, list) or not groups:
        return False
    lowered = text.lower()
    for group in groups:
        if not isinstance(group, list) or not group:
            return False
        if not any(str(option).lower() in lowered for option in group):
            return False
    return True


def _evidence_ids(task: dict[str, Any]) -> set[str]:
    rows = task.get("evidence")
    if not isinstance(rows, list):
        return set()
    return {str(item["id"]) for item in rows if isinstance(item, dict) and "id" in item}


def _excerpts(task: dict[str, Any]) -> str:
    rows = task.get("evidence")
    if not isinstance(rows, list):
        return ""
    return " ".join(str(item.get("excerpt") or "") for item in rows if isinstance(item, dict))


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _norm(value: str) -> str:
    return value.lower().replace("_", "").replace(" ", "")


def _overlap(left: str, right: str) -> float:
    tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
    if not tokens:
        return 1.0
    other = set(re.findall(r"[a-z0-9]+", right.lower()))
    return len(tokens & other) / len(tokens)


def _blob(body: dict[str, Any]) -> str:
    return json.dumps(body, default=str).lower()
