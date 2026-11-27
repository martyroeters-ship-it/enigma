"""Score scientific reasoning without treating a fluent paragraph as evidence.

The scorer is deterministic. A live run uses only the local Ollama server and
records a miss when that server is absent. Paid providers are not called.
Frontier models are named as alternatives that were not invoked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen

FRONTIER_NOT_CALLED = (
    "Claude, GPT, and Gemini-class models are plausible alternatives for the "
    "high-level tasks below. They were not called. Calling them would be paid API use."
)
LOCAL_MODEL = "qwen2.5:7b-instruct"
_OLLAMA = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class TaskScore:
    task_id: str
    kind: str
    passed: bool
    detail: str


def score_case_text(
    body: dict[str, Any],
    evidence_ids: set[str],
    contradictory_ids: set[str] | None = None,
) -> tuple[TaskScore, ...]:
    """Score one structured case. Routine extraction is a separate task."""
    scores: list[TaskScore] = []
    mechanism = str(body.get("mechanism") or "")
    persistence = str(body.get("persistence") or "")
    participants = body.get("participants")
    scores.append(
        TaskScore(
            "economic_reasoning",
            "scientific",
            bool(mechanism and persistence and isinstance(participants, list) and participants),
            "needs a mechanism, a reason it persists, and a participant",
        )
    )
    cited = _id_set(body.get("supporting_evidence_ids"))
    scores.append(
        TaskScore(
            "literature_synthesis",
            "scientific",
            len(cited & evidence_ids) >= 2,
            "must cite at least two supplied evidence ids",
        )
    )
    contradictory = _id_set(body.get("contradictory_evidence_ids"))
    if contradictory_ids is None:
        contradiction_ok = bool(contradictory) and contradictory <= evidence_ids
    elif not contradictory_ids:
        contradiction_ok = not contradictory
    else:
        contradiction_ok = bool(contradictory) and contradictory <= contradictory_ids
    scores.append(
        TaskScore(
            "contradiction",
            "scientific",
            contradiction_ok,
            "cite a supplied contradictory excerpt, or none when none was supplied",
        )
    )
    predictions = _as_list(body.get("predictions"))
    falsifiable = any(
        isinstance(item, dict) and item.get("statement") and item.get("observable")
        for item in predictions
    )
    scores.append(
        TaskScore(
            "hypothesis",
            "scientific",
            falsifiable,
            "a prediction needs a statement and an observable",
        )
    )
    program = _as_mapping(body.get("program"))
    design = str(program.get("design_class") or "")
    comps = _as_list(program.get("computations"))
    ops = {"identity", "zscore", "rank", "product", "difference", "threshold", "regime"}
    structured_ops = [item for item in comps if isinstance(item, dict) and item.get("op") in ops]
    scores.append(
        TaskScore(
            "methodology",
            "scientific",
            bool(design) and bool(structured_ops),
            "computations are operations on named columns, not a prose slogan",
        )
    )
    scores.append(
        TaskScore(
            "experiment",
            "scientific",
            any(
                isinstance(item, dict) and str(item.get("discriminates") or "").strip()
                for item in predictions
            ),
            "one prediction must say which competing explanation it discriminates",
        )
    )
    needs = _as_list(body.get("data_needs"))
    scores.append(
        TaskScore(
            "data_requirements",
            "scientific",
            any(isinstance(item, dict) and item.get("observable") for item in needs),
            "data needs name an observable",
        )
    )
    scores.append(
        TaskScore(
            "structured_output",
            "extraction",
            isinstance(body, dict) and "mechanism" in body and "program" in body,
            "the object has the required keys",
        )
    )
    return tuple(scores)


def _as_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _as_mapping(value: object) -> dict[Any, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _id_set(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def score_failure_diagnosis(text: str) -> TaskScore:
    lowered = text.lower()
    named = "g02" in lowered or "chronological" in lowered
    passed = named and "not evidence against" in lowered
    return TaskScore(
        "failure_diagnosis",
        "scientific",
        passed,
        "a G02 or chronological failure is a method failure, not evidence against the mechanism",
    )


def probe_local_model(timeout: float = 2.0) -> str:
    """Return 'available', 'absent', or an error string. Does not send a prompt."""
    try:
        with urlopen(f"{_OLLAMA}/api/version", timeout=timeout) as response:  # noqa: S310
            if response.status != 200:
                return "absent"
    except Exception:
        return "absent"
    return "available"


def run_local_suite(
    evidence_ids: set[str],
    complete: Any | None = None,
    contradictory_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Score a completion. `complete` is injected in tests. Live mode calls Ollama once."""
    if complete is None:
        if probe_local_model() != "available":
            return {
                "status": "model_unavailable",
                "model": LOCAL_MODEL,
                "paid_calls": 0,
                "frontier_not_called": FRONTIER_NOT_CALLED,
                "scores": [],
            }
        complete = _ollama_complete
    try:
        raw = complete(_PROMPT + " Evidence ids: " + ", ".join(sorted(evidence_ids)))
    except Exception as exc:
        return {
            "status": "model_failed",
            "model": LOCAL_MODEL,
            "error": str(exc),
            "paid_calls": 0,
            "frontier_not_called": FRONTIER_NOT_CALLED,
            "scores": [],
        }
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    body = parsed if isinstance(parsed, dict) else {}
    scores = list(score_case_text(body, evidence_ids, contradictory_ids))
    scores.append(score_failure_diagnosis(str(body.get("failure_diagnosis") or "")))
    return {
        "status": "scored",
        "model": LOCAL_MODEL,
        "paid_calls": 0,
        "frontier_not_called": FRONTIER_NOT_CALLED,
        "scores": [score.__dict__ for score in scores],
        "passed": sum(1 for score in scores if score.passed),
        "total": len(scores),
        "response_excerpt": raw[:1500],
        "parsed": body,
    }


_PROMPT = (
    "Return one JSON object with mechanism, persistence, participants, "
    "supporting_evidence_ids, contradictory_evidence_ids, predictions, "
    "data_needs, program, and failure_diagnosis. "
    "supporting_evidence_ids and contradictory_evidence_ids are arrays of strings. "
    "program is an object with design_class and computations, not a sentence. "
    "Use only the evidence ids you were given. "
    "If no contradictory id was supplied, contradictory_evidence_ids is an empty array. "
    "Do not claim the JSON is empirical proof."
)


def _ollama_complete(prompt: str) -> str:
    payload = json.dumps(
        {"model": LOCAL_MODEL, "prompt": prompt, "stream": False, "format": "json"}
    ).encode()
    request = Request(  # noqa: S310
        f"{_OLLAMA}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:  # noqa: S310
        body = json.loads(response.read().decode())
    return str(body.get("response") or "")
