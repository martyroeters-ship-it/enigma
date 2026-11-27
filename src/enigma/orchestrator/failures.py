"""Bounded retries for objectives the institution cannot currently execute.

Infrastructure, model, data, and implementation failures are not evidence about a
market mechanism. Each one is recorded, waited out with exponential backoff, and then
quarantined with a remediation. A scientific abort is not retried. Clearing the block
is a deliberate act; the scheduler does not discover a way around it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from enigma.research.horizons import sampling_frequency
from enigma.schemas.research import ResearchObjective
from enigma.store import Stores

FLAG = "execution.block"
MAX_ATTEMPTS = 3

_CLASSES = ("infrastructure", "model", "data", "implementation", "scientific")

_REMEDIATION = {
    "model": (
        "The model did not complete this objective. Restore the local model before clearing "
        "execution.block. This failure is not evidence against a mechanism."
    ),
    "data": (
        "The required data could not be acquired or did not match the sampling frequency. "
        "Register a point-in-time dataset before clearing execution.block."
    ),
    "implementation": (
        "The strategy or experiment did not execute. Repair the implementation before "
        "clearing execution.block. This is not a scientific result."
    ),
    "infrastructure": (
        "The runtime failed closed. Repair the sandbox, protocol pin, or artifact integrity "
        "before clearing execution.block."
    ),
    "scientific": (
        "The cycle stopped for a scientific or methodological reason. It will not be "
        "rescheduled automatically."
    ),
}


def classify_failure(reason: str) -> str:
    text = reason.lower()
    if any(word in text for word in ("falsified", "no edge", "hypothesis rejected")):
        return "scientific"
    if any(word in text for word in ("ollama", "model gateway", "provider unavailable", "llm ")):
        return "model"
    if any(
        word in text
        for word in ("no data", "missing column", "dataset", "acquire", "empty panel", "frequency")
    ):
        return "data"
    if any(
        word in text for word in ("strategy", "codegen", "entrypoint", "syntax", "not executable")
    ):
        return "implementation"
    if any(word in text for word in ("integrity", "sandbox", "docker", "protocol", "permission")):
        return "infrastructure"
    return "infrastructure"


def block_key(objective: ResearchObjective | None, fallback: str) -> str:
    if objective is None:
        return fallback
    topics = objective.topics
    if "continuous-backlog" in topics and len(topics) >= 3:
        mechanism, symbol, horizon = topics[0], topics[1], topics[2]
        if sampling_frequency((horizon,), "") == horizon:
            return "opp_" + f"{mechanism}|{symbol}|{horizon}".replace("|", "_")
    return objective.id


def load_blocks(stores: Stores) -> dict[str, dict[str, Any]]:
    raw = stores.flags.get(FLAG)
    if not raw:
        return {}
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    blocks = body.get("blocks") if isinstance(body, dict) else None
    if not isinstance(blocks, dict):
        return {}
    return {str(key): value for key, value in blocks.items() if isinstance(value, dict)}


def _save(stores: Stores, blocks: dict[str, dict[str, Any]]) -> None:
    stores.flags.set(FLAG, json.dumps({"blocks": blocks}, sort_keys=True), "system")


def record_execution_failure(stores: Stores, objective_id: str, reason: str, now: datetime) -> str:
    """Record one failed attempt. Returns the failure class."""
    objective = stores.documents.get(ResearchObjective, objective_id) if objective_id else None
    key = block_key(objective, objective_id or "unknown")
    failure_class = classify_failure(reason)
    if failure_class not in _CLASSES:
        failure_class = "infrastructure"
    blocks = load_blocks(stores)
    previous = blocks.get(key, {})
    attempts = int(previous.get("attempts") or 0) + 1
    if failure_class == "scientific" or attempts >= MAX_ATTEMPTS:
        next_eligible: str | None = None
    else:
        wait = timedelta(hours=2 ** (attempts - 1))
        next_eligible = (now + wait).isoformat()
    blocks[key] = {
        "failure_class": failure_class,
        "attempts": attempts,
        "reason": reason[:500],
        "remediation": _REMEDIATION[failure_class],
        "next_eligible": next_eligible,
        "updated_at": now.isoformat(),
    }
    _save(stores, blocks)
    stores.audit.record(
        "system",
        "execution.blocked",
        key,
        {"failure_class": failure_class, "attempts": attempts, "permanent": next_eligible is None},
    )
    return failure_class


def blocked_ids(stores: Stores, now: datetime) -> set[str]:
    """Objectives that must not be opened at `now`."""
    blocked: set[str] = set()
    for key, body in load_blocks(stores).items():
        eligible = body.get("next_eligible")
        if eligible is None:
            blocked.add(key)
            continue
        try:
            instant = datetime.fromisoformat(str(eligible))
        except ValueError:
            blocked.add(key)
            continue
        if instant.tzinfo is None:
            blocked.add(key)
            continue
        if now < instant:
            blocked.add(key)
    return blocked


def block_reason(stores: Stores, key: str) -> str:
    body = load_blocks(stores).get(key)
    if not body:
        return ""
    remediation = str(body.get("remediation") or "")
    failure_class = str(body.get("failure_class") or "infrastructure")
    return f"{failure_class}: {remediation}"
