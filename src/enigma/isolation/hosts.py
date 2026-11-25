"""Process role, credential split, and clock skew. Fail closed.

`ENIGMA_HOST_ROLE=single` is the one-process institution used in development.
It is not an M5 deployment. Research, validation, and execution roles refuse
the directories and credentials that are not theirs. Nothing here opens a
network connection or enables trading.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from enigma.core.errors import IntegrityError, PermissionDeniedError

VALIDATION_CREDENTIAL = "ENIGMA_VALIDATION_CREDENTIAL"
EXECUTION_CREDENTIAL = "ENIGMA_EXECUTION_CREDENTIAL"
_CLEARANCE_FIELDS = ("id", "strategy_package_id", "decision", "reason_codes")


class HostRole(StrEnum):
    RESEARCH = "research"
    VALIDATION = "validation"
    EXECUTION = "execution"
    SINGLE = "single"


def host_role(environ: Mapping[str, str] | None = None) -> HostRole:
    raw = (os.environ if environ is None else environ).get("ENIGMA_HOST_ROLE", "single")
    text = raw.strip().lower()
    try:
        return HostRole(text)
    except ValueError:
        raise PermissionDeniedError(f"unknown ENIGMA_HOST_ROLE {text!r}") from None


def assert_may_open_validation(role: HostRole) -> None:
    if role not in (HostRole.VALIDATION, HostRole.SINGLE):
        raise PermissionDeniedError(f"host role {role.value} cannot open the validation directory")


def assert_may_read_research_trials(role: HostRole) -> None:
    if role in (HostRole.VALIDATION, HostRole.EXECUTION):
        raise PermissionDeniedError(f"host role {role.value} cannot read the research trial ledger")


def assert_credential_separation(role: HostRole, environ: Mapping[str, str] | None = None) -> None:
    """Refuse a process that holds another host's credential. Values are not read out."""
    env = os.environ if environ is None else environ
    if role is HostRole.RESEARCH and _set(env, VALIDATION_CREDENTIAL):
        raise PermissionDeniedError("research host holds a validation credential")
    if role is HostRole.RESEARCH and _set(env, EXECUTION_CREDENTIAL):
        raise PermissionDeniedError("research host holds an execution credential")
    if role is HostRole.VALIDATION and _set(env, EXECUTION_CREDENTIAL):
        raise PermissionDeniedError("validation host holds an execution credential")
    if role is HostRole.EXECUTION and _set(env, VALIDATION_CREDENTIAL):
        raise PermissionDeniedError("execution host holds a validation credential")


def assert_clock_skew(local: datetime, peer: datetime, embargo: timedelta) -> None:
    """A skew larger than the embargo must not widen the research window."""
    if embargo < timedelta(0):
        raise ValueError("embargo must be non-negative")
    skew = abs(local - peer)
    if skew > embargo:
        raise IntegrityError(f"host clock skew {skew} exceeds the embargo {embargo}")


def peer_clock(environ: Mapping[str, str] | None = None) -> datetime | None:
    raw = (os.environ if environ is None else environ).get("ENIGMA_PEER_CLOCK", "").strip()
    if not raw:
        return None
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        raise IntegrityError("ENIGMA_PEER_CLOCK must be timezone-aware")
    return parsed


def execution_clearance(review: Mapping[str, Any]) -> dict[str, Any]:
    """What an execution host may see: the clearance, not trial rows."""
    return {key: review[key] for key in _CLEARANCE_FIELDS if key in review}


def _set(env: Mapping[str, str], name: str) -> bool:
    return bool(env.get(name, "").strip())
