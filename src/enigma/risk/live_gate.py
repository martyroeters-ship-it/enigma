"""Live-capital gate. An agent cannot mark it passed, and this module cannot enable trading.

A human may record an authorization file outside the agent document planes. That file
does not change `registries/risk/limits.yaml`. `RiskLimits.load` still raises if
`live_trading_enabled` is true. M6 acceptance is a human act this code does not perform.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from enigma.core.errors import PermissionDeniedError
from enigma.schemas.common import AgentRole, SystemActor

_ACTOR = AgentRole | SystemActor


def mark_live_gate_passed(actor: _ACTOR) -> None:
    """Always refuses. There is no actor, including a human, for whom this returns."""
    del actor
    raise PermissionDeniedError("this process cannot mark the live-capital gate passed")


def record_human_authorization(
    actor: _ACTOR,
    directory: Path,
    *,
    account: str,
    venue: str,
    max_notional_usd: float,
    expires_on: date,
) -> Path:
    """Write a human authorization outside agent planes. Does not enable live trading."""
    if actor is not SystemActor.HUMAN:
        raise PermissionDeniedError("an agent cannot record live-capital authorization")
    if max_notional_usd <= 0:
        raise ValueError("max_notional_usd must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "live-capital-authorization.json"
    path.write_text(
        json.dumps(
            {
                "actor": SystemActor.HUMAN.value,
                "account": account,
                "venue": venue,
                "max_notional_usd": max_notional_usd,
                "expires_on": expires_on.isoformat(),
                "live_trading_enabled": False,
                "note": "This file does not enable live trading.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path
