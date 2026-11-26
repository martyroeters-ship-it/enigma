"""Deterministic risk service: decides clearance or veto from validation output and limits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from enigma.core.clock import Clock
from enigma.core.hashing import sha256_of
from enigma.schemas.common import SystemActor
from enigma.schemas.cycle import RiskDecision, RiskReview
from enigma.schemas.research import Experiment, StrategyPackage
from enigma.schemas.validation import GateOutcome, RedactedValidationSummary
from enigma.store import Stores


@dataclass(frozen=True)
class RiskLimits:
    version: str
    paper_trading_enabled: bool
    max_gross_leverage: float
    max_sleeve_notional_usd: float
    max_drawdown_research: float
    min_holdout_net_sharpe: float

    @classmethod
    def load(cls, path: Path) -> RiskLimits:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw.get("live_trading_enabled"):
            raise ValueError("live trading is constitutionally prohibited")
        return cls(
            version=str(raw["version"]),
            paper_trading_enabled=bool(raw["paper_trading_enabled"]),
            max_gross_leverage=float(raw["max_gross_leverage"]),
            max_sleeve_notional_usd=float(raw["max_sleeve_notional_usd"]),
            max_drawdown_research=float(raw["max_drawdown_research"]),
            min_holdout_net_sharpe=float(raw["min_holdout_net_sharpe"]),
        )


class RiskService:
    def __init__(self, stores: Stores, limits: RiskLimits, clock: Clock) -> None:
        self.stores = stores
        self.limits = limits
        self.clock = clock

    def evaluate(self, package_id: str, summary_id: str) -> dict[str, Any]:
        docs = self.stores.documents
        package = docs.require(StrategyPackage, package_id)
        summary = docs.require(RedactedValidationSummary, summary_id)
        if summary.strategy_package_id != package_id:
            raise ValueError("validation summary does not belong to this package")
        experiments = docs.list(Experiment, where={"strategy_package_id": package_id}, limit=50)
        drawdown = max((e.metrics.get("max_drawdown", 0.0) for e in experiments), default=0.0)
        checks = {
            "validation_passed": summary.overall is GateOutcome.PASS,
            "paper_trading_enabled": self.limits.paper_trading_enabled,
            "capacity_covers_sleeve": package.cost_model.capacity_notional_usd
            >= self.limits.max_sleeve_notional_usd,
            "research_drawdown_within_limit": drawdown <= self.limits.max_drawdown_research,
        }
        reasons = [
            code
            for code, ok in (
                ("validation_not_passed", checks["validation_passed"]),
                ("paper_trading_disabled", checks["paper_trading_enabled"]),
                ("capacity_below_sleeve", checks["capacity_covers_sleeve"]),
                ("research_drawdown_exceeds_limit", checks["research_drawdown_within_limit"]),
            )
            if not ok
        ]
        decision = RiskDecision.VETO if reasons else RiskDecision.CLEARANCE
        review = RiskReview(
            id="rsk_" + sha256_of([package_id, summary_id, self.limits.version])[7:31],
            strategy_package_id=package_id,
            decision=decision,
            reason_codes=tuple(reasons),
            checks=checks,
            limits_version=self.limits.version,
            created_by=SystemActor.RISK_SERVICE,
            created_at=self.clock.now(),
        )
        docs.put(review, recorded_by=SystemActor.RISK_SERVICE.value)
        self.stores.audit.record(
            SystemActor.RISK_SERVICE.value,
            "risk.decision",
            package_id,
            {"decision": decision.value, "reasons": reasons, "review_id": review.id},
        )
        return {
            "review_id": review.id,
            "decision": decision.value,
            "reason_codes": reasons,
            "checks": checks,
        }
