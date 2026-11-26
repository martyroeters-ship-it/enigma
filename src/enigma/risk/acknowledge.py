"""Human risk acknowledge gate required before paper capital (M2)."""

from __future__ import annotations

from enigma.core.clock import Clock
from enigma.core.hashing import sha256_hex
from enigma.schemas.common import SystemActor
from enigma.schemas.cycle import RiskDecision, RiskReview
from enigma.schemas.execution import HumanRiskAcknowledge
from enigma.store import Stores


def acknowledge_risk(
    stores: Stores,
    clock: Clock,
    *,
    risk_review_id: str,
    acknowledged_by: str,
    note: str,
) -> HumanRiskAcknowledge:
    review = stores.documents.require(RiskReview, risk_review_id)
    if review.decision is not RiskDecision.CLEARANCE:
        raise ValueError("only CLEARANCE reviews can be acknowledged for paper")
    ack = HumanRiskAcknowledge(
        id="hrack_" + sha256_hex(f"{risk_review_id}|{acknowledged_by}|{note}")[:26],
        risk_review_id=review.id,
        strategy_package_id=review.strategy_package_id,
        acknowledged_by=acknowledged_by,
        note=note,
        created_at=clock.now(),
    )
    stores.documents.put(ack, recorded_by=SystemActor.HUMAN.value)
    stores.audit.record(
        SystemActor.HUMAN.value,
        "risk.human_acknowledge",
        ack.id,
        {"risk_review_id": risk_review_id, "by": acknowledged_by},
    )
    return ack
