"""Paper degradation detector: pauses sessions when live paper metrics breach bands."""

from __future__ import annotations

from collections.abc import Sequence

from enigma.core.clock import Clock
from enigma.core.hashing import sha256_hex
from enigma.execution.paper import PaperBroker
from enigma.schemas.common import AgentRole
from enigma.schemas.execution import (
    DegradationAlert,
    DegradationKind,
    PaperFill,
    PaperSession,
)
from enigma.store import Stores

_ACTOR = AgentRole.EXECUTION_MONITORING


def rolling_equity_drawdown(equities: Sequence[float]) -> float:
    if not equities:
        return 0.0
    peak = equities[0]
    worst = 0.0
    for e in equities:
        peak = max(peak, e)
        if peak > 0:
            worst = max(worst, (peak - e) / peak)
    return worst


def estimate_fill_sharpe(pnls: Sequence[float]) -> float:
    """Crude per-fill Sharpe (not annualized). Zero-variance → 0."""
    if len(pnls) < 2:
        return 0.0
    mean = float(sum(pnls) / len(pnls))
    var = float(sum((x - mean) ** 2 for x in pnls) / (len(pnls) - 1))
    if var <= 0:
        return 0.0
    return float(mean / (var**0.5))


class DegradationDetector:
    def __init__(
        self,
        stores: Stores,
        clock: Clock,
        broker: PaperBroker,
        *,
        max_drawdown: float = 0.15,
        min_fill_sharpe: float = -0.5,
        max_avg_fee_bps: float = 20.0,
    ) -> None:
        self.stores = stores
        self.clock = clock
        self.broker = broker
        self.max_drawdown = max_drawdown
        self.min_fill_sharpe = min_fill_sharpe
        self.max_avg_fee_bps = max_avg_fee_bps

    def evaluate(
        self,
        session_id: str,
        *,
        equity_path: Sequence[float],
        fills: Sequence[PaperFill],
        fill_pnls: Sequence[float],
    ) -> list[DegradationAlert]:
        session = self.stores.documents.require(PaperSession, session_id)
        alerts: list[DegradationAlert] = []
        dd = rolling_equity_drawdown(equity_path)
        if dd > self.max_drawdown:
            alerts.append(
                self._alert(
                    session,
                    DegradationKind.DRAWDOWN,
                    dd,
                    f"paper drawdown {dd:.3f} exceeds {self.max_drawdown:.3f}",
                    {"drawdown": dd, "limit": self.max_drawdown},
                )
            )
        sharpe = estimate_fill_sharpe(fill_pnls)
        if len(fill_pnls) >= 8 and sharpe < self.min_fill_sharpe:
            alerts.append(
                self._alert(
                    session,
                    DegradationKind.SHARPE_DROP,
                    abs(sharpe - self.min_fill_sharpe),
                    f"paper fill Sharpe {sharpe:.3f} below {self.min_fill_sharpe:.3f}",
                    {"fill_sharpe": sharpe, "limit": self.min_fill_sharpe},
                )
            )
        if fills:
            avg_fee_bps = (
                10_000.0
                * sum(f.fee_usd for f in fills)
                / max(sum(f.price * f.quantity for f in fills), 1e-12)
            )
            if avg_fee_bps > self.max_avg_fee_bps:
                alerts.append(
                    self._alert(
                        session,
                        DegradationKind.FILL_COST_SPIKE,
                        avg_fee_bps - self.max_avg_fee_bps,
                        f"avg fee {avg_fee_bps:.2f} bps exceeds {self.max_avg_fee_bps:.2f}",
                        {"avg_fee_bps": avg_fee_bps, "limit": self.max_avg_fee_bps},
                    )
                )
        for alert in alerts:
            self.stores.documents.put(alert, recorded_by=_ACTOR.value)
            self.stores.audit.record(
                _ACTOR.value,
                "paper.degradation",
                alert.id,
                {"kind": alert.kind.value, "severity": alert.severity},
            )
        if alerts:
            self.broker.pause(session_id, reason=alerts[0].kind.value, actor=_ACTOR.value)
        return alerts

    def _alert(
        self,
        session: PaperSession,
        kind: DegradationKind,
        severity: float,
        message: str,
        metrics: dict[str, float],
    ) -> DegradationAlert:
        return DegradationAlert(
            id="pdgr_" + sha256_hex(f"{session.id}|{kind.value}|{message}")[:26],
            session_id=session.id,
            strategy_package_id=session.strategy_package_id,
            kind=kind,
            severity=severity,
            message=message,
            metrics=metrics,
            created_at=self.clock.now(),
        )
