"""One deterministic market scan per schedule window.

Reads stored observations and research-plane parquet panels. When an anomaly ranks,
it records a research objective. The daemon may open a cycle for that objective.
The scan does not open a paper session or place an order.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from enigma.core.hashing import sha256_hex
from enigma.research.scan import Anomaly, scan_rows
from enigma.schemas.common import AgentRole, Instrument
from enigma.schemas.market import MarketObservation, MarketScanReport
from enigma.schemas.research import DatasetVersion, ResearchObjective
from enigma.store import Stores

_FLAG = "market-scan:"


def _rows_from_datasets(
    stores: Stores, by_key: dict[tuple[str, str], list[dict[str, object]]], cutoff: datetime
) -> None:
    needed = {"symbol", "available_at", "funding_rate"}
    for dataset in stores.documents.list(DatasetVersion, limit=20):
        if dataset.plane != "research":
            continue
        path = Path(dataset.uri)
        if not path.is_file():
            continue
        try:
            frame = pl.read_parquet(path)
        except (OSError, pl.exceptions.PolarsError):
            continue
        if not needed <= set(frame.columns):
            continue
        for row in frame.select("symbol", "available_at", "funding_rate").iter_rows(named=True):
            stamp = row["available_at"]
            rate = row["funding_rate"]
            if stamp is None or rate is None or stamp > cutoff:
                continue
            by_key.setdefault((str(row["symbol"]), "funding_rate"), []).append(
                {"available_at": stamp, "funding_rate": float(rate)}
            )


def run_market_scan(stores: Stores, now: datetime, *, interval_hours: float) -> bool:
    if interval_hours <= 0:
        return False
    window = str(int(now.timestamp() // (interval_hours * 3600)))
    flag = _FLAG + window
    if stores.flags.get(flag) is not None:
        return False
    observations = stores.documents.list(MarketObservation, limit=5000, newest_first=False)
    by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
    for obs in observations:
        if obs.available_at > now:
            continue
        by_key.setdefault((obs.symbol, obs.field), []).append(
            {"available_at": obs.available_at, obs.field: obs.value}
        )
    _rows_from_datasets(stores, by_key, now)
    ranked: list[Anomaly] = []
    for (symbol, field), rows in sorted(by_key.items()):
        ranked.extend(scan_rows(rows, symbol=symbol, field=field, cutoff=now))
    ranked.sort(key=lambda item: (-abs(item.zscore), item.available_at.isoformat(), item.symbol))
    top = ranked[0] if ranked else None
    objective_id = None
    if top is not None and top.suggested_mechanism:
        objective_id = "obj_" + sha256_hex(f"{window}|{top.symbol}|{top.field}")[:26].upper()
        if stores.documents.get(ResearchObjective, objective_id) is None:
            base = top.symbol[:-4] if top.symbol.endswith("USDT") else top.symbol
            stores.documents.put(
                ResearchObjective(
                    id=objective_id,
                    statement=(
                        f"Observation {top.symbol} {top.field} z={top.zscore:.3f} "
                        f"suggests {top.suggested_mechanism}. This is not a validation result."
                    ),
                    instruments=(
                        Instrument(
                            symbol=top.symbol,
                            base=base,
                            quote="USDT",
                            kind="perpetual",
                            venue="observed",
                        ),
                    ),
                    topics=("market-scan", top.field, top.suggested_mechanism),
                    created_by=AgentRole.CIO,
                    created_at=now,
                ),
                recorded_by=AgentRole.CIO.value,
            )
    report_id = "scn_" + sha256_hex(window)[:26].upper()
    stores.documents.put(
        MarketScanReport(
            id=report_id,
            window_key=window,
            cutoff=now,
            observation_count=sum(len(rows) for rows in by_key.values()),
            anomaly_count=len(ranked),
            top_symbol=top.symbol if top else "",
            top_field=top.field if top else "",
            top_zscore=top.zscore if top else None,
            suggested_mechanism=top.suggested_mechanism if top else "",
            objective_id=objective_id,
            created_at=now,
        ),
        recorded_by="system",
    )
    stores.flags.set(flag, report_id, "system")
    stores.audit.record(
        "system",
        "market.scan",
        report_id,
        {"anomalies": len(ranked), "objective_id": objective_id},
    )
    return True
