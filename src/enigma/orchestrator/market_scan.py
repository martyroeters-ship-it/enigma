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
from enigma.core.paths import StorageRoots, resolve_uri
from enigma.research.scan import Anomaly, scan_rows
from enigma.schemas.common import AgentRole, Instrument
from enigma.schemas.market import MarketObservation, MarketScanReport
from enigma.schemas.research import DatasetVersion, ResearchObjective
from enigma.store import Stores

_FLAG = "market-scan:"


_PANEL_FIELDS = ("funding_rate", "basis", "perp_quote_volume")


def _dataset_file(uri: str, roots: StorageRoots | None) -> Path | None:
    path = Path(uri)
    if path.is_file():
        return path
    if roots is None or not uri.startswith("inst:"):
        return None
    return resolve_uri(uri, roots)


def _rows_from_datasets(
    stores: Stores,
    by_key: dict[tuple[str, str], list[dict[str, object]]],
    cutoff: datetime,
    roots: StorageRoots | None = None,
) -> None:
    needed = {"symbol", "available_at"}
    for dataset in stores.documents.list(DatasetVersion, limit=20):
        if dataset.plane != "research":
            continue
        path = _dataset_file(dataset.uri, roots)
        if path is None:
            continue
        try:
            frame = pl.read_parquet(path)
        except (OSError, pl.exceptions.PolarsError):
            continue
        fields = [name for name in _PANEL_FIELDS if name in frame.columns]
        if not fields or not needed <= set(frame.columns):
            continue
        for row in frame.select("symbol", "available_at", *fields).iter_rows(named=True):
            stamp = row["available_at"]
            if stamp is None or stamp > cutoff:
                continue
            for field in fields:
                value = row[field]
                if value is None:
                    continue
                by_key.setdefault((str(row["symbol"]), field), []).append(
                    {"available_at": stamp, field: float(value)}
                )


def _drop_shared_series(by_key: dict[tuple[str, str], list[dict[str, object]]]) -> None:
    """Two symbols with the same values are one series, not two discoveries."""
    signatures: dict[tuple[str, tuple[tuple[str, float], ...]], str] = {}
    drop: list[tuple[str, str]] = []
    for symbol, field in sorted(by_key):
        rows = by_key[(symbol, field)]
        points: list[tuple[str, float]] = []
        for row in rows:
            raw = row.get(field)
            stamp = row.get("available_at")
            if not isinstance(raw, int | float) or stamp is None:
                continue
            points.append((str(stamp), float(raw)))
        signature = tuple(sorted(points))
        owner = signatures.setdefault((field, signature), symbol)
        if owner != symbol:
            drop.append((symbol, field))
    for key in drop:
        del by_key[key]


def _latest_anomalies(
    ranked: list[Anomaly], by_key: dict[tuple[str, str], list[dict[str, object]]]
) -> list[Anomaly]:
    """A historical spike is not a new anomaly the next time the window is scanned."""
    fresh: list[Anomaly] = []
    for item in ranked:
        rows = by_key.get((item.symbol, item.field), [])
        stamps: list[datetime] = []
        for row in rows:
            stamp = row.get("available_at")
            if isinstance(stamp, datetime):
                stamps.append(stamp)
        if not stamps or item.available_at != max(stamps):
            continue
        fresh.append(item)
    return fresh


def run_market_scan(
    stores: Stores,
    now: datetime,
    *,
    interval_hours: float,
    roots: StorageRoots | None = None,
) -> bool:
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
    _rows_from_datasets(stores, by_key, now, roots)
    _drop_shared_series(by_key)
    if not by_key:
        return False
    ranked: list[Anomaly] = []
    for (symbol, field), rows in sorted(by_key.items()):
        ranked.extend(scan_rows(rows, symbol=symbol, field=field, cutoff=now))
    ranked = _latest_anomalies(ranked, by_key)
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
