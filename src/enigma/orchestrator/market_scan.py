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
from enigma.data.panel import PANEL_COLUMNS
from enigma.research.scan import Anomaly, scan_rows
from enigma.schemas.common import AgentRole, Instrument
from enigma.schemas.market import MarketObservation, MarketScanReport
from enigma.schemas.research import DatasetVersion, ResearchObjective
from enigma.store import Stores

_FLAG = "market-scan:"


def _parquet(dataset: DatasetVersion, roots: StorageRoots | None) -> Path | None:
    direct = Path(dataset.uri)
    if direct.is_file():
        return direct
    if roots is None:
        return None
    found = resolve_uri(dataset.uri, roots)
    return found if found is not None and found.is_file() else None


def _rows_from_datasets(
    stores: Stores,
    by_key: dict[tuple[str, str], list[dict[str, object]]],
    cutoff: datetime,
    roots: StorageRoots | None,
) -> None:
    fields = tuple(
        name for name in ("funding_rate", "basis", "perp_quote_volume") if name in PANEL_COLUMNS
    )
    for dataset in stores.documents.list(DatasetVersion, limit=20):
        if dataset.plane != "research":
            continue
        path = _parquet(dataset, roots)
        if path is None:
            continue
        try:
            frame = pl.read_parquet(path)
        except (OSError, pl.exceptions.PolarsError):
            continue
        present = [
            field for field in fields if field in frame.columns and "symbol" in frame.columns
        ]
        if "available_at" not in frame.columns or not present:
            continue
        for row in frame.select("symbol", "available_at", *present).iter_rows(named=True):
            stamp = row["available_at"]
            if stamp is None or stamp > cutoff:
                continue
            for field in present:
                value = row[field]
                if value is None:
                    continue
                by_key.setdefault((str(row["symbol"]), field), []).append(
                    {"available_at": stamp, field: float(value)}
                )


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
    if not by_key:
        return False
    notes: list[str] = []
    by_key, shared = _drop_shared_series(by_key)
    if shared:
        notes.append("identical series across instruments: " + ", ".join(shared))
    ranked: list[Anomaly] = []
    latest: dict[tuple[str, str], datetime] = {}
    for (symbol, field), rows in sorted(by_key.items()):
        stamps = [row["available_at"] for row in rows if isinstance(row["available_at"], datetime)]
        if stamps:
            latest[(symbol, field)] = max(stamps)
        ranked.extend(scan_rows(rows, symbol=symbol, field=field, cutoff=now))
    ranked.sort(key=lambda item: (-abs(item.zscore), item.available_at.isoformat(), item.symbol))
    already = _reported(stores)
    fresh = [
        item
        for item in ranked
        if latest.get((item.symbol, item.field)) == item.available_at
        and (item.symbol, item.field, item.available_at.isoformat()) not in already
    ]
    historical = len(ranked) - len(fresh)
    if historical:
        notes.append(f"historical anomalies not treated as new: {historical}")
    top = fresh[0] if fresh else None
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
            top_available_at=top.available_at if top else None,
            note="; ".join(notes),
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


def _drop_shared_series(
    by_key: dict[tuple[str, str], list[dict[str, object]]],
) -> tuple[dict[tuple[str, str], list[dict[str, object]]], list[str]]:
    fingerprints: dict[tuple[str, tuple[tuple[str, float], ...]], list[str]] = {}
    for (symbol, field), rows in by_key.items():
        points: list[tuple[str, float]] = []
        for row in rows:
            raw = row.get(field)
            if isinstance(raw, int | float):
                points.append((str(row["available_at"]), round(float(raw), 10)))
        series = tuple(sorted(points))
        fingerprints.setdefault((field, series), []).append(symbol)
    drop: set[tuple[str, str]] = set()
    notes: list[str] = []
    for (field, _series), symbols in fingerprints.items():
        if len(symbols) < 2 or not _series:
            continue
        notes.append(f"{field}:{'='.join(sorted(symbols))}")
        for symbol in sorted(symbols)[1:]:
            drop.add((symbol, field))
    return {key: rows for key, rows in by_key.items() if key not in drop}, notes


def _reported(stores: Stores) -> set[tuple[str, str, str]]:
    found: set[tuple[str, str, str]] = set()
    for report in stores.documents.list(MarketScanReport, limit=200):
        if report.top_symbol and report.top_field and report.top_available_at is not None:
            found.add((report.top_symbol, report.top_field, report.top_available_at.isoformat()))
    return found
