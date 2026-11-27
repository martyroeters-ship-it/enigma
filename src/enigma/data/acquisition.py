"""Fulfill a data request from a declared observable, or record that it is unmet.

The observable is a field on the request. The description is not scanned for phrases.
No provider means no rows. Another column is not substituted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from enigma.core.hashing import sha256_hex
from enigma.data.catalog import DataCatalog
from enigma.schemas.common import AgentRole
from enigma.schemas.inquiry import AcquisitionRecord
from enigma.schemas.quant import DatasetAcquisitionRequest
from enigma.schemas.research import DatasetVersion
from enigma.store import Stores

_REG = Path(__file__).resolve().parents[3] / "registries" / "data" / "public_providers.yaml"


@dataclass(frozen=True)
class Provider:
    id: str
    license: str
    reliability: str
    history_start: str
    spend_usd: float
    observables: frozenset[str]
    adapter: str


def load_providers(path: Path | None = None) -> tuple[Provider, ...]:
    raw = yaml.safe_load((path or _REG).read_text(encoding="utf-8")) or {}
    out: list[Provider] = []
    for item in raw.get("providers", []):
        out.append(
            Provider(
                id=str(item["id"]),
                license=str(item.get("license", "")),
                reliability=str(item.get("reliability", "")),
                history_start=str(item.get("history_start", "")),
                spend_usd=float(item.get("spend_usd", 0)),
                observables=frozenset(str(name) for name in item.get("observables", [])),
                adapter=str(item.get("adapter", "")),
            )
        )
    return tuple(out)


def assess(observable: str, providers: tuple[Provider, ...] | None = None) -> str:
    """A short assessment. Empty observable is unspecified, not guessed."""
    if not observable.strip():
        return "unspecified: the request did not name an observable"
    chosen = providers if providers is not None else load_providers()
    matches = [p for p in chosen if observable in p.observables and p.spend_usd <= 0]
    if not matches:
        return f"unmet: no zero-cost public provider lists {observable}"
    provider = matches[0]
    return (
        f"provider {provider.id} lists {observable} from {provider.history_start}. "
        f"License: {provider.license} Reliability: {provider.reliability}"
    )


def fulfill(
    stores: Stores,
    request: DatasetAcquisitionRequest,
    catalog: DataCatalog | None,
    *,
    symbols: tuple[str, ...],
    start: datetime,
    end: datetime,
    frequency: str,
    now: datetime,
) -> AcquisitionRecord:
    observable = request.columns[0] if request.columns else ""
    providers = load_providers()
    matches = [p for p in providers if observable in p.observables and p.spend_usd <= 0]
    record_id = "acq_" + sha256_hex(request.id)[:26]
    if not observable:
        return _store(
            stores,
            AcquisitionRecord(
                id=record_id,
                request_id=request.id,
                observable="",
                status="unspecified",
                assessment=assess(""),
                created_at=now,
            ),
        )
    if not matches or matches[0].adapter != "perp_funding_panel":
        return _store(
            stores,
            AcquisitionRecord(
                id=record_id,
                request_id=request.id,
                observable=observable,
                status="unmet",
                assessment=assess(observable, providers),
                created_at=now,
            ),
        )
    if catalog is None:
        return _store(
            stores,
            AcquisitionRecord(
                id=record_id,
                request_id=request.id,
                observable=observable,
                status="unmet",
                provider_id=matches[0].id,
                assessment=(
                    "adapter is implemented but no catalog was supplied; no rows were invented"
                ),
                created_at=now,
            ),
        )
    dataset = catalog.acquire_perp_funding_panel(list(symbols), start, end, frequency)
    if observable not in dataset.columns:
        return _store(
            stores,
            AcquisitionRecord(
                id=record_id,
                request_id=request.id,
                observable=observable,
                status="unmet",
                provider_id=matches[0].id,
                assessment=(
                    f"acquired panel does not contain {observable}; the panel was not relabeled"
                ),
                created_at=now,
            ),
        )
    return _store(
        stores,
        AcquisitionRecord(
            id=record_id,
            request_id=request.id,
            observable=observable,
            status="registered",
            provider_id=matches[0].id,
            dataset_version_id=dataset.id,
            content_hash=dataset.content_hash,
            assessment=assess(observable, providers) + f" PIT certified={dataset.pit_certified}.",
            created_at=now,
        ),
    )


def settle_open_requests(stores: Stores, dataset: DatasetVersion, now: datetime) -> tuple[str, ...]:
    """Attach open requests to a dataset that was already acquired. Do not download again."""
    providers = load_providers()
    settled: list[str] = []
    for request in stores.documents.list(DatasetAcquisitionRequest, limit=500):
        record_id = "acq_" + sha256_hex(request.id)[:26]
        if stores.documents.get(AcquisitionRecord, record_id) is not None:
            continue
        observable = request.columns[0] if request.columns else ""
        if observable and observable in dataset.columns:
            record = AcquisitionRecord(
                id=record_id,
                request_id=request.id,
                observable=observable,
                status="registered",
                provider_id="binance_public_archive",
                dataset_version_id=dataset.id,
                content_hash=dataset.content_hash,
                assessment=(
                    assess(observable, providers) + f" PIT certified={dataset.pit_certified}."
                ),
                created_at=now,
            )
        else:
            record = AcquisitionRecord(
                id=record_id,
                request_id=request.id,
                observable=observable,
                status="unmet" if observable else "unspecified",
                assessment=assess(observable, providers),
                created_at=now,
            )
        _store(stores, record)
        settled.append(record.id)
    return tuple(settled)


def _store(stores: Stores, record: AcquisitionRecord) -> AcquisitionRecord:
    if stores.documents.get(AcquisitionRecord, record.id) is None:
        stores.documents.put(record, recorded_by=AgentRole.DATA_ENGINEERING.value)
    return record
