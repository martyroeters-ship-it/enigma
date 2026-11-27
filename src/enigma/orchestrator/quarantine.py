"""Keep one inaccessible dataset from stopping the rest of the institution.

An objective that needs a missing artifact is recorded and skipped. Validation is
not run for it, and a sealed holdout is not opened with a substitute file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from enigma.core.config import Settings
from enigma.core.errors import IntegrityError
from enigma.orchestrator.cio import Candidate
from enigma.schemas.cycle import NextExperimentProposal
from enigma.schemas.research import ResearchObjective
from enigma.store import Stores
from enigma.store.relocate import HomeReport, inspect_home

FLAG = "artifact.quarantine"


@dataclass
class Quarantine:
    datasets: dict[str, str] = field(default_factory=dict)
    symbols: dict[str, str] = field(default_factory=dict)
    candidates: dict[str, str] = field(default_factory=dict)

    def reason(self, candidate: Candidate, symbols: tuple[str, ...]) -> str:
        if candidate.id in self.candidates:
            return self.candidates[candidate.id]
        for dataset_id in candidate.required_datasets:
            if dataset_id in self.datasets:
                return self.datasets[dataset_id]
        for symbol in symbols:
            if symbol in self.symbols:
                return self.symbols[symbol]
        return ""


def load_quarantine(stores: Stores) -> Quarantine:
    raw = stores.flags.get(FLAG)
    if not raw:
        return Quarantine()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return Quarantine()
    if not isinstance(body, dict):
        return Quarantine()
    return Quarantine(
        datasets=_strings(body.get("datasets")),
        symbols=_strings(body.get("symbols")),
        candidates=_strings(body.get("candidates")),
    )


def save_quarantine(stores: Stores, quarantine: Quarantine) -> None:
    payload = {
        "datasets": quarantine.datasets,
        "symbols": quarantine.symbols,
        "candidates": quarantine.candidates,
    }
    stores.flags.set(FLAG, json.dumps(payload, sort_keys=True), "system")


def apply_preflight_quarantine(stores: Stores, report: HomeReport) -> Quarantine:
    """Replace dataset and symbol blocks from the latest preflight. Keeps no stale symbol."""
    quarantine = Quarantine()
    for issue in report.unresolved:
        detail = issue.detail or issue.uri
        if issue.dataset_id:
            quarantine.datasets[issue.dataset_id] = detail
        for symbol in issue.symbols:
            quarantine.symbols[symbol] = detail
    save_quarantine(stores, quarantine)
    return quarantine


def quarantine_candidate(
    stores: Stores, candidate: Candidate, reason: str, symbols: tuple[str, ...] = ()
) -> None:
    """Remember one objective that could not be opened. Other objectives keep running."""
    current = load_quarantine(stores)
    current.candidates[candidate.id] = reason[:500]
    for symbol in symbols:
        current.symbols.setdefault(symbol, reason[:500])
    save_quarantine(stores, current)
    stores.audit.record(
        "system",
        "objective.quarantined",
        candidate.id,
        {"reason": reason[:500], "symbols": list(symbols)},
    )


def candidate_symbols(stores: Stores, candidate: Candidate) -> tuple[str, ...]:
    if candidate.kind == "opportunity" and len(candidate.refs) == 3:
        return (candidate.refs[1],)
    if candidate.kind == "market_scan" and candidate.refs:
        objective = stores.documents.get(ResearchObjective, candidate.refs[0])
        if objective is not None:
            return tuple(sorted({item.symbol for item in objective.instruments}))
    if candidate.proposal_id:
        proposal = stores.documents.get(NextExperimentProposal, candidate.proposal_id)
        if proposal is None:
            return ()
        parent = stores.checkpoints.load(proposal.parent_cycle_id)
        if parent is None:
            return ()
        objective_id = str(parent[1].get("objective_id") or "")
        objective = stores.documents.get(ResearchObjective, objective_id)
        if objective is None:
            return ()
        return tuple(sorted({item.symbol for item in objective.instruments}))
    return ()


def prepare_runtime(settings: Settings, stores: Stores) -> HomeReport:
    """Startup check. Hash mismatches refuse to start. Missing files are quarantined."""
    report = inspect_home(settings, stores)
    apply_preflight_quarantine(stores, report)
    if report.refused:
        raise IntegrityError(report.text())
    return report


def _strings(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}
