"""Register a scientific case only when its evidence is real.

A model's paragraph is the interpretation field. It is not supporting evidence.
Registration does not consult a mechanism library or a phrase list.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from enigma.core.hashing import sha256_hex
from enigma.research.program import program_problem
from enigma.schemas.common import AgentRole, Instrument
from enigma.schemas.inquiry import ScientificCase
from enigma.schemas.knowledge import EvidenceItem, MarketMechanism
from enigma.schemas.quant import DatasetAcquisitionRequest
from enigma.schemas.research import ResearchObjective
from enigma.store import Stores

_DISCOVERED = "discovered"


@dataclass(frozen=True)
class Registration:
    ok: bool
    reason: str
    mechanism_id: str = ""
    objective_id: str = ""
    acquisition_ids: tuple[str, ...] = ()


def assess_evidence(case: ScientificCase, evidence: dict[str, EvidenceItem]) -> str:
    """Empty when the citations are usable empirical excerpts."""
    problem = program_problem(case.program)
    if problem:
        return problem
    sources: set[str] = set()
    for eid in case.supporting_evidence_ids:
        item = evidence.get(eid)
        if item is None:
            return f"supporting evidence {eid} is not in the store"
        if item.is_ai_interpretation:
            return f"supporting evidence {eid} is a model interpretation, not an excerpt"
        if not item.excerpt.strip():
            return f"supporting evidence {eid} has no excerpt"
        sources.add(item.source_id)
    if len(sources) < 2:
        return "supporting evidence must come from at least two sources"
    for eid in case.contradictory_evidence_ids:
        if not eid:
            continue
        item = evidence.get(eid)
        if item is None:
            return f"contradictory evidence {eid} is not in the store"
        if item.is_ai_interpretation:
            return f"contradictory evidence {eid} is a model interpretation, not an excerpt"
    return ""


def register_case(
    stores: Stores,
    case: ScientificCase,
    *,
    held_observables: set[str],
    now: datetime,
) -> Registration:
    """Persist the case, or refuse it. Missing observables become acquisition requests."""
    evidence = {item.id: item for item in stores.documents.list(EvidenceItem, limit=10_000)}
    # Evidence supplied with the case but not yet stored is not accepted.
    reason = assess_evidence(case, evidence)
    if reason:
        return Registration(False, reason)
    if stores.documents.get(ScientificCase, case.id) is None:
        stores.documents.put(case, recorded_by=AgentRole.QUANT_RESEARCH.value)
    mechanism_id = "mech_" + sha256_hex(case.id)[:26]
    if stores.documents.get(MarketMechanism, mechanism_id) is None:
        stores.documents.put(
            MarketMechanism(
                id=mechanism_id,
                name=case.mechanism[:200],
                description=case.persistence,
                evidence_ids=case.supporting_evidence_ids,
                created_by=AgentRole.QUANT_RESEARCH,
                created_at=now,
            ),
            recorded_by=AgentRole.QUANT_RESEARCH.value,
        )
    program_inputs = {name for comp in case.program.computations for name in comp.inputs}
    if case.program.condition is not None:
        program_inputs.update(case.program.condition.inputs)
    acquisitions: list[str] = []
    for need in case.data_needs:
        if need.observable in held_observables:
            continue
        request_id = "dar_" + sha256_hex(f"{case.id}|{need.observable}")[:26]
        if stores.documents.get(DatasetAcquisitionRequest, request_id) is None:
            stores.documents.put(
                DatasetAcquisitionRequest(
                    id=request_id,
                    requested_by=AgentRole.QUANT_RESEARCH,
                    description=need.observable,
                    columns=(need.observable,),
                    rationale=need.why,
                    source_draft_title=case.mechanism[:200],
                    generation_id=case.id,
                    created_at=now,
                ),
                recorded_by=AgentRole.DATA_ENGINEERING.value,
            )
        acquisitions.append(request_id)
    missing_program_inputs = program_inputs - held_observables
    objective_id = "obj_" + sha256_hex(f"case|{case.id}")[:26].upper()
    if stores.documents.get(ResearchObjective, objective_id) is None and not missing_program_inputs:
        symbols = case.program.symbols
        stores.documents.put(
            ResearchObjective(
                id=objective_id,
                statement=(
                    f"{case.mechanism} Prediction: {case.predictions[0].statement} "
                    "The interpretation is not evidence."
                ),
                instruments=tuple(_instrument(symbol) for symbol in symbols),
                topics=(
                    case.id,
                    case.program.design_class,
                    case.program.sampling_frequency,
                    _DISCOVERED,
                ),
                created_by=AgentRole.QUANT_RESEARCH,
                created_at=now,
            ),
            recorded_by=AgentRole.QUANT_RESEARCH.value,
        )
    stores.audit.record(
        AgentRole.QUANT_RESEARCH.value,
        "research.case_registered",
        case.id,
        {
            "mechanism_id": mechanism_id,
            "objective_id": "" if missing_program_inputs else objective_id,
            "acquisitions": len(acquisitions),
        },
    )
    if missing_program_inputs:
        return Registration(
            True,
            "registered, waiting on data that was not substituted",
            mechanism_id,
            "",
            tuple(acquisitions),
        )
    return Registration(True, "registered", mechanism_id, objective_id, ())


def next_discovered_objective(stores: Stores) -> ResearchObjective | None:
    used = {
        str(cp["state"].get("objective_id"))
        for cp in stores.checkpoints.list("research_cycle")
        if cp["state"].get("objective_id")
    }
    for objective in stores.documents.list(ResearchObjective, limit=500):
        if _DISCOVERED not in objective.topics or objective.id in used:
            continue
        return objective
    return None


def _instrument(symbol: str) -> Instrument:
    return Instrument(
        symbol=symbol,
        base=symbol.removesuffix("USDT"),
        quote="USDT",
        kind="perpetual",
        venue="binance_usdm",
    )
