"""Register a scientific case only when its evidence is real.

A model's paragraph is the interpretation field. It is not supporting evidence.
Registration does not consult a mechanism library or a phrase list.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from enigma.core.hashing import sha256_hex
from enigma.data.acquisition import load_providers
from enigma.research.hypothesis import FALSIFICATION
from enigma.research.program import compile_program, program_columns, program_problem
from enigma.schemas.common import AgentRole, Instrument
from enigma.schemas.inquiry import ScientificCase
from enigma.schemas.knowledge import EvidenceItem, MarketMechanism
from enigma.schemas.quant import DatasetAcquisitionRequest
from enigma.schemas.research import Hypothesis, ResearchObjective
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


def case_for_objective(
    objective: ResearchObjective, get: Callable[[type[Any], str], Any]
) -> ScientificCase | None:
    """The case named by a discovered objective, if it is actually stored."""
    if _DISCOVERED not in objective.topics:
        return None
    for topic in objective.topics:
        case = get(ScientificCase, topic)
        if isinstance(case, ScientificCase):
            return case
    return None


def hypothesis_from_case(
    case: ScientificCase,
    objective: ResearchObjective,
    *,
    cycle_id: str,
    dataset_version_id: str,
    now: datetime,
) -> Hypothesis:
    """The pre-registered case is the hypothesis. The seed library is not consulted."""
    prediction = case.predictions[0]
    criteria = list(FALSIFICATION)
    if prediction.discriminates.strip():
        criteria.append(f"Does not discriminate: {prediction.discriminates.strip()}")
    features = program_columns(case.program)
    if not features:
        raise ValueError("the research program names no columns")
    digest = sha256_hex(f"{cycle_id}|{case.id}|{case.program.sampling_frequency}")
    return Hypothesis(
        id="hyp_" + digest[:26].upper(),
        objective_id=objective.id,
        family_id=f"fam_{case.id}",
        title=case.mechanism[:240],
        mechanism=case.persistence,
        prediction=prediction.statement,
        formal_statement=prediction.statement,
        falsification_criteria=tuple(criteria),
        instruments=objective.instruments,
        horizon=case.program.sampling_frequency,
        features=features,
        search_space={},
        trial_budget=1,
        cost_model_version="cm-binance-usdm-v1",
        dataset_version_ids=(dataset_version_id,),
        evidence_ids=case.supporting_evidence_ids,
        contradictory_evidence_ids=tuple(eid for eid in case.contradictory_evidence_ids if eid),
        uncertainty_notes=(
            "Registered scientific case. The interpretation field is not evidence. "
            "The seed library was not scored."
        ),
        generation_method=f"research-program:{case.id}",
        methodology_version="research-program-1",
        created_by=AgentRole.QUANT_RESEARCH,
        created_at=now,
    )


def compiled_case_strategy(
    hypothesis: Hypothesis, get: Callable[[type[Any], str], Any]
) -> tuple[str, str, tuple[str, ...]] | None:
    """Source, entrypoint, and columns for a registered program. None for other hypotheses."""
    prefix = "research-program:"
    if not hypothesis.generation_method.startswith(prefix):
        return None
    case_id = hypothesis.generation_method.removeprefix(prefix)
    case = get(ScientificCase, case_id)
    if not isinstance(case, ScientificCase):
        raise ValueError(f"research program {case_id} is not in the store")
    return compile_program(case.program), "ResearchProgramStrategy", program_columns(case.program)


def registered_case_retrieval(
    case: ScientificCase, get: Callable[[type[Any], str], Any]
) -> dict[str, Any] | None:
    """Evidence already stored with the case. None when a citation is missing or is model text."""
    evidence_ids = list(case.supporting_evidence_ids) + [
        eid for eid in case.contradictory_evidence_ids if eid
    ]
    items: list[EvidenceItem] = []
    for evidence_id in evidence_ids:
        item = get(EvidenceItem, evidence_id)
        if not isinstance(item, EvidenceItem) or item.is_ai_interpretation:
            return None
        if not item.excerpt.strip():
            return None
        items.append(item)
    source_ids = sorted({item.source_id for item in items})
    if len(source_ids) < 2:
        return None
    return {
        "source_ids": source_ids,
        "evidence_ids": evidence_ids,
        "queries": ["registered_case"],
        "http_requests": 0,
        "case_id": case.id,
    }


def case_from_model(stores: Stores, body: dict[str, Any], now: datetime) -> str:
    """Register a model object only when it is a complete case citing stored evidence."""
    if not body:
        return "model output is not a scientific case"
    payload = dict(body)
    digest = sha256_hex(json.dumps(body, sort_keys=True, default=str))
    payload.setdefault("id", "case_" + digest[:26])
    payload.setdefault("created_at", now)
    if not str(payload.get("interpretation") or "").strip():
        payload["interpretation"] = "Model text is not evidence."
    try:
        case = ScientificCase.model_validate(payload)
    except ValidationError as exc:
        return f"model output is not a scientific case ({exc.error_count()} invalid fields)"
    listed = {
        name
        for provider in load_providers()
        if provider.spend_usd <= 0
        for name in provider.observables
    }
    registration = register_case(stores, case, held_observables=listed, now=now)
    if registration.ok:
        return f"{registration.reason}; objective {registration.objective_id or 'waiting on data'}"
    return registration.reason


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
