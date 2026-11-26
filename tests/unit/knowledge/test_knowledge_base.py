"""KnowledgeBase: dedupe, evidence FK (AT4), retrieval, prior knowledge, provenance, promotion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from enigma.core.clock import FrozenClock
from enigma.core.errors import IntegrityError, PermissionDeniedError
from enigma.knowledge import (
    DanglingEdgeError,
    KnowledgeBase,
    OrphanEvidenceError,
    UnknownEntityError,
)
from enigma.schemas.common import Actor, AgentRole, ClaimLevel, Instrument, Plane, SystemActor
from enigma.schemas.knowledge import (
    Author,
    ClaimPromotion,
    EdgeRelation,
    EvidenceItem,
    FailureCondition,
    FailureKind,
    Lesson,
    Source,
    SourceKind,
)
from enigma.schemas.research import Experiment, ExperimentStatus, Hypothesis
from enigma.schemas.validation import GateOutcome, GateResult, ValidationReport
from enigma.store import PlaneView, Stores

NOW = datetime(2026, 11, 25, tzinfo=UTC)
H = "sha256:" + "a" * 64
ALL = frozenset(Plane)
BTC = Instrument(symbol="BTCUSDT", base="BTC", quote="USDT", kind="perpetual", venue="binance_usdm")


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(NOW, step=timedelta(seconds=1))


def kb_for(
    stores: Stores,
    clock: FrozenClock,
    readable: frozenset[Plane] = ALL,
    writable: frozenset[Plane] = ALL,
    actor: str = "system",
) -> KnowledgeBase:
    return KnowledgeBase(PlaneView(stores.documents, readable, writable, actor), clock)


@pytest.fixture
def kb(stores: Stores, clock: FrozenClock) -> KnowledgeBase:
    return kb_for(stores, clock)


def source(sid: str, provider: str = "arxiv", content: str = "a", **ids: str) -> Source:
    return Source(
        id=sid,
        kind=SourceKind.PREPRINT,
        provider=provider,
        url=f"https://example.org/{sid}",
        title=f"Funding rate study {sid}",
        abstract="We find the perpetual funding rate predicts basis returns with t = 4.2.",
        external_ids=dict(ids),
        content_hash="sha256:" + content * 64,
        retrieved_by=AgentRole.INTELLIGENCE,
        retrieved_at=NOW,
    )


def evidence(
    eid: str,
    sid: str,
    text: str = "Funding rate predicts basis returns (t = 4.2).",
    level: ClaimLevel = ClaimLevel.L0_UNVERIFIED,
    topics: tuple[str, ...] = ("funding_rate", "basis"),
) -> EvidenceItem:
    return EvidenceItem(
        id=eid,
        source_id=sid,
        excerpt=text,
        locator="abstract:chars=0-10",
        claim=text,
        topics=topics,
        extraction_method="deterministic:quant-sentence-v1",
        claim_level=level,
        created_by=AgentRole.INTELLIGENCE,
        created_at=NOW,
    )


def hypothesis(hid: str = "hyp_1", evidence_ids: tuple[str, ...] = ()) -> Hypothesis:
    return Hypothesis(
        id=hid,
        objective_id="obj_1",
        family_id="fam_1",
        title="Funding carry predicts basis",
        mechanism="Leveraged longs pay funding which mean-reverts the perpetual basis",
        prediction="High funding precedes basis compression",
        formal_statement="E[dB|F>q] < 0",
        falsification_criteria=("no significant coefficient",),
        instruments=(BTC,),
        horizon="8h",
        features=("funding_rate", "basis"),
        trial_budget=5,
        cost_model_version="cm-1",
        evidence_ids=evidence_ids,
        generation_method="template:funding-carry-v1",
        methodology_version="0.1.0",
        created_by=AgentRole.QUANT_RESEARCH,
        created_at=NOW,
    )


def experiment(
    xid: str = "exp_1", status: ExperimentStatus = ExperimentStatus.SUCCEEDED
) -> Experiment:
    return Experiment(
        id=xid,
        hypothesis_id="hyp_1",
        family_id="fam_1",
        trial_index=1,
        strategy_package_id="pkg_1",
        dataset_version_id="ds_1",
        research_window=(NOW, NOW),
        code_sha="abc",
        config_hash=H,
        seed=1,
        status=status,
        created_by=AgentRole.QUANT_RESEARCH,
    )


def report(rid: str, outcome: GateOutcome) -> ValidationReport:
    gate = GateResult(
        gate_id="G01", name="dsr", result=outcome, stat=1.0, threshold=0.5, comparator=">="
    )
    return ValidationReport(
        id=rid,
        request_id="vreq_1",
        strategy_package_id="pkg_1",
        protocol_version="0.1.0-slice",
        required_gates=("G01",),
        gates=(gate,),
        overall=outcome,
        holdout_dataset_ids=("hold_1",),
        trial_count=1,
        cost_assumptions_hash=H,
        validator_build_sha="v1",
        created_at=NOW,
    )


# ---------------------------------------------------------------- sources & dedupe
def test_add_source_dedupes_by_normalized_external_id_and_merges(kb: KnowledgeBase) -> None:
    first = kb.add_source(source("src_1", "arxiv", "a", arxiv="2506.08573v1"))
    second = kb.add_source(
        source(
            "src_2", "openalex", "b", openalex="W1", doi="https://doi.org/10.48550/arXiv.2506.08573"
        )
    )
    assert first == second == "src_1"
    merged = kb.view.require(Source, "src_1")
    assert merged.external_ids["openalex"] == "W1"
    assert merged.external_ids["providers"] == "arxiv,openalex"
    assert len(kb.view.history(Source, "src_1")) == 2  # merge is a new version, not a rewrite
    assert kb.view.get(Source, "src_2") is None


def test_add_source_dedupes_by_content_hash(kb: KnowledgeBase) -> None:
    assert kb.add_source(source("src_1", content="c")) == "src_1"
    assert kb.add_source(source("src_2", content="c")) == "src_1"
    assert kb.add_source(source("src_3", content="d")) == "src_3"


def test_duplicates_stored_without_dedupe_never_become_canonical(
    stores: Stores, clock: FrozenClock
) -> None:
    kb = kb_for(stores, clock)
    kb.add_source(source("src_1", content="a", doi="10.1234/x"))
    kb.add_source(source("src_dup", content="b", doi="10.1234/x"), dedupe=False)
    kb.link("src_dup", "src_1", EdgeRelation.DUPLICATE_OF, "dedupe:id-match")
    fresh = kb_for(stores, clock)
    assert fresh.find_source({"doi": "10.1234/X"}) == "src_1"
    assert fresh.canonical_source_id("src_dup") == "src_1"


def test_add_author_dedupes_on_strong_ids(kb: KnowledgeBase) -> None:
    kb.add_source(source("src_1", content="a"))
    kb.add_source(source("src_2", content="b"))
    a1 = kb.add_author(
        Author(id="auth_1", name="J. Kim", external_ids={"openalex": "A1"}, source_ids=("src_1",))
    )
    a2 = kb.add_author(
        Author(
            id="auth_2", name="Jaehyun Kim", external_ids={"openalex": "A1"}, source_ids=("src_2",)
        )
    )
    a3 = kb.add_author(Author(id="auth_3", name="J. Kim", external_ids={"openalex": "A9"}))
    assert a1 == a2 == "auth_1" and a3 == "auth_3"
    assert kb.view.require(Author, "auth_1").source_ids == ("src_1", "src_2")


# ---------------------------------------------------------------- AT4 evidence FK
def test_orphan_evidence_is_rejected(kb: KnowledgeBase) -> None:
    with pytest.raises(OrphanEvidenceError):
        kb.add_evidence(evidence("evid_1", "src_missing"))
    assert isinstance(OrphanEvidenceError("x"), IntegrityError)
    assert kb.view.get(EvidenceItem, "evid_1") is None


def test_evidence_with_existing_source_is_accepted_and_idempotent(kb: KnowledgeBase) -> None:
    kb.add_source(source("src_1"))
    assert kb.add_evidence(evidence("evid_1", "src_1")) == "evid_1"
    assert kb.add_evidence(evidence("evid_2", "src_1")) == "evid_1"


def test_new_evidence_must_be_l0(kb: KnowledgeBase) -> None:
    kb.add_source(source("src_1"))
    with pytest.raises(PermissionDeniedError):
        kb.add_evidence(evidence("evid_1", "src_1", level=ClaimLevel.L1_HYPOTHESIS))


def test_ai_interpretation_requires_model_id(kb: KnowledgeBase) -> None:
    kb.add_source(source("src_1"))
    item = evidence("evid_1", "src_1").model_copy(update={"is_ai_interpretation": True})
    with pytest.raises(IntegrityError):
        kb.add_evidence(item)


def test_hypothesis_must_cite_existing_evidence(kb: KnowledgeBase) -> None:
    with pytest.raises(IntegrityError):
        kb.add_hypothesis(hypothesis(evidence_ids=("evid_missing",)))


def test_edges_require_existing_endpoints(kb: KnowledgeBase) -> None:
    kb.add_source(source("src_1", content="a"))
    kb.add_source(source("src_2", content="b"))
    with pytest.raises(DanglingEdgeError):
        kb.link("src_1", "src_nope", EdgeRelation.CITES, "openalex:referenced_works")
    with pytest.raises(IntegrityError):
        kb.link("src_1", "src_1", EdgeRelation.CITES, "x")
    first = kb.link("src_1", "src_2", EdgeRelation.CITES, "openalex:referenced_works")
    assert kb.link("src_1", "src_2", EdgeRelation.CITES, "openalex:referenced_works") == first


def test_research_view_cannot_write_control_kinds(stores: Stores, clock: FrozenClock) -> None:
    kb = kb_for(
        stores,
        clock,
        readable=frozenset({Plane.RESEARCH, Plane.CONTROL}),
        writable=frozenset({Plane.RESEARCH}),
        actor="intelligence",
    )
    lesson = Lesson(
        id="lesson_1",
        statement="x",
        topics=(),
        derived_from_ids=(),
        outcome="failure",
        claim_level=ClaimLevel.L0_UNVERIFIED,
        created_by=AgentRole.META_LEARNING,
        created_at=NOW,
    )
    with pytest.raises(PermissionDeniedError):
        kb.add_lesson(lesson)


# ---------------------------------------------------------------- retrieval
def _seed_knowledge(kb: KnowledgeBase) -> None:
    kb.add_source(source("src_1"))
    kb.add_evidence(evidence("evid_1", "src_1"))
    kb.add_evidence(
        evidence(
            "evid_2",
            "src_1",
            "Volatility clustering in order book depth (3 bps).",
            topics=("volatility",),
        )
    )
    kb.add_hypothesis(hypothesis(evidence_ids=("evid_1",)))
    kb.add_lesson(
        Lesson(
            id="lesson_1",
            statement="Funding carry signals decayed after taker fees",
            topics=("funding_rate",),
            derived_from_ids=("refl_1",),
            outcome="failure",
            claim_level=ClaimLevel.L1_HYPOTHESIS,
            created_by=AgentRole.META_LEARNING,
            created_at=NOW,
        )
    )
    kb.add_failure(
        FailureCondition(
            id="fail_1",
            kind=FailureKind.COST_FRAGILITY,
            description="Funding carry edge vanished under realistic funding and fee costs",
            subject_id="pkg_1",
            created_by=SystemActor.VALIDATION_SERVICE,
            created_at=NOW,
        )
    )


def test_retrieve_bm25_with_kind_and_topic_filters(kb: KnowledgeBase) -> None:
    _seed_knowledge(kb)
    hits = kb.retrieve("funding basis")
    assert hits and hits[0].score >= hits[-1].score
    assert {h.kind for h in hits} >= {"evidence", "hypothesis", "lesson", "source"}
    only_evidence = kb.retrieve("funding", kinds=("evidence",))
    assert {h.id for h in only_evidence} == {"evid_1"}
    assert only_evidence[0].claim_level is ClaimLevel.L0_UNVERIFIED
    topical = kb.retrieve(
        "funding volatility", topics=("funding_rate",), kinds=("evidence", "lesson")
    )
    assert {h.id for h in topical} == {"evid_1", "lesson_1"}
    assert kb.retrieve("!!!") == []


def test_prior_knowledge_bundles_lessons_failures_hypotheses_evidence(kb: KnowledgeBase) -> None:
    _seed_knowledge(kb)
    kb.add_source(source("src_2", content="b"))
    kb.add_evidence(evidence("evid_3", "src_2", "Funding rate does not predict basis returns."))
    kb.link("evid_1", "evid_3", EdgeRelation.CONTRADICTS, "heuristic:polarity-v1", 0.4)
    prior = kb.prior_knowledge(["funding", "carry"])
    assert [x.id for x in prior.lessons] == ["lesson_1"]
    assert [x.id for x in prior.failure_conditions] == ["fail_1"]
    assert [x.id for x in prior.hypotheses] == ["hyp_1"]
    assert {x.id for x in prior.evidence} >= {"evid_1", "evid_3"}
    assert len(prior.contradictions) == 1 and not prior.is_empty
    assert kb.prior_knowledge(["zzzz"]).is_empty


def test_provenance_chain(kb: KnowledgeBase) -> None:
    kb.add_source(source("src_1", content="a", doi="10.1234/x"))
    kb.add_source(
        source("src_dup", provider="crossref", content="b", doi="10.1234/x"), dedupe=False
    )
    kb.link("src_dup", "src_1", EdgeRelation.DUPLICATE_OF, "dedupe:id-match")
    kb.add_author(Author(id="auth_1", name="A. Author", source_ids=("src_1",)))
    kb.link("src_1", "auth_1", EdgeRelation.AUTHORED_BY, "arxiv:authorship")
    kb.add_evidence(evidence("evid_1", "src_dup"))
    prov = kb.provenance("evid_1")
    assert prov.source.id == "src_dup"
    assert prov.canonical_source is not None and prov.canonical_source.id == "src_1"
    assert [a.id for a in prov.authors] == ["auth_1"]
    assert prov.effective_claim_level is ClaimLevel.L0_UNVERIFIED
    with pytest.raises(UnknownEntityError):
        kb.provenance("evid_missing")


# ---------------------------------------------------------------- promotion
def _promotion_setup(kb: KnowledgeBase) -> None:
    kb.add_source(source("src_1"))
    kb.add_evidence(evidence("evid_1", "src_1"))
    kb.add_hypothesis(hypothesis(evidence_ids=("evid_1",)))
    kb.view.put(experiment("exp_ok"))
    kb.view.put(experiment("exp_failed", ExperimentStatus.FAILED))
    kb.view.put(report("vr_pass", GateOutcome.PASS))
    kb.view.put(report("vr_fail", GateOutcome.FAIL))


def test_full_promotion_ladder(kb: KnowledgeBase) -> None:
    _promotion_setup(kb)
    p1 = kb.promote("evid_1", ClaimLevel.L1_HYPOTHESIS, AgentRole.QUANT_RESEARCH, "hyp_1")
    assert p1.from_level is ClaimLevel.L0_UNVERIFIED and p1.entity_kind == "evidence"
    kb.promote("evid_1", ClaimLevel.L2_HISTORICAL, AgentRole.STRATEGY_ENGINEERING, "exp_ok")
    p3 = kb.promote(
        "evid_1", ClaimLevel.L3_INDEPENDENTLY_VALIDATED, SystemActor.VALIDATION_SERVICE, "vr_pass"
    )
    assert kb.claim_level("evid_1") is ClaimLevel.L3_INDEPENDENTLY_VALIDATED
    assert (
        kb.view.require(EvidenceItem, "evid_1").claim_level is ClaimLevel.L3_INDEPENDENTLY_VALIDATED
    )
    history = kb.view.list(ClaimPromotion, where={"entity_id": "evid_1"})
    assert len(history) == 3 and p3 in history
    assert kb.stats().evidence_by_claim_level == {"L3": 1}


def test_hypothesis_l3_is_recorded_by_promotion_only(kb: KnowledgeBase) -> None:
    _promotion_setup(kb)
    kb.promote("hyp_1", ClaimLevel.L2_HISTORICAL, AgentRole.QUANT_RESEARCH, "exp_ok")
    kb.promote(
        "hyp_1", ClaimLevel.L3_INDEPENDENTLY_VALIDATED, SystemActor.VALIDATION_SERVICE, "vr_pass"
    )
    assert kb.claim_level("hyp_1") is ClaimLevel.L3_INDEPENDENTLY_VALIDATED
    assert kb.view.require(Hypothesis, "hyp_1").claim_level is ClaimLevel.L2_HISTORICAL


DENIED: list[tuple[str, ClaimLevel, Actor, str, ClaimLevel]] = [
    # (description, target, actor, justification, starting level)
    (
        "intelligence cannot promote",
        ClaimLevel.L1_HYPOTHESIS,
        AgentRole.INTELLIGENCE,
        "hyp_1",
        ClaimLevel.L0_UNVERIFIED,
    ),
    (
        "L1 needs a real hypothesis",
        ClaimLevel.L1_HYPOTHESIS,
        AgentRole.QUANT_RESEARCH,
        "hyp_nope",
        ClaimLevel.L0_UNVERIFIED,
    ),
    (
        "no skipping L0->L2",
        ClaimLevel.L2_HISTORICAL,
        AgentRole.QUANT_RESEARCH,
        "exp_ok",
        ClaimLevel.L0_UNVERIFIED,
    ),
    (
        "no skipping L0->L3",
        ClaimLevel.L3_INDEPENDENTLY_VALIDATED,
        SystemActor.VALIDATION_SERVICE,
        "vr_pass",
        ClaimLevel.L0_UNVERIFIED,
    ),
    (
        "L2 needs succeeded experiment",
        ClaimLevel.L2_HISTORICAL,
        AgentRole.QUANT_RESEARCH,
        "exp_failed",
        ClaimLevel.L1_HYPOTHESIS,
    ),
    (
        "L2 not by validation",
        ClaimLevel.L2_HISTORICAL,
        AgentRole.INDEPENDENT_VALIDATION,
        "exp_ok",
        ClaimLevel.L1_HYPOTHESIS,
    ),
    (
        "L3 not by quant research",
        ClaimLevel.L3_INDEPENDENTLY_VALIDATED,
        AgentRole.QUANT_RESEARCH,
        "vr_pass",
        ClaimLevel.L2_HISTORICAL,
    ),
    (
        "L3 not by validation agent role",
        ClaimLevel.L3_INDEPENDENTLY_VALIDATED,
        AgentRole.INDEPENDENT_VALIDATION,
        "vr_pass",
        ClaimLevel.L2_HISTORICAL,
    ),
    (
        "L3 not by human",
        ClaimLevel.L3_INDEPENDENTLY_VALIDATED,
        SystemActor.HUMAN,
        "vr_pass",
        ClaimLevel.L2_HISTORICAL,
    ),
    (
        "L3 needs PASS report",
        ClaimLevel.L3_INDEPENDENTLY_VALIDATED,
        SystemActor.VALIDATION_SERVICE,
        "vr_fail",
        ClaimLevel.L2_HISTORICAL,
    ),
    (
        "L3 needs existing report",
        ClaimLevel.L3_INDEPENDENTLY_VALIDATED,
        SystemActor.VALIDATION_SERVICE,
        "vr_nope",
        ClaimLevel.L2_HISTORICAL,
    ),
    (
        "no demotion via promote",
        ClaimLevel.L0_UNVERIFIED,
        AgentRole.QUANT_RESEARCH,
        "hyp_1",
        ClaimLevel.L1_HYPOTHESIS,
    ),
    (
        "no same-level",
        ClaimLevel.L2_HISTORICAL,
        AgentRole.QUANT_RESEARCH,
        "exp_ok",
        ClaimLevel.L2_HISTORICAL,
    ),
]


@pytest.mark.parametrize(
    ("why", "target", "actor", "ref", "start"), DENIED, ids=[d[0] for d in DENIED]
)
def test_promotion_denials(
    kb: KnowledgeBase, why: str, target: ClaimLevel, actor: Actor, ref: str, start: ClaimLevel
) -> None:
    _promotion_setup(kb)
    if start in (ClaimLevel.L1_HYPOTHESIS, ClaimLevel.L2_HISTORICAL):
        kb.promote("evid_1", ClaimLevel.L1_HYPOTHESIS, AgentRole.QUANT_RESEARCH, "hyp_1")
    if start is ClaimLevel.L2_HISTORICAL:
        kb.promote("evid_1", ClaimLevel.L2_HISTORICAL, AgentRole.QUANT_RESEARCH, "exp_ok")
    before = kb.claim_level("evid_1")
    with pytest.raises(PermissionDeniedError):
        kb.promote("evid_1", target, actor, ref)
    assert kb.claim_level("evid_1") is before


def test_l3_impossible_without_validation_plane_read(stores: Stores, clock: FrozenClock) -> None:
    admin = kb_for(stores, clock)
    _promotion_setup(admin)
    admin.promote("evid_1", ClaimLevel.L1_HYPOTHESIS, AgentRole.QUANT_RESEARCH, "hyp_1")
    admin.promote("evid_1", ClaimLevel.L2_HISTORICAL, AgentRole.QUANT_RESEARCH, "exp_ok")
    research = kb_for(
        stores,
        clock,
        readable=frozenset({Plane.RESEARCH, Plane.CONTROL}),
        writable=frozenset({Plane.RESEARCH, Plane.CONTROL}),
    )
    with pytest.raises(PermissionDeniedError):
        research.promote(
            "evid_1",
            ClaimLevel.L3_INDEPENDENTLY_VALIDATED,
            SystemActor.VALIDATION_SERVICE,
            "vr_pass",
        )


def test_promotion_requires_control_plane_write(stores: Stores, clock: FrozenClock) -> None:
    admin = kb_for(stores, clock)
    _promotion_setup(admin)
    intel = kb_for(
        stores,
        clock,
        readable=frozenset({Plane.RESEARCH, Plane.CONTROL}),
        writable=frozenset({Plane.RESEARCH}),
    )
    with pytest.raises(PermissionDeniedError):
        intel.promote("evid_1", ClaimLevel.L1_HYPOTHESIS, AgentRole.QUANT_RESEARCH, "hyp_1")


def test_unknown_entity(kb: KnowledgeBase) -> None:
    with pytest.raises(UnknownEntityError):
        kb.claim_level("evid_missing")


# ---------------------------------------------------------------- stats
def test_stats(kb: KnowledgeBase) -> None:
    _seed_knowledge(kb)
    kb.add_source(source("src_2", provider="openalex", content="b"))
    kb.add_source(source("src_3", provider="crossref", content="c", doi="10.9999/z"), dedupe=False)
    kb.link("src_3", "src_2", EdgeRelation.DUPLICATE_OF, "dedupe:shingle-jaccard", 0.9)
    stats = kb.stats(newest=2)
    assert stats.counts["source"] == 3 and stats.counts["evidence"] == 2
    assert stats.counts["hypothesis"] == 1 and stats.counts["lesson"] == 1
    assert stats.sources_by_provider == {"arxiv": 1, "crossref": 1, "openalex": 1}
    assert stats.evidence_by_claim_level == {"L0": 2}
    assert stats.edges_by_relation == {"duplicate_of": 1}
    assert stats.duplicate_sources == 1
    assert len(stats.newest_sources) == 2
