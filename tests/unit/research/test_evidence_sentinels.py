"""LLM sentinel evidence ids must not fail closed as unknown citations."""

from __future__ import annotations

from datetime import UTC, datetime

from enigma.research.checks import check_proposal, normalize_proposal
from enigma.research.mechanisms import canonical_domain, family_id_for_domain
from enigma.schemas.quant import (
    ColumnUse,
    DatasetInfo,
    EvidenceExcerpt,
    ExecutionCostEstimate,
    ExperimentalDesign,
    HypothesisProposal,
    ParameterGrid,
    ResearchContext,
)

NOW = datetime(2026, 3, 1, tzinfo=UTC)


def _proposal(**overrides: object) -> HypothesisProposal:
    base = {
        "id": "hyp_1",
        "title": "Funding edge",
        "domain": "funding_crowding_reversal",
        "market_mechanism": "funding",
        "inefficiency_rationale": "crowding",
        "persistence_rationale": "hours",
        "signal_definition": "sign(funding_rate)",
        "target_definition": "forward return",
        "test_statistic": "net_sharpe",
        "required_data": [
            ColumnUse(dataset_id="ds_1", column="funding_rate", role="signal", min_lag_bars=0),
            ColumnUse(dataset_id="ds_1", column="close", role="target", min_lag_bars=1),
        ],
        "horizon_bars": 8,
        "horizon": "8h",
        "null_hypothesis": "no edge",
        "alternative_hypothesis": "funding predicts returns",
        "design": ExperimentalDesign(
            parameters=(ParameterGrid(name="W", values=(4, 8)),),
            grid_size=2,
            train_test_protocol="expanding",
            primary_metric="net_sharpe",
        ),
        "confounders": ("basis",),
        "execution_costs": ExecutionCostEstimate(
            turnover_per_bar=0.1,
            bars_per_year=1095.0,
            cost_bps_per_unit_turnover=2.0,
            annual_cost_bps=219.0,
            cost_model_version="cm-binance-usdm-v1",
            rationale="fees",
        ),
        "failure_conditions": ("holdout sharpe < 0",),
        "supporting_evidence_ids": ("ev_1",),
        "contradictory_evidence_ids": ("none",),
        "novelty_statement": "cost-aware funding",
        "complexity_class": "rule",
        "complexity_rationale": "sign rule",
        "instruments": ("BTCUSDT",),
    }
    base.update(overrides)
    return HypothesisProposal.model_validate(base)


def test_contradictory_none_sentinel_is_ignored() -> None:
    ctx = ResearchContext(
        id="ctx",
        objective_id="obj",
        objective="funding",
        research_cutoff=NOW,
        instruments=("BTCUSDT",),
        cost_model_version="cm-binance-usdm-v1",
        cost_bps_per_unit_turnover=2.0,
        evidence=(
            EvidenceExcerpt(
                id="ev_1",
                source_id="src",
                text="funding positive",
                claim_level="L1",
                topics=("funding",),
            ),
            EvidenceExcerpt(
                id="ev_2",
                source_id="src2",
                text="carry collapsed",
                claim_level="L1",
                topics=("funding",),
            ),
        ),
        datasets=(
            DatasetInfo(
                dataset_version_id="ds_1",
                name="x",
                frequency="1h",
                bar_hours=1.0,
                symbols=("BTCUSDT",),
                columns=("funding_rate", "close"),
                coverage_start=NOW,
                coverage_end=NOW,
                rows=10,
            ),
        ),
    )
    failures = check_proposal(_proposal(), ctx)
    assert not any(f.check == "evidence_exists" for f in failures)


def _ctx_with_real_ids() -> ResearchContext:
    return ResearchContext(
        id="ctx",
        objective_id="obj",
        objective="funding",
        research_cutoff=NOW,
        instruments=("BTCUSDT",),
        cost_model_version="cm-binance-usdm-v1",
        cost_bps_per_unit_turnover=2.0,
        evidence=(
            EvidenceExcerpt(
                id="evid_01AAA",
                source_id="src",
                text="funding positive",
                claim_level="L1",
                topics=("funding",),
            ),
            EvidenceExcerpt(
                id="evid_01BBB",
                source_id="src2",
                text="carry collapsed",
                claim_level="L1",
                topics=("funding",),
            ),
        ),
        datasets=(
            DatasetInfo(
                dataset_version_id="dsv_01REAL",
                name="x",
                frequency="1h",
                bar_hours=1.0,
                symbols=("BTCUSDT",),
                columns=("funding_rate", "close"),
                coverage_start=NOW,
                coverage_end=NOW,
                rows=10,
            ),
        ),
    )


def test_normalize_remaps_short_evidence_and_dataset() -> None:
    ctx = _ctx_with_real_ids()
    draft = _proposal(
        required_data=[
            ColumnUse(
                dataset_id="invented_ds", column="funding_rate", role="signal", min_lag_bars=-2
            ),
            ColumnUse(dataset_id="invented_ds", column="close", role="target", min_lag_bars=1),
        ],
        supporting_evidence_ids=("ev_1",),
        contradictory_evidence_ids=("none",),
        execution_costs=ExecutionCostEstimate(
            turnover_per_bar=0.1,
            bars_per_year=1095.0,
            cost_bps_per_unit_turnover=2.0,
            annual_cost_bps=219.0,
            cost_model_version="wrong-cm",
            rationale="fees",
        ),
    )
    fixed = normalize_proposal(draft, ctx)
    assert fixed.supporting_evidence_ids == ("evid_01AAA",)
    assert fixed.contradictory_evidence_ids == ()
    assert all(u.dataset_id == "dsv_01REAL" for u in fixed.required_data)
    assert fixed.required_data[0].min_lag_bars == 0
    assert fixed.execution_costs.cost_model_version == "cm-binance-usdm-v1"
    assert not check_proposal(fixed, ctx)


def test_normalize_aliases_returns_column_onto_perp_close() -> None:
    """Cached Ollama drafts used column 'returns'; remap onto the sealed PIT panel."""
    ctx = ResearchContext(
        id="ctx",
        objective_id="obj",
        objective="funding predicts returns",
        research_cutoff=NOW,
        instruments=("BTCUSDT",),
        cost_model_version="cm-binance-usdm-v1",
        cost_bps_per_unit_turnover=2.0,
        evidence=(
            EvidenceExcerpt(
                id="evid_01AAA",
                source_id="src",
                text="funding positive",
                claim_level="L1",
                topics=("funding",),
            ),
        ),
        datasets=(
            DatasetInfo(
                dataset_version_id="dsv_01REAL",
                name="binance_perp_funding_8h",
                frequency="8h",
                bar_hours=8.0,
                symbols=("BTCUSDT",),
                columns=(
                    "symbol",
                    "ts",
                    "available_at",
                    "perp_open",
                    "perp_high",
                    "perp_low",
                    "perp_close",
                    "perp_volume",
                    "spot_close",
                    "funding_rate",
                    "basis",
                ),
                coverage_start=NOW,
                coverage_end=NOW,
                rows=5000,
            ),
        ),
    )
    draft = _proposal(
        required_data=[
            ColumnUse(dataset_id="src_invented", column="returns", role="target", min_lag_bars=1),
            ColumnUse(dataset_id="src_invented", column="returns", role="target", min_lag_bars=16),
        ],
        supporting_evidence_ids=("evid_01AAA", "evid_missing"),
        contradictory_evidence_ids=(),
    )
    fixed = normalize_proposal(draft, ctx)
    assert fixed.required_data[0].role == "signal"
    assert fixed.required_data[0].column == "funding_rate"
    assert all(u.column == "perp_close" for u in fixed.required_data if u.role == "target")
    assert all(u.dataset_id == "dsv_01REAL" for u in fixed.required_data)
    assert fixed.supporting_evidence_ids == ("evid_01AAA",)
    assert not check_proposal(fixed, ctx)


def test_unknown_domain_fails_and_aliases_canonicalize() -> None:
    ctx = ResearchContext(
        id="ctx",
        objective_id="obj",
        objective="funding",
        research_cutoff=NOW,
        instruments=("BTCUSDT",),
        cost_model_version="cm-binance-usdm-v1",
        cost_bps_per_unit_turnover=2.0,
        evidence=(
            EvidenceExcerpt(
                id="ev_1",
                source_id="src",
                text="funding positive",
                claim_level="L1",
                topics=("funding",),
            ),
        ),
        datasets=(
            DatasetInfo(
                dataset_version_id="ds_1",
                name="binance_perp_funding_8h",
                frequency="8h",
                bar_hours=8.0,
                symbols=("BTCUSDT",),
                columns=("funding_rate", "close"),
                coverage_start=NOW,
                coverage_end=NOW,
                rows=100,
            ),
        ),
    )
    failures = check_proposal(_proposal(domain="fam_fam_cryptocurrency"), ctx)
    assert any(f.check == "registered_mechanism" for f in failures)
    fixed = normalize_proposal(_proposal(domain="funding"), ctx)
    assert fixed.domain == "funding_crowding_reversal"
    assert "registered_mechanism" not in {f.check for f in check_proposal(fixed, ctx)}
    assert canonical_domain("fam_fam_cryptocurrency") == "cryptocurrency"
    assert family_id_for_domain("fam_funding_carry") == "fam_funding_carry"
    assert family_id_for_domain("fam_fam_cryptocurrency") == "fam_cryptocurrency"
