"""AT15: research cannot read validation reports or holdout custody; summaries leak no paths."""

from __future__ import annotations

import pytest

from _synthetic import MOMENTUM, make_panel, package
from _validation_env import EnvFactory
from enigma.core.errors import PermissionDeniedError
from enigma.schemas.common import Plane
from enigma.schemas.validation import (
    HoldoutManifest,
    RedactedValidationSummary,
    ValidationReport,
)
from enigma.store import PlaneView
from enigma.validation import ValidationService
from enigma.validation.holdout import VALIDATION_DB_NAME


def test_research_plane_cannot_read_report_or_manifest(make_env: EnvFactory) -> None:
    env = make_env(make_panel(seed=61))
    report = env.service.validate(env.submit(package(MOMENTUM, "Momentum", ("perp_close",))))
    with pytest.raises(PermissionDeniedError):
        env.research.get(ValidationReport, report.id)
    with pytest.raises(PermissionDeniedError):
        env.research.list(ValidationReport)
    with pytest.raises(PermissionDeniedError):
        env.research.list(HoldoutManifest)
    # The control DB holds no holdout manifest at all; custody lives in the validation DB.
    assert env.stores.documents.count(HoldoutManifest) == 0
    assert env.service.custody.documents.count(HoldoutManifest) == 1
    summaries = env.research.list(RedactedValidationSummary)
    assert [s.report_id for s in summaries] == [report.id]


def test_redacted_summary_does_not_reveal_validation_paths(make_env: EnvFactory) -> None:
    env = make_env(make_panel(seed=62))
    report = env.service.validate(env.submit(package(MOMENTUM, "Momentum", ("perp_close",))))
    summary = env.research.list(RedactedValidationSummary)[0]
    blob = summary.model_dump_json()
    for secret in (
        str(env.validation_dir),
        env.validation_dir.name,
        VALIDATION_DB_NAME,
        "sqlite",
        "hmf_",
        "dsv_validation_test",
        env.frame.get_column("ts").max().isoformat()[:10],
    ):
        assert secret not in blob
    assert set(RedactedValidationSummary.model_fields) == {
        "id",
        "report_id",
        "strategy_package_id",
        "protocol_version",
        "overall",
        "gate_outcomes",
        "failed_gate_names",
    }
    assert summary.overall is report.overall
    # Nor does the full report carry filesystem locations.
    assert str(env.validation_dir) not in report.model_dump_json()


def test_service_refuses_a_research_scoped_view(make_env: EnvFactory) -> None:
    env = make_env(make_panel(seed=63, n_bars=600))
    research_only = PlaneView(
        env.stores.documents,
        readable=frozenset({Plane.CONTROL, Plane.RESEARCH}),
        writable=frozenset({Plane.CONTROL, Plane.RESEARCH}),
        actor="quant_research",
    )
    with pytest.raises(PermissionDeniedError):
        ValidationService(
            env.validation_dir,
            research_only,
            env.service.holdout_source,
            env.stores.audit,
            env.fsm,
        )
