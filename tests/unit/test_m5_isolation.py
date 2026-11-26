"""M5 host boundary. These tests do not mark the two-machine acceptance passed."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from enigma.core.errors import IntegrityError, PermissionDeniedError
from enigma.isolation.hosts import (
    HostRole,
    assert_clock_skew,
    assert_credential_separation,
    assert_may_read_research_trials,
    execution_clearance,
    host_role,
)
from enigma.validation.holdout import HoldoutCustody

ROOT = Path(__file__).resolve().parents[2]


def test_research_and_execution_cannot_open_the_validation_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for role in ("research", "execution"):
        monkeypatch.setenv("ENIGMA_HOST_ROLE", role)
        with pytest.raises(PermissionDeniedError, match="validation directory"):
            HoldoutCustody(tmp_path / role)


def test_validation_role_can_open_and_single_role_stays_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENIGMA_HOST_ROLE", "validation")
    opened = HoldoutCustody(tmp_path / "validation")
    opened.close()
    monkeypatch.delenv("ENIGMA_HOST_ROLE", raising=False)
    single = HoldoutCustody(tmp_path / "single")
    single.close()


def test_other_hosts_cannot_read_the_trial_ledger() -> None:
    assert_may_read_research_trials(HostRole.RESEARCH)
    assert_may_read_research_trials(HostRole.SINGLE)
    for role in (HostRole.VALIDATION, HostRole.EXECUTION):
        with pytest.raises(PermissionDeniedError, match="trial ledger"):
            assert_may_read_research_trials(role)


def test_credentials_stay_on_their_host() -> None:
    assert_credential_separation(HostRole.RESEARCH, {})
    with pytest.raises(PermissionDeniedError, match="validation credential"):
        assert_credential_separation(
            HostRole.RESEARCH, {"ENIGMA_VALIDATION_CREDENTIAL": "not-printed"}
        )
    with pytest.raises(PermissionDeniedError, match="execution credential"):
        assert_credential_separation(
            HostRole.VALIDATION, {"ENIGMA_EXECUTION_CREDENTIAL": "not-printed"}
        )


def test_clock_skew_beyond_the_embargo_fails() -> None:
    local = datetime(2026, 11, 25, tzinfo=UTC)
    embargo = timedelta(hours=8)
    assert_clock_skew(local, local + timedelta(hours=1), embargo)
    with pytest.raises(IntegrityError, match="clock skew"):
        assert_clock_skew(local, local + timedelta(days=2), embargo)


def test_execution_clearance_drops_trial_rows() -> None:
    view = execution_clearance(
        {
            "id": "rsk_1",
            "strategy_package_id": "pkg_1",
            "decision": "veto",
            "reason_codes": ["validation_not_passed"],
            "experiments": [{"id": "exp_1", "metrics": {"net_sharpe": 1.2}}],
        }
    )
    assert "experiments" not in view
    assert view["decision"] == "veto"


def test_unknown_host_role_is_refused() -> None:
    with pytest.raises(PermissionDeniedError, match="unknown"):
        host_role({"ENIGMA_HOST_ROLE": "root"})


def test_isolate_hosts_script_fails_closed_without_the_unix_users() -> None:
    script = ROOT / "deploy" / "m5" / "isolate-hosts.sh"
    result = subprocess.run(  # noqa: S603 - fixed interpreter, repository script
        ["/bin/sh", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "enigma-validation" in result.stdout
