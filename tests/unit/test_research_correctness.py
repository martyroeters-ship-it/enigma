"""Sampling frequency, embargo equality, and bounded execution failures."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from cycle_rig import Rig

from enigma.backtest.statistics import sharpe_ratio
from enigma.data.panel import periods_per_year
from enigma.orchestrator.failures import (
    block_reason,
    blocked_ids,
    classify_failure,
    record_execution_failure,
)
from enigma.validation.corrections import (
    note_g02_equality_candidates,
    research_dataset_overlaps_embargo,
)
from enigma.validation.gates import embargo_bars
from enigma.validation.protocol import load_protocol, protocol_for_sampling


def test_annualization_matches_the_sampling_frequency() -> None:
    assert periods_per_year("8h") == 1095
    assert periods_per_year("1h") == 8760
    assert periods_per_year("1d") == 365
    returns = np.array([0.01, -0.002, 0.004, 0.003])
    hourly = sharpe_ratio(returns, 8760)
    eight = sharpe_ratio(returns, 1095)
    daily = sharpe_ratio(returns, 365)
    assert hourly == pytest.approx(eight * math.sqrt(8760 / 1095))
    assert daily == pytest.approx(eight * math.sqrt(365 / 1095))
    assert hourly != pytest.approx(eight)


def test_frequency_protocols_keep_the_eight_hour_seal_and_the_full_gate_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed = load_protocol()
    full = load_protocol(version="1.0.0")
    assert sealed.version == "0.1.0-slice"
    assert tuple(full.required_gates) == (
        "G01",
        "G02",
        "G03",
        "G04",
        "G06",
        "G07",
        "G08",
        "G09",
        "G10",
        "G11",
        "G12",
        "G15",
        "G16",
    )
    hourly = load_protocol(version="0.3.0-1h")
    daily = load_protocol(version="0.3.0-1d")
    assert tuple(hourly.required_gates) == tuple(full.required_gates)
    assert tuple(daily.required_gates) == tuple(full.required_gates)
    assert hourly.periods_per_year == 8760
    assert daily.periods_per_year == 365
    assert embargo_bars(hourly.gate("G02"), 1) == 168
    assert embargo_bars(hourly.gate("G02"), 100) == 300
    assert embargo_bars(daily.gate("G02"), 1) == 7
    assert int(hourly.gate("G10")["vol_window_bars"]) == 168
    assert int(daily.gate("G11")["mean_block_bars"]) == 4
    assert hourly.gate("G04") == full.gate("G04")
    assert hourly.gate("G15")["benchmarks"] == full.gate("G15")["benchmarks"]
    assert protocol_for_sampling("1h", sealed).version == "0.3.0-1h"
    assert protocol_for_sampling("8h", full).version == "1.0.0"
    monkeypatch.setenv("ENIGMA_VALIDATION_PROTOCOL", "1.0.0")
    assert load_protocol().version == "1.0.0"


def test_dataset_ending_on_the_embargo_start_is_not_an_overlap() -> None:
    embargo = 1_700_000_000_000_000_000
    assert research_dataset_overlaps_embargo(embargo, embargo) is False
    assert research_dataset_overlaps_embargo(embargo + 1, embargo) is True
    assert research_dataset_overlaps_embargo(embargo - 1, embargo) is False


def test_g02_note_does_not_rewrite_reports(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "g02.sqlite3")
    payload = note_g02_equality_candidates(rig.stores)
    assert payload["rewritten"] is False
    assert payload["holdout_rows_read"] is False
    assert payload["candidate_summary_ids"] == []
    stored = rig.stores.flags.get("correction.g02-embargo-equality")
    assert stored is not None
    assert "g02-embargo-equality-strict" in stored


def test_failure_classes_backoff_then_quarantine(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "block.sqlite3")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert classify_failure("ollama provider unavailable") == "model"
    assert classify_failure("dataset frequency has no bars") == "data"
    assert classify_failure("strategy codegen raised SyntaxError") == "implementation"
    assert classify_failure("docker sandbox failed") == "infrastructure"
    assert classify_failure("hypothesis rejected by the pre-registered test") == "scientific"

    record_execution_failure(rig.stores, "opp_a", "docker sandbox failed", now)
    assert "opp_a" in blocked_ids(rig.stores, now)
    assert "opp_a" not in blocked_ids(rig.stores, now + timedelta(hours=2))
    record_execution_failure(rig.stores, "opp_a", "docker sandbox failed", now + timedelta(hours=2))
    assert "opp_a" in blocked_ids(rig.stores, now + timedelta(hours=3))
    assert "opp_a" not in blocked_ids(rig.stores, now + timedelta(hours=5))
    record_execution_failure(rig.stores, "opp_a", "docker sandbox failed", now + timedelta(hours=5))
    assert "opp_a" in blocked_ids(rig.stores, now + timedelta(days=30))
    assert block_reason(rig.stores, "opp_a").startswith("infrastructure:")

    record_execution_failure(rig.stores, "opp_b", "hypothesis rejected", now)
    assert "opp_b" in blocked_ids(rig.stores, now + timedelta(days=30))
    assert block_reason(rig.stores, "opp_b").startswith("scientific:")
