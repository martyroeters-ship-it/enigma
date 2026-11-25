from __future__ import annotations

from datetime import UTC, datetime, timedelta

from enigma.research.scan import scan_rows

START = datetime(2026, 11, 1, tzinfo=UTC)


def _rows(values: list[float], *, future: float | None = None) -> list[dict[str, object]]:
    rows = [
        {"available_at": START + timedelta(hours=index), "funding_rate": value}
        for index, value in enumerate(values)
    ]
    if future is not None:
        rows.append(
            {
                "available_at": START + timedelta(hours=len(values) + 5),
                "funding_rate": future,
            }
        )
    return rows


def test_spike_ranks_first_and_future_rows_are_excluded() -> None:
    values = [0.0] * 19 + [10.0]
    cutoff = START + timedelta(hours=19)
    ranked = scan_rows(
        _rows(values, future=100.0), symbol="BTCUSDT", field="funding_rate", cutoff=cutoff
    )
    assert len(ranked) == 1
    assert ranked[0].index == 19
    assert ranked[0].zscore > 2.5
    assert ranked[0].suggested_mechanism == "funding_crowding_reversal"
    assert ranked[0].available_at <= cutoff


def test_quiet_series_and_short_window_rank_nothing() -> None:
    quiet = scan_rows(
        _rows([0.01] * 30),
        symbol="BTCUSDT",
        field="funding_rate",
        cutoff=START + timedelta(hours=29),
    )
    assert quiet == ()
    short = scan_rows(
        _rows([0.0, 10.0]),
        symbol="BTCUSDT",
        field="funding_rate",
        cutoff=START + timedelta(hours=1),
    )
    assert short == ()


def test_scan_is_deterministic() -> None:
    rows = _rows([0.0] * 12 + [4.0] + [0.0] * 12 + [-4.0])
    cutoff = START + timedelta(hours=len(rows))
    kwargs = {"symbol": "ETHUSDT", "field": "funding_rate", "cutoff": cutoff, "threshold": 2.0}
    first = scan_rows(rows, **kwargs)
    second = scan_rows(list(reversed(rows)), **kwargs)
    assert first == second
    assert len(first) == 2
    assert abs(first[0].zscore) >= abs(first[1].zscore)
