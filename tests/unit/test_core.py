from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from enigma.core import FrozenClock, canonical_json, new_id, sha256_of
from enigma.core.ids import id_prefix


def test_ids_are_prefixed_unique_and_sortable() -> None:
    ids = [new_id("hyp") for _ in range(200)]
    assert len(set(ids)) == 200
    assert all(id_prefix(i) == "hyp" for i in ids)
    assert len(ids[0]) == len("hyp_") + 26


def test_invalid_prefix_rejected() -> None:
    with pytest.raises(ValueError):
        new_id("Bad-Prefix")


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})
    assert sha256_of({"x": 1.5}) == sha256_of({"x": 1.5})
    assert sha256_of({"x": 1}).startswith("sha256:")


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_frozen_clock_requires_tz_and_steps() -> None:
    with pytest.raises(ValueError):
        FrozenClock(datetime(2024, 1, 1))
    clock = FrozenClock(datetime(2024, 1, 1, tzinfo=UTC), step=timedelta(seconds=1))
    assert clock.now() < clock.now()
