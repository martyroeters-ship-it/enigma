from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from enigma.data.binance import BinanceArchiveClient
from enigma.data.hashing import frame_content_hash
from enigma.data.panel import build_perp_funding_panel

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 2, 1, tzinfo=UTC)
# Pinned hash of the recorded BTCUSDT+ETHUSDT 2024-01 panel: guards cross-machine and
# cross-version stability of the canonical serialization. Changing it requires a new scheme.
GOLDEN = "sha256:ed8718aa445efe1709a1946dbdb9fb3cc044365f142734557ff695c99f873d95"


@pytest.fixture
def panel(client: BinanceArchiveClient) -> pl.DataFrame:
    return build_perp_funding_panel(["BTCUSDT", "ETHUSDT"], START, END, client=client)


def test_hash_is_pinned(panel: pl.DataFrame) -> None:
    assert frame_content_hash(panel) == GOLDEN


def test_hash_independent_of_row_order_and_parquet_roundtrip(
    panel: pl.DataFrame, tmp_path: Path
) -> None:
    shuffled = panel.sample(fraction=1.0, shuffle=True, seed=7)
    assert frame_content_hash(shuffled) == frame_content_hash(panel)
    path = tmp_path / "p.parquet"
    panel.write_parquet(path, compression="snappy")
    assert frame_content_hash(pl.read_parquet(path)) == frame_content_hash(panel)


def test_hash_sensitive_to_values_nulls_and_names(panel: pl.DataFrame) -> None:
    base = frame_content_hash(panel)
    nudged = panel.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("perp_close") + 1e-9)
        .otherwise(pl.col("perp_close"))
        .alias("perp_close")
    )
    assert frame_content_hash(nudged) != base
    nulled = panel.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("basis")).alias("basis")
    )
    assert frame_content_hash(nulled) != base
    assert frame_content_hash(panel.rename({"basis": "basis2"})) != base
    assert frame_content_hash(panel.head(10)) != base
