"""Seeded synthetic panels and strategy sources for tests only. Never real market results."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from enigma.schemas.common import AgentRole
from enigma.schemas.research import CostModel, StrategyPackage
from enigma.strategy import build_package

BAR = timedelta(hours=8)
START = datetime(2023, 1, 1, tzinfo=UTC)
CREATED = datetime(2026, 11, 25, tzinfo=UTC)


def make_panel(
    symbols: Sequence[str] = ("BTCUSDT", "ETHUSDT"),
    n_bars: int = 1200,
    seed: int = 0,
    edge: float = 0.0,
    vol: float = 0.01,
    funding_mean: float = 0.0,
    funding_sd: float = 1e-4,
) -> pl.DataFrame:
    """Long-format 8h perp panel. With edge > 0, returns load on the lagged observed basis."""
    rng = np.random.default_rng(seed)
    frames = []
    ts = [START + i * BAR for i in range(n_bars)]
    for symbol in symbols:
        phi = 0.97
        s = np.zeros(n_bars)
        shocks = rng.normal(0.0, np.sqrt(1 - phi**2), n_bars)
        for t in range(1, n_bars):
            s[t] = phi * s[t - 1] + shocks[t]
        eps = rng.normal(0.0, 1.0, n_bars)
        r = np.zeros(n_bars)
        r[1:] = edge * vol * s[:-1] + vol * eps[1:]
        close = 100.0 * np.cumprod(1.0 + r)
        basis = 0.001 * s
        funding = funding_mean + funding_sd * rng.normal(size=n_bars)
        frames.append(
            pl.DataFrame(
                {
                    "symbol": [symbol] * n_bars,
                    "ts": ts,
                    "available_at": [t + BAR for t in ts],
                    "perp_open": np.concatenate([[100.0], close[:-1]]),
                    "perp_high": close * 1.002,
                    "perp_low": close * 0.998,
                    "perp_close": close,
                    "perp_volume": np.full(n_bars, 1000.0),
                    "perp_quote_volume": close * 1000.0,
                    "spot_close": close / (1.0 + basis),
                    "funding_rate": funding,
                    "funding_time": [t + BAR for t in ts],
                    "basis": basis,
                },
                schema_overrides={
                    "ts": pl.Datetime("us", "UTC"),
                    "available_at": pl.Datetime("us", "UTC"),
                    "funding_time": pl.Datetime("us", "UTC"),
                },
            )
        )
    return pl.concat(frames).sort(["symbol", "ts"])


BASIS_SIGN = '''
"""Long when the observed basis is above a threshold, short when below."""
import numpy as np
from enigma.strategy.interface import Strategy, StrategyMeta


class BasisSign(Strategy):
    def configure(self, params):
        self.threshold = float(params.get("threshold", 0.0))

    def required_columns(self):
        return ("basis",)

    def metadata(self):
        return StrategyMeta(name="basis_sign", version="1")

    def target_positions(self, data):
        b = data["basis"]
        return np.where(b > self.threshold, 1.0, np.where(b < -self.threshold, -1.0, 0.0))
'''

MOMENTUM = """
import numpy as np
from enigma.strategy.interface import Strategy, StrategyMeta


class Momentum(Strategy):
    def configure(self, params):
        self.lookback = int(params.get("lookback", 3))

    def required_columns(self):
        return ("perp_close",)

    def metadata(self):
        return StrategyMeta(name="momentum", version="1")

    def target_positions(self, data):
        c = data["perp_close"]
        out = np.zeros(c.shape[0])
        k = self.lookback
        if c.shape[0] > k:
            out[k:] = np.sign(c[k:] - c[:-k])
        return out
"""

RANDOM_SIGN = """
import numpy as np
from enigma.strategy.interface import Strategy, StrategyMeta


class RandomSign(Strategy):
    def configure(self, params):
        self.seed = int(params.get("seed", 0))

    def required_columns(self):
        return ("perp_close",)

    def metadata(self):
        return StrategyMeta(name="random_sign", version="1")

    def target_positions(self, data):
        n = data["perp_close"].shape[0]
        draws = np.random.default_rng(self.seed).random(n)
        return np.where(draws < 0.5, -1.0, 1.0)
"""

# Uses the NEXT bar's close: textbook look-ahead.
LEAKY = """
import numpy as np
from enigma.strategy.interface import Strategy, StrategyMeta


class Leaky(Strategy):
    def required_columns(self):
        return ("perp_close",)

    def metadata(self):
        return StrategyMeta(name="leaky", version="1")

    def target_positions(self, data):
        c = data["perp_close"]
        nxt = np.append(c[1:], c[-1:])
        return np.sign(nxt - c)
"""

CRASHES = """
import numpy as np
from enigma.strategy.interface import Strategy, StrategyMeta


class Crashes(Strategy):
    def required_columns(self):
        return ("perp_close",)

    def metadata(self):
        return StrategyMeta(name="crashes", version="1")

    def target_positions(self, data):
        raise RuntimeError("boom")
"""


def cost_model(**overrides: float) -> CostModel:
    values: dict[str, float] = {
        "taker_fee_bps": 4.0,
        "maker_fee_bps": 2.0,
        "half_spread_bps": 1.0,
        "impact_bps_per_million": 10.0,
        "capacity_notional_usd": 5_000_000.0,
    }
    values.update(overrides)
    return CostModel(version="cm-test-1", **values)  # type: ignore[arg-type]


def package(
    source: str,
    entrypoint: str,
    columns: Sequence[str],
    params: dict[str, float | int | str] | None = None,
    *,
    package_id: str | None = None,
    hypothesis_id: str = "hyp_test",
    cm: CostModel | None = None,
) -> StrategyPackage:
    return build_package(
        source,
        entrypoint,
        params or {},
        cm or cost_model(),
        hypothesis_id,
        name=entrypoint.lower(),
        required_columns=columns,
        created_by=AgentRole.STRATEGY_ENGINEERING,
        created_at=CREATED,
        package_id=package_id,
    )
