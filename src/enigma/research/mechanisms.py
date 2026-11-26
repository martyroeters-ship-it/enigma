"""Mechanism library for the M1 hypothesis class: funding and basis structure in perpetuals.

Each mechanism is a causal story with a falsifiable formal statement, a pre-registered parameter
grid, and a strategy template. Templates use only numpy and decide position at bar t from data
known at that bar's `available_at` (no look-ahead); the validation plane verifies this
independently (G01).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

_ZSCORE_HELPER = """
def _rolling_z(x, lookback):
    n = x.shape[0]
    out = np.full(n, np.nan)
    csum = np.concatenate(([0.0], np.cumsum(x)))
    csq = np.concatenate(([0.0], np.cumsum(x * x)))
    for t in range(lookback - 1, n):
        s = csum[t + 1] - csum[t + 1 - lookback]
        q = csq[t + 1] - csq[t + 1 - lookback]
        mean = s / lookback
        var = max(q / lookback - mean * mean, 0.0)
        if var > 1e-18:
            out[t] = (x[t] - mean) / math.sqrt(var)
    return out
"""


@dataclass(frozen=True)
class Mechanism:
    id: str
    name: str
    mechanism: str
    prediction: str
    formal_statement: str
    keywords: tuple[str, ...]
    contra_keywords: tuple[str, ...]
    features: tuple[str, ...]
    param_grid: dict[str, tuple[float | int, ...]]
    entrypoint: str
    source_template: str
    horizon: str = "8h"

    def trials(self, budget: int) -> list[dict[str, float | int]]:
        """Deterministic, pre-registered trial list (grid order), truncated to the budget."""
        keys = sorted(self.param_grid)
        grid = [
            dict(zip(keys, values, strict=True))
            for values in product(*(self.param_grid[k] for k in keys))
        ]
        return grid[:budget]


FUNDING_CROWDING_REVERSAL = Mechanism(
    id="funding_crowding_reversal",
    name="Funding-rate crowding reversal",
    mechanism=(
        "Perpetual funding transfers cash from the crowded side to the other side. Abnormally "
        "high funding relative to its recent history indicates leveraged long crowding, which "
        "is fragile: longs pay to hold, marginal buyers are exhausted, and deleveraging or "
        "liquidations push price down. Abnormally negative funding indicates crowded shorts "
        "and the reverse. The position also collects funding while positioned against the crowd."
    ),
    prediction=(
        "When the funding-rate z-score over the trailing lookback exceeds +k, the next-bar "
        "perpetual return net of funding is negative on average; below -k it is positive."
    ),
    formal_statement=(
        "Let z_t = (f_t - mean(f_{t-L+1..t})) / sd(f_{t-L+1..t}). "
        "H1: E[r_{t+1} | z_t > k] < 0 and E[r_{t+1} | z_t < -k] > 0, where r_{t+1} is the "
        "8h perp return including funding paid by longs. H0: both conditional means equal 0."
    ),
    keywords=(
        "funding",
        "crowd",
        "leverage",
        "liquidation",
        "sentiment",
        "reversal",
        "contrarian",
        "open interest",
        "deleverag",
        "overheat",
    ),
    contra_keywords=("momentum", "persist", "trend"),
    features=("funding_rate",),
    param_grid={"lookback": (21, 42, 90), "z_entry": (1.0, 1.5, 2.0)},
    entrypoint="FundingCrowdingReversal",
    source_template="""
import math

import numpy as np

from enigma.strategy.interface import Strategy, StrategyMeta
"""
    + _ZSCORE_HELPER
    + """

class FundingCrowdingReversal(Strategy):
    def configure(self, params):
        self.lookback = int(params["lookback"])
        self.z_entry = float(params["z_entry"])

    def required_columns(self):
        return ("funding_rate",)

    def metadata(self):
        return StrategyMeta(name="funding_crowding_reversal", version="1")

    def target_positions(self, data):
        f = np.nan_to_num(np.asarray(data["funding_rate"], dtype=float), nan=0.0)
        z = _rolling_z(f, self.lookback)
        pos = np.zeros(f.shape[0])
        pos[z > self.z_entry] = -1.0
        pos[z < -self.z_entry] = 1.0
        return pos
""",
)

FUNDING_CARRY = Mechanism(
    id="funding_carry",
    name="Persistent funding carry",
    mechanism=(
        "Structural demand for leveraged long exposure keeps perpetual funding positive most "
        "of the time. Funding is persistent, so a short perpetual position held when trailing "
        "average funding exceeds a threshold collects a carry that compensates for directional "
        "risk, as long as the premium is not erased by price drift and trading costs."
    ),
    prediction=(
        "Holding short perp exposure when the trailing mean funding exceeds the threshold c "
        "earns positive net returns (funding received exceeds price drift plus costs)."
    ),
    formal_statement=(
        "Let m_t = mean(f_{t-L+1..t}). H1: E[-r_{t+1} | m_t > c] > 0 net of costs, where "
        "r_{t+1} includes funding. H0: E[-r_{t+1} | m_t > c] <= 0."
    ),
    keywords=(
        "carry",
        "funding",
        "premium",
        "arbitrage",
        "basis trade",
        "cash-and-carry",
        "persist",
        "risk premium",
    ),
    contra_keywords=("arbitraged away", "no premium", "decline"),
    features=("funding_rate",),
    param_grid={"lookback": (9, 21, 42), "threshold_bps": (0.5, 1.0, 2.0)},
    entrypoint="FundingCarry",
    source_template="""
import numpy as np

from enigma.strategy.interface import Strategy, StrategyMeta


class FundingCarry(Strategy):
    def configure(self, params):
        self.lookback = int(params["lookback"])
        self.threshold = float(params["threshold_bps"]) / 10000.0

    def required_columns(self):
        return ("funding_rate",)

    def metadata(self):
        return StrategyMeta(name="funding_carry", version="1")

    def target_positions(self, data):
        f = np.nan_to_num(np.asarray(data["funding_rate"], dtype=float), nan=0.0)
        n = f.shape[0]
        csum = np.concatenate(([0.0], np.cumsum(f)))
        pos = np.zeros(n)
        for t in range(self.lookback - 1, n):
            mean = (csum[t + 1] - csum[t + 1 - self.lookback]) / self.lookback
            if mean > self.threshold:
                pos[t] = -1.0
        return pos
""",
)

BASIS_REVERSION = Mechanism(
    id="basis_reversion",
    name="Perp-spot basis mean reversion",
    mechanism=(
        "The perpetual is tied to spot by funding and arbitrage. When the perp trades at an "
        "unusually large premium (or discount) to spot, arbitrageurs short the perp and buy "
        "spot (or the reverse), and funding adjusts, pulling the basis back. Part of the "
        "convergence shows up in the perp leg's return relative to its recent behaviour."
    ),
    prediction=(
        "When the basis z-score exceeds +k the next-bar perp return is negative on average; "
        "below -k it is positive."
    ),
    formal_statement=(
        "Let b_t = perp_close_t / spot_close_t - 1 and z_t its trailing z-score over L bars. "
        "H1: E[r_{t+1} | z_t > k] < 0 and E[r_{t+1} | z_t < -k] > 0. H0: both equal 0."
    ),
    keywords=(
        "basis",
        "premium",
        "arbitrage",
        "spot",
        "convergence",
        "mean reversion",
        "mispricing",
        "index price",
    ),
    contra_keywords=("persistent premium", "momentum"),
    features=("basis",),
    param_grid={"lookback": (21, 42, 90), "z_entry": (1.5, 2.0, 2.5)},
    entrypoint="BasisReversion",
    source_template="""
import math

import numpy as np

from enigma.strategy.interface import Strategy, StrategyMeta
"""
    + _ZSCORE_HELPER
    + """

class BasisReversion(Strategy):
    def configure(self, params):
        self.lookback = int(params["lookback"])
        self.z_entry = float(params["z_entry"])

    def required_columns(self):
        return ("basis",)

    def metadata(self):
        return StrategyMeta(name="basis_reversion", version="1")

    def target_positions(self, data):
        b = np.nan_to_num(np.asarray(data["basis"], dtype=float), nan=0.0)
        z = _rolling_z(b, self.lookback)
        pos = np.zeros(b.shape[0])
        pos[z > self.z_entry] = -1.0
        pos[z < -self.z_entry] = 1.0
        return pos
""",
)

FUNDING_MOMENTUM = Mechanism(
    id="funding_momentum",
    name="Funding-confirmed trend continuation",
    mechanism=(
        "Rising funding reflects growing demand for leveraged exposure in the direction of the "
        "recent trend. Slow-moving capital and information diffusion keep that demand "
        "persistent, so trends confirmed by funding continue for several bars before crowding "
        "costs dominate."
    ),
    prediction=(
        "When trailing return and trailing funding change have the same sign, the next-bar "
        "return has that sign on average."
    ),
    formal_statement=(
        "Let R_t = sum r_{t-L+1..t} and D_t = f_t - f_{t-L}. H1: E[sign(R_t) r_{t+1} | "
        "sign(R_t) = sign(D_t)] > 0. H0: it equals 0."
    ),
    keywords=("momentum", "trend", "persist", "diffusion", "underreact", "time-series momentum"),
    contra_keywords=("reversal", "contrarian", "mean reversion"),
    features=("funding_rate", "perp_close"),
    param_grid={"lookback": (9, 21, 42)},
    entrypoint="FundingMomentum",
    source_template="""
import numpy as np

from enigma.strategy.interface import Strategy, StrategyMeta


class FundingMomentum(Strategy):
    def configure(self, params):
        self.lookback = int(params["lookback"])

    def required_columns(self):
        return ("funding_rate", "perp_close")

    def metadata(self):
        return StrategyMeta(name="funding_momentum", version="1")

    def target_positions(self, data):
        f = np.nan_to_num(np.asarray(data["funding_rate"], dtype=float), nan=0.0)
        p = np.asarray(data["perp_close"], dtype=float)
        n = p.shape[0]
        pos = np.zeros(n)
        L = self.lookback
        for t in range(L, n):
            trend = np.sign(p[t] / p[t - L] - 1.0)
            dfund = np.sign(f[t] - f[t - L])
            if trend != 0.0 and trend == dfund:
                pos[t] = trend
        return pos
""",
)

MECHANISMS: dict[str, Mechanism] = {
    m.id: m for m in (FUNDING_CROWDING_REVERSAL, FUNDING_CARRY, BASIS_REVERSION, FUNDING_MOMENTUM)
}

# Names the model actually emits. "cryptocurrency" is intentionally absent: it is not a mechanism.
_DOMAIN_ALIASES = {
    "funding": "funding_crowding_reversal",
    "funding_rate": "funding_crowding_reversal",
    "crowding": "funding_crowding_reversal",
    "funding_crowding": "funding_crowding_reversal",
    "carry": "funding_carry",
    "funding_carry": "funding_carry",
    "basis": "basis_reversion",
    "basis_mean_reversion": "basis_reversion",
    "basis_reversion": "basis_reversion",
    "momentum": "funding_momentum",
    "funding_momentum": "funding_momentum",
    "novel": "novel",
}


def canonical_domain(domain: str) -> str:
    """Strip repeated ``fam_`` prefixes and map aliases onto a mechanism id.

    Unknown text is returned unchanged (aside from the prefix strip and
    normalization) so a check can reject it instead of silently relabeling it.
    """
    text = domain.strip()
    while text.startswith("fam_"):
        text = text.removeprefix("fam_")
    key = text.casefold().replace(" ", "_").replace("-", "_")
    return _DOMAIN_ALIASES.get(key, key)


def family_id_for_domain(domain: str) -> str:
    """One ``fam_`` prefix. ``fam_fam_cryptocurrency`` does not become a family."""
    return f"fam_{canonical_domain(domain)}"
