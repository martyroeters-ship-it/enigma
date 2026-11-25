"""Thin, deterministic bar backtest engine (ADR-012).

Timing (per symbol, bars s = 0..n-1, close C_s known at available_at[s]):
- p_t is the target decided at the close of bar t.
- With L = cost_model.latency_bars, it is filled at the close of bar t+L: the executed position
  after the close of bar s is e_s = p_{s-L} (0 for s < L).
- The position held during bar s is h_s = e_{s-1} (h_0 = 0).

Per-bar accounting, as fractions of the sleeve notional:
- price:   h_s * (C_s / C_{s-1} - 1)
- funding: -h_s * funding_rate_s. The rate in row s settled at a funding event inside bar s
           (at or before available_at[s]) on the position held into it, so a long pays positive
           funding. Trades at the close of bar s happen after that settlement.
- costs:   on turnover |d_s| = |e_s - e_{s-1}| at the close of bar s:
           |d_s| * (taker_fee_bps + half_spread_bps) / 1e4
           + |d_s| * impact_bps_per_million * (|d_s| * sleeve_notional / 1e6) / 1e4
           all scaled by `cost_multiplier` (validation stress). Funding is a cash flow, not a
           cost, and is never scaled.
- gross = price + funding; net = gross - costs.

Multi-symbol: `notional_usd` is split equally across symbols (sleeves). Per-bar portfolio
series are the equal-weight mean of sleeve series on the union of bar timestamps (a sleeve
without a bar contributes 0). A sleeve notional above `capacity_notional_usd` is rejected.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import polars as pl

from enigma.backtest.metrics import compute_metrics
from enigma.core.errors import EnigmaError
from enigma.core.hashing import canonical_json, sha256_of
from enigma.schemas.research import CostModel

DEFAULT_PERIODS_PER_YEAR = 1095.0
SERIES_FIELDS = ("price", "funding", "cost", "gross", "net", "turnover", "exposure", "trades")


class CapacityExceededError(EnigmaError):
    """Requested notional exceeds the cost model's declared capacity."""


@dataclass(frozen=True)
class SymbolSeries:
    symbol: str
    ts: np.ndarray
    target: np.ndarray
    executed: np.ndarray
    held: np.ndarray
    price: np.ndarray
    funding: np.ndarray
    cost: np.ndarray
    gross: np.ndarray
    net: np.ndarray
    turnover: np.ndarray


@dataclass(frozen=True)
class BacktestResult:
    ts: np.ndarray
    series: dict[str, np.ndarray]
    per_symbol: dict[str, SymbolSeries]
    metrics: dict[str, float]
    equity: np.ndarray
    cost_model_hash: str
    notional_usd: float
    cost_multiplier: float
    periods_per_year: float
    seed: int
    result_hash: str = field(default="")

    @property
    def net_returns(self) -> np.ndarray:
        return self.series["net"]

    @property
    def gross_returns(self) -> np.ndarray:
        return self.series["gross"]

    def window(self, start_ns: int | None = None, end_ns: int | None = None) -> BacktestResult:
        """Portfolio series restricted to bars with start_ns <= ts < end_ns, metrics recomputed.

        Positions entering the window are inherited; trades before it are not charged in it.
        """
        mask = np.ones(self.ts.shape[0], dtype=bool)
        if start_ns is not None:
            mask &= self.ts >= start_ns
        if end_ns is not None:
            mask &= self.ts < end_ns
        series = {k: v[mask] for k, v in self.series.items()}
        per_symbol = {
            sym: SymbolSeries(
                sym,
                *(
                    getattr(s, name)[_symbol_mask(s.ts, start_ns, end_ns)]
                    for name in (
                        "ts",
                        "target",
                        "executed",
                        "held",
                        "price",
                        "funding",
                        "cost",
                        "gross",
                        "net",
                        "turnover",
                    )
                ),
            )
            for sym, s in self.per_symbol.items()
        }
        return _finalize(
            self.ts[mask],
            series,
            per_symbol,
            self.cost_model_hash,
            self.notional_usd,
            self.cost_multiplier,
            self.periods_per_year,
            self.seed,
        )


def _symbol_mask(ts: np.ndarray, start_ns: int | None, end_ns: int | None) -> np.ndarray:
    mask = np.ones(ts.shape[0], dtype=bool)
    if start_ns is not None:
        mask &= ts >= start_ns
    if end_ns is not None:
        mask &= ts < end_ns
    return mask


class BacktestEngine(Protocol):
    def run(
        self,
        positions: Mapping[str, np.ndarray],
        frame: pl.DataFrame,
        cost_model: CostModel,
        seed: int,
        *,
        notional_usd: float,
        cost_multiplier: float = 1.0,
    ) -> BacktestResult: ...


def _result_hash(
    ts: np.ndarray,
    series: Mapping[str, np.ndarray],
    per_symbol: Mapping[str, SymbolSeries],
    metrics: Mapping[str, float],
    header: Mapping[str, object],
) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json(dict(header)).encode())
    digest.update(np.ascontiguousarray(ts, dtype="<i8").tobytes())
    for name in SERIES_FIELDS:
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(series[name], dtype="<f8").tobytes())
    for sym in sorted(per_symbol):
        digest.update(sym.encode())
        digest.update(np.ascontiguousarray(per_symbol[sym].net, dtype="<f8").tobytes())
        digest.update(np.ascontiguousarray(per_symbol[sym].executed, dtype="<f8").tobytes())
    digest.update(canonical_json(dict(metrics)).encode())
    return "sha256:" + digest.hexdigest()


def _finalize(
    ts: np.ndarray,
    series: dict[str, np.ndarray],
    per_symbol: dict[str, SymbolSeries],
    cost_model_hash: str,
    notional_usd: float,
    cost_multiplier: float,
    periods_per_year: float,
    seed: int,
) -> BacktestResult:
    metrics = compute_metrics(series, periods_per_year)
    equity = np.cumprod(1.0 + series["net"]) if series["net"].size else np.ones(0)
    header = {
        "cost_model_hash": cost_model_hash,
        "notional_usd": notional_usd,
        "cost_multiplier": cost_multiplier,
        "periods_per_year": periods_per_year,
        "seed": seed,
    }
    return BacktestResult(
        ts=ts,
        series=series,
        per_symbol=per_symbol,
        metrics=metrics,
        equity=equity,
        cost_model_hash=cost_model_hash,
        notional_usd=notional_usd,
        cost_multiplier=cost_multiplier,
        periods_per_year=periods_per_year,
        seed=seed,
        result_hash=_result_hash(ts, series, per_symbol, metrics, header),
    )


class BarBacktestEngine:
    def __init__(self, periods_per_year: float = DEFAULT_PERIODS_PER_YEAR) -> None:
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        self.periods_per_year = float(periods_per_year)

    def run(
        self,
        positions: Mapping[str, np.ndarray],
        frame: pl.DataFrame,
        cost_model: CostModel,
        seed: int,
        *,
        notional_usd: float,
        cost_multiplier: float = 1.0,
    ) -> BacktestResult:
        """Account `positions` (symbol -> targets, one per bar of that symbol) on `frame`.

        The engine is deterministic; `seed` is recorded in the result and its hash only.
        """
        if not positions:
            raise ValueError("positions must contain at least one symbol")
        if notional_usd <= 0 or cost_multiplier < 0:
            raise ValueError("notional_usd must be > 0 and cost_multiplier >= 0")
        symbols = sorted(positions)
        sleeve = notional_usd / len(symbols)
        if sleeve > cost_model.capacity_notional_usd:
            raise CapacityExceededError(
                f"sleeve notional {sleeve:.2f} USD exceeds capacity"
                f" {cost_model.capacity_notional_usd:.2f} USD ({cost_model.version})"
            )
        per_symbol = {
            sym: self._symbol(sym, positions[sym], frame, cost_model, sleeve, cost_multiplier)
            for sym in symbols
        }
        ts = np.unique(np.concatenate([s.ts for s in per_symbol.values()]))
        series = {name: np.zeros(ts.shape[0]) for name in SERIES_FIELDS}
        for s in per_symbol.values():
            idx = np.searchsorted(ts, s.ts)
            for name in ("price", "funding", "cost", "gross", "net", "turnover"):
                series[name][idx] += getattr(s, name)
            series["exposure"][idx] += np.abs(s.held)
            series["trades"][idx] += (s.turnover > 0).astype(np.float64)
        for name in SERIES_FIELDS:
            if name != "trades":
                series[name] /= len(symbols)
        return _finalize(
            ts,
            series,
            per_symbol,
            sha256_of(cost_model),
            float(notional_usd),
            float(cost_multiplier),
            self.periods_per_year,
            int(seed),
        )

    @staticmethod
    def _symbol(
        symbol: str,
        target: np.ndarray,
        frame: pl.DataFrame,
        cost_model: CostModel,
        sleeve_notional: float,
        cost_multiplier: float,
    ) -> SymbolSeries:
        sub = frame.filter(pl.col("symbol") == symbol).sort("ts")
        n = sub.height
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (n,):
            raise ValueError(f"{symbol}: positions shape {target.shape} != ({n},)")
        if not np.all(np.isfinite(target)) or np.any(np.abs(target) > 1.0 + 1e-12):
            raise ValueError(f"{symbol}: positions must be finite and within [-1, 1]")
        ts = sub.get_column("ts").dt.epoch("ns").to_numpy().astype(np.int64)
        close = sub.get_column("perp_close").cast(pl.Float64).to_numpy()
        if n == 0 or not np.all(np.isfinite(close)) or np.any(close <= 0):
            raise ValueError(f"{symbol}: perp_close must be present, finite and positive")
        funding = sub.get_column("funding_rate").cast(pl.Float64).fill_null(0.0).to_numpy()
        funding = np.nan_to_num(funding, nan=0.0)
        lag = cost_model.latency_bars
        executed = np.zeros(n)
        if lag < n:
            executed[lag:] = target[: n - lag]
        held = np.zeros(n)
        held[1:] = executed[:-1]
        bar_return = np.zeros(n)
        bar_return[1:] = close[1:] / close[:-1] - 1.0
        price = held * bar_return
        funding_pnl = -held * funding if cost_model.apply_funding else np.zeros(n)
        turnover = np.abs(np.diff(executed, prepend=0.0))
        linear_bps = cost_model.taker_fee_bps + cost_model.half_spread_bps
        impact_bps = cost_model.impact_bps_per_million * turnover * sleeve_notional / 1e6
        cost = cost_multiplier * turnover * (linear_bps + impact_bps) / 1e4
        gross = price + funding_pnl
        return SymbolSeries(
            symbol=symbol,
            ts=ts,
            target=target,
            executed=executed,
            held=held,
            price=price,
            funding=funding_pnl,
            cost=cost,
            gross=gross,
            net=gross - cost,
            turnover=turnover,
        )
