# ADR-012 — Thin custom bar-based backtest engine for M1 (resolves ADR-003)

**Status:** Accepted (implementation team, 2026-11-25) · **Resolves:** ADR-003

## Context
ADR-003 required choosing VectorBT **or** a thin custom engine behind a `BacktestEngine`
protocol. The M1 hypothesis class is funding/basis structure on BTC/ETH perpetuals at 8h/1h bars.

## Decision
A thin, deterministic, NumPy-based engine (`enigma.backtest`) behind a `BacktestEngine` protocol:
- Positions are decided from information at bar *t* and filled at bar *t + latency_bars*.
- Explicit per-bar accounting of price PnL, **funding payments at the actual funding timestamps**,
  taker fees, half-spread, and a linear impact proxy, all from the versioned `CostModel`.
- Capacity: notional above `capacity_notional_usd` is rejected, not silently clipped.
- Results are hashed (`result_hash`) for the reproducibility gate G16.

## Evidence / rationale
- Funding is a cash flow at discrete timestamps on the *position held at that time*; VectorBT
  has no native perpetual funding model, so it would need custom callbacks anyway, which removes
  its main advantage.
- A ~300-line engine is fully auditable and unit-testable against hand-computed PnL (golden
  tests in `tests/`), which validation needs more than raw speed at M1 data sizes
  (≈10⁴ bars per instrument).
- Avoids a numba/LLVM dependency tree in the sandbox image.

## Alternatives
- VectorBT: fast, but funding and cost realism must be bolted on; heavier dependency.
- NautilusTrader: realistic event-driven simulation; overkill for bar-level M1 and heavy to
  sandbox. Re-evaluate for M2 paper trading (order-book-aware execution).

## Limitations
- Bar-level fills only: no queue position, no intrabar path, no partial fills.
- Linear impact is a proxy; Market Microstructure must calibrate it against paper fills in M2.
