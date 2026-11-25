# Enigma — Quantitative Methodology

**Status:** Draft for review  
**Product:** Enigma  
**Related:** [VALIDATION_PROTOCOL.md](./VALIDATION_PROTOCOL.md) · [KNOWLEDGE_ARCHITECTURE.md](./KNOWLEDGE_ARCHITECTURE.md)

---

## 1. Research Posture

Enigma investigates market inefficiencies with **falsifiable mechanisms**, not indicator combinations optimized until the backtest looks good.

Two discovery modes (both required eventually; slice starts with reasoning-driven):

| Mode | Description | Contamination risk |
|------|-------------|-------------------|
| Reasoning-driven | Mechanism → prediction → test | Lower if features pre-registered |
| Data-driven search | Computational discovery over feature/model space | High multiple-testing; requires strict ledger + deflation |

**No methodology is assumed profitable before Independent Validation.**

---

## 2. Initial Market Scope

- **Assets:** BTC, ETH
- **Instruments:** spot and perpetual futures
- **Phase constraint:** no live capital; paper only after validation + risk clearance
- Extensibility: schemas carry `instrument_id`, `venue`, `asset_class` so equities/tradfi can join later without rewriting FSMs

---

## 3. Supported Investigation Domains

Prioritized for early phases (P0 = slice-relevant):

| Domain | Priority | Notes |
|--------|----------|-------|
| Funding-rate anomalies | P0 | Natural crypto perp mechanism |
| Basis / cash-and-carry structure | P0 | Spot vs perp |
| Momentum / mean reversion (explicit horizon) | P1 | Easy to overfit; gates must be strict |
| Volatility modelling | P1 | |
| Market regime detection | P1 | Input to portfolio later |
| Order flow / liquidity / microstructure | P2 | Needs richer data |
| Liquidation dynamics | P2 | |
| Cross-exchange discrepancies | P2 | Multi-venue ops cost |
| Derivatives positioning | P2 | |
| On-chain intelligence | P2 | Separate data plane |
| Macro liquidity / information diffusion / behavioural | P3 | |
| ML / evolutionary / RL search | P3 | Only after harness + multiple-testing controls mature |

---

## 4. Hypothesis Standard

A registered hypothesis **must** include:

1. **Mechanism** — structural/causal story (who pays whom, why it persists or decays)
2. **Prediction** — directional/statistical claim with horizon
3. **Instruments & venues**
4. **Falsification criteria** — what result kills it
5. **Pre-registered features / parameters** (or explicit search space + budget)
6. **DatasetVersion** IDs
7. **Cost model version** assumptions
8. **Multiple-testing family ID** (which search family this draw belongs to)

Hypotheses without mechanisms are rejected at CIO intake for M1+ (may be logged as L0 notes only).

---

## 5. Experiment Design Baseline

### 5.1 Data hygiene

- Point-in-time joins only; no future stamps in features
- Train / validation / test chronological separation in research plane
- Embargo between sets where label horizons overlap
- Survivorship and listing/delisting notes on DatasetVersion
- Explicit handling of funding payment times, insurance funds, fee tiers

### 5.2 Research-plane tests (do **not** confer L3)

- In-sample fit diagnostics
- Chronological validation split metrics
- Simple sensitivity sweeps (logged; inflate trial count)
- Benchmark comparison (buy-hold, funding-neutral flat, naive momentum)

### 5.3 Cost & capacity (always on)

Every backtest applies versioned assumptions for:

- Fees (maker/taker)
- Slippage model (at least spread + impact proxy)
- Funding payments
- Latency buffer where relevant
- Notional capacity cap (reject strategies that need impossible liquidity)

---

## 6. Strategy Package Interface

Strategies expose a narrow interface (implemented later in `enigma-research` / packages):

```text
configure(config: StrategyConfig) -> None
on_data(bar_or_event: MarketEvent) -> list[OrderIntent]
required_data_schema() -> SchemaRef
metadata() -> StrategyMeta  # hypothesis_id, version, cost_model_version
```

- Deterministic given seed + data
- No network I/O inside strategy code
- No reading wall clock for signals (injection only via event time)

---

## 7. Modelling Stack Guidance

| Technique | When allowed |
|-----------|--------------|
| Linear / penalized linear, simple state filters | Default first |
| Tree models (e.g. LightGBM) | After leakage tests; feature timestamps audited |
| Deep learning / RL | Deferred; requires dedicated compute budget + stricter PBO |
| Evolutionary search | Deferred; trial ledger mandatory |

Prefer interpretable baselines that share the same validation gates as complex models. Complexity must beat the baseline on **independent** validation, not research vanity metrics.

---

## 8. Regime Awareness

- Tag experiments with realized regime labels **computed only from information available at decision time** in research; validation re-labels with its own PIT rules.
- A strategy that only works in one rare regime must declare capacity and expected time-to-opportunity; Validation may fail on fragility.

---

## 9. Anti-Patterns (Rejected)

- Optimizing indicators until Sharpe maximizes
- Report-only LLM “this looks robust”
- Dropping failed trials from the registry
- Using test-set feedback to tweak features without new hypothesis ID
- Cross-exchange alpha that ignores transfer latency and fees

---

## 10. Open Uncertainties

1. First backtest engine: VectorBT vs thin custom event-driven (see ADR-003).
2. Single-venue vs multi-venue for funding/basis slice.
3. Exact embargo length as function of horizon (formula TBD in Validation Protocol params registry).
4. Whether on-chain data enters before or after multi-venue microstructure.
