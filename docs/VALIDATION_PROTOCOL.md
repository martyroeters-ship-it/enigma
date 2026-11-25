# Enigma — Validation Protocol

**Status:** Draft for review  
**Product:** Enigma  
**Authority:** Programmatic gates only — LLM commentary cannot change pass/fail  
**Related:** [CONSTITUTION.md](./CONSTITUTION.md) · [QUANTITATIVE_METHODOLOGY.md](./QUANTITATIVE_METHODOLOGY.md)

---

## 1. Purpose

Independent Validation exists to **invalidate** strategies: catch overfitting, leakage, fragile assumptions, and false discoveries before paper/live capital.

Outcomes are produced by the **validation service** (`enigma-validation`). The Independent Validation Agent may explain reports; it may not override gate results.

---

## 2. Isolation Rules

| Rule | Enforcement |
|------|-------------|
| Holdout data inaccessible from research plane | Separate credentials / network / bucket |
| Validation workers cannot push to research feature stores | IAM deny |
| Research cannot read ValidationReport detail that reveals holdout path structure beyond pass/fail aggregates | API field filters (open: exact redaction policy) |
| Gate thresholds versioned in `registries/validation/protocol_vX.yaml` | Change requires human ADR — agents cannot edit |

---

## 3. Protocol Versioning

Every `ValidationReport` cites `protocol_version`. Changing thresholds or gate set:

1. Human opens ADR
2. Historical false pos/neg simulation on frozen packages
3. Dual approval (Risk + technical owner)
4. New semver; old reports remain under old version

---

## 4. Mandatory Gate Suite (v0 draft)

All gates return `PASS` | `FAIL` with machine-readable stats. **Overall PASS iff all mandatory gates PASS.**

| Gate ID | Name | Intent |
|---------|------|--------|
| G01 | PIT integrity | No future timestamps in features/labels vs event time |
| G02 | Chronological split | Train/val/test order respected; embargo applied |
| G03 | Out-of-sample | Holdout performance meets floor **and** is not driven by ≤N days |
| G04 | Walk-forward | Rolling retrain/fold stability within bounds |
| G05 | Purged CV | Where overlapping labels: purged CV + embargo |
| G06 | Multiple-testing | Family-wise correction using trial ledger |
| G07 | Deflated Sharpe | DSR above threshold given trial count |
| G08 | PBO | Probability of backtest overfitting below ceiling |
| G09 | Parameter sensitivity | Neighborhood of params does not cliff-fail |
| G10 | Regime stress | Failure concentration check across regimes |
| G11 | Bootstrap / MC | Interval / null comparison as configured |
| G12 | Costs | Fees, slippage, funding applied; net metric used for G03–G08 |
| G13 | Liquidity / capacity | Strategy respects declared notional caps |
| G14 | Latency (if relevant) | Signal-to-order delay model applied |
| G15 | Benchmark dominance | Beats pre-registered benchmarks on net metric (or justified non-dominance rule) |
| G16 | Reproducibility | Re-run with same seeds/hashes matches within tolerance |

**M1 note:** Protocol `0.1.0-slice` enables **G01, G02, G03 (smoke), G06, G07, G12, G16** only (see [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) §4.3). G04 and other gates are deferred to M2. Missing full-suite gates are FAIL-closed for **promotion beyond research/paper eligibility**; the M1 *process* may still COMPLETE with strategy FAIL (ADR-007). Soft vs hard numeric thresholds for G03/G07 must be declared in `registries/validation/protocol_0.1.0-slice.yaml` before claiming gate maturity.

---

## 5. Trial Ledger & Multiple Testing

- Every research experiment increments a **trial counter** in its hypothesis family.
- Search spaces declare estimated trial budget up front; excess trials tighten DSR/PBO.
- Deleted or “forgotten” trials are still counted if configs were evaluated (honor system replaced by mandatory experiment registration before run).

---

## 6. Cost Model

`cost_model_version` hashes:

- Fee schedule per venue
- Slippage function + parameters
- Funding payment rules
- Minimum liquidity assumptions

Validation **re-applies** costs; it does not trust research-reported net PnL.

---

## 7. Live Paper Trading Gate (post-M1)

After overall PASS:

- Optional `G17 Paper congruence`: paper results vs backtest within band over T days
- Degradation detector feeds Risk for pause

Not required for M1 slice completion (slice ends at independent validation + reflection + next experiment proposal).

---

## 8. ValidationReport Schema

See [KNOWLEDGE_ARCHITECTURE.md](./KNOWLEDGE_ARCHITECTURE.md) §4.5. Additional requirements:

- `gates[].stat` and `gates[].threshold` always present
- `overall` computed by pure function `aggregate(gates)`; unit-tested
- `validator_build_sha` + `protocol_version` + `holdout_dataset_ids` required
- Human-readable summary optional and non-authoritative

---

## 9. Adversarial Checks (Process)

Validators actively look for:

- Look-ahead / leakage
- Overfit parameter islands
- Unrealistic fills
- Hidden multiple testing (unregistered sweeps)
- Regime cherry-picking
- Capacity fantasy

Findings attach `FailureCondition` nodes; packages remain failed until a **new** package ID is submitted.

---

## 10. Acceptance Criteria for the Validation Service Itself

- Identical inputs → identical `overall` and gate results (determinism test)
- Mutated future-leak feature → G01 FAIL
- Unregistered trial inflation → G06/G07 tighten or FAIL
- No code path where LLM output sets `overall`
- Research credentials cannot read holdout object paths (integration test)

---

## 11. Open Uncertainties

1. Exact numeric thresholds for DSR/PBO/OOS floors (must be set in protocol YAML before claiming L3).
2. Redaction level of ValidationReport fields visible to research agents.
3. Whether paper congruence is mandatory before any future live capital (recommend **yes**).
4. Bootstrap methodology choice (stationary vs circular blocks for returns).
