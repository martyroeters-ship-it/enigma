# ADR-013 — Conservative hard thresholds for protocol `0.1.0-slice`

**Status:** Accepted as default (implementation team under user-delegated authority,
2026-11-25). Human may tighten or replace via a new protocol version.

## Context
VALIDATION_PROTOCOL §11 and ARCHITECTURE_REVIEW flagged a "threshold vacuum" for G03/G07 and
asked for `registries/validation/protocol_0.1.0-slice.yaml`. The user authorized the team to pick
conservative defaults.

## Decision
`registries/validation/protocol_0.1.0-slice.yaml` defines required gates
**G01, G02, G03, G06, G07, G12, G16**, all **hard**:

| Gate | Threshold |
|------|-----------|
| G01 PIT | 0 violations |
| G02 split | last 30% sealed holdout; embargo ≥ max(21 bars, 3 × label horizon) |
| G03 OOS | holdout annualized net Sharpe ≥ 0.5 over ≥ 270 bars; PnL > 0 after removing best 5 days |
| G06 multiple testing | Holm–Bonferroni over all registered trials in the family, α = 0.05 |
| G07 DSR | Deflated Sharpe probability ≥ 0.95 (Bailey & López de Prado, 2014) |
| G12 costs | net Sharpe > 0 with all costs × 2 |
| G16 reproducibility | twin-run metric delta ≤ 1e-9 |

The validation service pins the YAML's sha256; an edited file without a code-level version bump
fails closed.

## Rationale
- No soft gates: ADR-007 already lets the *process* complete when the *strategy* fails, so
  there is no incentive to soften thresholds to "finish" M1.
- DSR ≥ 0.95 and Holm correction are standard, conservative false-discovery controls.
- Sharpe 0.5 net after costs is a low bar for an edge claim but high enough to reject noise
  given ≥ 90 days of holdout.

## Consequences
Most early hypotheses are expected to FAIL. That is the intended, honest outcome.

## Limitations
Annualization assumes 8h bars (1095/yr); hypotheses on other horizons must set their own
`annualization_periods_per_year` via a new protocol version, not ad hoc.
