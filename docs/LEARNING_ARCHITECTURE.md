# Enigma — Learning Architecture

**Status:** Draft for review  
**Product:** Enigma  
**Principle:** Three evidence-driven learning cycles. Self-modification is versioned, tested, auditable, and reversible. Agents never unilaterally change evaluation criteria, risk limits, security permissions, or validation requirements.

---

## 1. Overview

```text
┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│  Market learning │   │ Institutional learn. │   │ Technological learn. │
│  discover→edge   │   │ observe→improve org  │   │ discover→adopt tech  │
└────────┬─────────┘   └──────────┬───────────┘   └──────────┬───────────┘
         │                        │                          │
         └────────────────────────┼──────────────────────────┘
                                  ▼
                     Evidence store + version control
                     (experiments, KPIs, ADRs, rollbacks)
```

All cycles share: **measure → propose → evaluate independently → adopt/reject → re-measure**.

---

## 1.1 M1 binding

The first institutional proof wires a **thin** slice of market + institutional learning: external research → hypothesis → validate → reflection → lesson → next experiment (steps 3–12 in [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)). Full ImprovementProposal adoption with measured lift remains M4.

---

## 2. Market Learning Cycle

**Owner roles:** Intelligence, Data Eng, Microstructure, Quant Research, Strategy Eng, Independent Validation.

| Step | Output | Claim level |
|------|--------|-------------|
| Discover information | Source/Evidence nodes | L0 |
| Formulate hypothesis | Hypothesis with mechanism | L1 |
| Experiment | Experiment + research metrics | L2 candidate |
| Independently validate | ValidationReport | L3 iff PASS |
| Record findings | Registry updates + failures | durable |
| Improve understanding | Updated mechanism notes, regime links | versioned |

**Controls:**

- No skipping Independent Validation for “obvious” edges.
- Multiple-testing ledger increments on each hypothesis family test ([VALIDATION_PROTOCOL.md](./VALIDATION_PROTOCOL.md)).
- Research success does not imply deployability.

**State touchpoints:** Hypothesis and strategy FSMs in [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## 3. Institutional Learning Cycle

**Owner role:** Meta-Learning Agent, with CIO prioritization and Independent Validation / human gates for adoption.

| Step | Output |
|------|--------|
| Observe performance | Agent KPI snapshots, task failure taxonomies |
| Reflect | Structured reflections + systemic weakness tickets |
| Propose improvements | `ImprovementProposal` (prompt, tool, workflow, software) |
| Independently evaluate | Benchmark harness vs frozen baseline |
| Adopt or reject | Version bump + ADR if architectural |
| Measure subsequent performance | Pre-registered success metric |

**Forbidden auto-changes:**

- Risk limits
- Security permissions
- Validation gate definitions / thresholds
- An agent’s own KPI definitions
- Constitution text

**Rollback:** Every adoption records `previous_version` and automated revert path (config flag or package pin).

---

## 4. Technological Learning Cycle

**Owner role:** Technology & Innovation Agent → Meta-Learning / ADR board.

| Step | Output |
|------|--------|
| Discover innovations | Watchlist entries with citations |
| Evaluate relevance | Fit score vs Enigma requirements |
| Prototype | Sandboxed spike (time-boxed) |
| Benchmark | Against institutional baseline on fixed tasks |
| Independently validate | Reproducible benchmark + security review |
| Integrate | Only after demonstrated improvement |
| Monitor | Post-adoption regression KPIs |

**Challenge:** Reject “innovation theater.” No new infra without a benchmark that shows meaningful advantage on Enigma’s actual workloads (latency, cost, correctness, operability).

---

## 5. Learning State Machines

### 5.1 ImprovementProposal FSM

```text
PROPOSED → BENCHMARKING → APPROVED → ROLLED_OUT → (REVERTED)
                 ↘ REJECTED
```

### 5.2 InnovationProposal FSM

```text
WATCHLIST → SPIKE → BENCHMARKED → ADR_REVIEW → ACCEPTED_INTEGRATED | REJECTED
```

Transitions require typed events and evidence URIs.

---

## 6. Evidence Requirements

| Change type | Minimum evidence |
|-------------|------------------|
| Prompt / procedural memory | Blind paired eval on fixed task set; delta ≥ pre-registered ε |
| Research methodology | Replication on frozen datasets; Validation protocol still passes on controls |
| Validation gate change | Human + ADR; simulation of historical false pos/neg; **never** agent-only |
| Dependency / framework | Benchmark + complexity cost note + rollback plan |
| Risk limit change | Human risk owner only |

---

## 7. Feedback Without Contamination

- Paper/live performance summaries may flow to Quant Research as **aggregated** metrics.
- Raw holdout paths, timestamps of secret splits, and unblinding keys stay in validation plane.
- Meta-Learning may see gate fail rates, not holdout returns series, unless dual-control access is granted.

---

## 8. Reflection Schema (Minimum)

```yaml
task_id: uuid
agent_id: string
outcome: success | failure | aborted
what_worked: string
what_failed: string
suspected_systemic_issue: string | null
proposed_followup: string | null
created_at: datetime
```

Reflections feed Meta-Learning queues; they are not self-authorization to change systems.

---

## 9. Continuous Improvement KPIs (Institutional)

- Median time L1→validation decision
- Fraction of L2 candidates that fail independent validation (healthily high early)
- ImprovementProposal win rate with measured lift
- Rollback frequency post-adoption (lower is better after quality bar rises)
- Constitutional incident count (must be zero)

---

## 10. Open Uncertainties

1. Cadence of institutional learning reviews (weekly vs triggered).
2. Whether prompt changes require the same validation service as strategies or a lighter eval harness (recommend separate **agent-eval** harness with human audit sampling).
3. Budget share reserved for Tech Innovation spikes vs core research.
