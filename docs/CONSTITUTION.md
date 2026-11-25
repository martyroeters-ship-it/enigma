# Enigma — Institutional Constitution

**Status:** Draft for review  
**Scope:** Immutable governance principles for all agents, services, and human operators  
**Authority:** Overrides agent instructions, prompts, and operational convenience  

This constitution is the highest-priority institutional document. No agent, workflow, model, or operator may override it through prompt engineering, configuration drift, or unilateral software change.

---

## 1. Purpose

Enigma exists to discover, validate, and (eventually) deploy genuine statistical trading edges under institutional standards of scientific rigor, capital preservation, and engineering reliability.

It is **not**:

- A retail trading bot
- An LLM signal generator
- A technical-indicator dashboard
- A wrapper that presents model confidence as evidence

Optimize for sustainable, risk-adjusted, **net** returns across changing regimes — never for narrative sophistication or backtest aesthetics.

---

## 2. Absolute Prohibitions

No agent, service, or human acting through the platform may:

1. **Bypass risk controls** — including soft limits, hard limits, circuit breakers, or kill switches.
2. **Override failed validation gates** — a strategy that fails Independent Validation remains non-deployable.
3. **Fabricate evidence** — no invented data, metrics, citations, experiment results, or provenance.
4. **Self-modify permissions** — agents cannot expand their own authority, budgets, or data access.
5. **Access restricted datasets without authorization** — especially holdout/evaluation sets and production credentials.
6. **Execute unrestricted generated code** — all agent-generated code runs in sandboxes with resource limits.
7. **Access production trading credentials from research environments** — research, validation, and execution are isolated.
8. **Deploy unapproved production changes** — software and strategy promotion require explicit approval gates.
9. **Allocate real capital without explicit authorization** — initial phase: **no live capital deployment**.
10. **Exceed assigned computational or financial budgets** — hard enforcement, not advisory warnings alone.
11. **Unilaterally change evaluation criteria, risk limits, security permissions, or validation requirements**.
12. **Present placeholder or mock functionality as operational capability**.

Violations are security incidents. Systems must fail closed.

---

## 3. Scientific Integrity

1. Historical profitability ≠ future profitability.
2. Model/LLM confidence ≠ empirical evidence.
3. Complexity ≠ intelligence.
4. Validation criteria are **programmatic**, never discretionary LLM judgment.
5. Failed experiments are retained permanently in the research registry (selection-bias accounting).
6. Every claim of edge must cite: dataset version, methodology version, validation report ID, and uncertainty.
7. Distinguish explicitly among:
   - Unverified information
   - Theoretical hypotheses
   - Historically validated statistical relationships
   - Independently validated live market edges

---

## 4. Isolation Doctrine

Three environments with **no shared secrets** and **controlled data flow**:

| Plane | Purpose | Capital | Credentials |
|-------|---------|---------|-------------|
| **Research** | Discovery, hypothesis, experimentation | None | Research-only data APIs |
| **Validation** | Adversarial evaluation, holdouts | None | Holdout/eval secrets only |
| **Execution** | Paper (initial) then live (future) | Paper only initially | Execution vault only |

Research may propose artifacts. Validation may accept/reject. Execution may run only approved artifacts. Risk Management holds independent **veto** over deployment.

---

## 5. Authority Hierarchy

1. This Constitution  
2. Security & Risk policies (`SECURITY.md`, risk limit registries)  
3. Validation Protocol (`VALIDATION_PROTOCOL.md`)  
4. Architectural Decisions (ADRs)  
5. Agent contracts and versioned methodologies  
6. Runtime prompts and operational configs  

Lower layers cannot contradict higher layers. Conflict resolution defaults to the higher layer and **fail closed**.

---

## 6. Change Control for the Constitution

- Constitution changes require **human approval** and a documented ADR.
- Agents may propose constitutional amendments; they may never enact them.
- Prior versions remain archived and immutable.
- Emergency operational freezes (kill switch) do not require constitutional amendment; they are mandated by Section 2.

---

## 7. Initial Market & Capital Posture

- **Markets (Phase 0–1):** BTC and ETH spot + perpetual futures.
- **Capital:** No live capital deployment in the initial development phase.
- **Paper trading:** Allowed only after Independent Validation accepts a strategy under the Validation Protocol.
- Architecture must remain extensible to additional instruments and asset classes without rewriting isolation boundaries.

---

## 8. Accountability

Every consequential decision must produce:

- Actor (agent ID / human ID)
- Timestamp (UTC)
- Inputs / artifact versions
- Decision outcome
- Rationale code or structured reason (not free-form only)
- Audit log entry (append-only)

Absence of an audit trail invalidates the decision for promotion purposes.

---

## 9. Non-Negotiable Engineering Standards

- Strong typing at trust boundaries
- Explicit interfaces between agents and services
- Reproducible experiments (seed, data version, code SHA, config hash)
- Deterministic math where feasible; stochastic methods must record seeds
- Comprehensive automated tests for validation gates and risk enforcement
- Provider-independent model and exchange integrations
- No fabricated operational claims in docs, UI, or agent speech

---

## 10. Ratification

This draft is **pending human ratification**. Until ratified, treat as binding design intent for all specification and Cursor rule work. After ratification, treat as immutable law until formally amended.
