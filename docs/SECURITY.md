# Enigma — Security & Governance

**Status:** Draft for review  
**Product:** Enigma  
**Related:** [CONSTITUTION.md](./CONSTITUTION.md) · [VALIDATION_PROTOCOL.md](./VALIDATION_PROTOCOL.md)

---

## 1. Objectives

Protect scientific integrity, capital (when enabled), credentials, and holdout confidentiality. Fail closed. Prefer simple enforceable controls over elaborate unmonitored policy text.

---

## 2. Security Domains

| Domain | Controls |
|--------|----------|
| Plane isolation | Separate networks, identities, secrets, storage prefixes |
| Data governance | Claim levels; holdout custody; PIT integrity |
| Code integrity | Sandbox; signed/hashed packages; no unrestricted exec |
| Identity & access | Least privilege; role allowlists; human owners for risk/validation protocol |
| Audit & detection | Append-only logs; denial events; anomaly on cross-plane access attempts |
| Operational resilience | Kill switch; rollback; budget hard stops |

---

## 3. Plane Isolation Model

```text
research.*     → research DB + research object store + research API keys
validation.*   → validation DB + holdout store + validation-only keys
execution.*    → execution DB + paper broker keys (no live keys in initial phase)
```

**Rules:**

1. Workers carry explicit `ENIGMA_PLANE` env; mis-set plane refuses to start.
2. Cross-plane data movement only via **promotion APIs** (artifact hash + metadata), never raw DB mounts.
3. Shared Postgres for *control plane registry* is allowed if row-level/column policies prevent holdout series storage there; **holdout market data** must not live in the research catalog. Prefer separate validation datastore for M1+.
4. CI runs isolation tests (research role cannot read validation secrets).

---

## 4. Secrets

- Per-plane secret namespaces (e.g. Docker secrets / vault paths).
- No secrets in git, Cursor rules, or knowledge embeddings.
- Rotation runbook required before any live keys exist (future).
- Initial phase: **no production exchange trading keys** in any plane.

---

## 5. Sandboxing

| Workload | Network | FS | Limits |
|----------|---------|----|--------|
| Strategy backtest | Deny all | Package + data mount RO | CPU, mem, wall time |
| Generated tool code | Deny all default | Scratch RW | Same |
| Intelligence fetch | Allowlist egress | Scratch RW | Budgeted requests |
| Data ingestion | Allowlist venues/APIs | Dataset RW in research store | Rate limits |

Sandbox escape = P0 incident.

---

## 6. Audit Log (minimum fields)

`at, actor, plane, action, resource, correlation_id, decision(allow|deny), reason_code`

Retain online ≥ 12 months; cold thereafter. Audit log is append-only; agents have no delete permission.

---

## 7. Validation & Risk Governance

- Validation protocol YAML: human-controlled; PR + ADR to change.
- Risk limits registry: human-controlled; Risk Agent may tighten temporarily, not permanently loosen.
- Kill switch: authorized human + Risk Agent; halts execution plane order path.

---

## 8. LLM / Agent Abuse Cases

| Abuse | Mitigation |
|-------|------------|
| Prompt asks to skip gates | Tool layer has no skip; rules + code enforce |
| Exfiltrate holdout via summary | Redact; aggregate-only research views |
| Infinite cost loops | Hard budgets on session |
| Supply-chain deps | Pin versions; review new packages in Tech Innovation flow |

---

## 9. Emergency Procedures

1. Activate kill switch → execution order adapter no-ops  
2. Freeze FSM promotions to paper/live  
3. Preserve audit + artifacts  
4. Incident review before resume  

---

## 10. M1 Security Acceptance

- AT7 sandbox network deny  
- AT15 cross-plane secret deny  
- Audit events for FSM + tool deny  
- No live trading keys in compose files  

---

## 11. Open Uncertainties

1. Control-plane DB shared vs fully split (cost vs hardness).  
2. Multi-tenant future (single-operator assumed now).  
3. Legal/ToS compliance process for crawling before enabling broad Intelligence egress.
