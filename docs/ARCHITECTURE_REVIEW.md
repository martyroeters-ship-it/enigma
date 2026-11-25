# Enigma — Architecture Review

**Status:** Critical self-review after foundational specs  
**Product:** Enigma  
**Repo:** `martyroeters-ship-it/enigma`  
**Audience:** User + coordinator + implementation team  
**Date:** 2026-11-25 (Europe/Istanbul)

---

## Verdict

Specs are **ready to drive M0/M1 implementation** with a clear 13-step autonomous vertical slice, workstream contracts, and constitutional isolation. Several open decisions remain intentional. Docs were corrected where early drafts contradicted the authorized M1 slice or over-deferred required components.

**Approve next:** (1) ratify Constitution + ADR-002/007 naming, (2) lock ADR-001 (uv) and ADR-003 (backtest engine) via short spikes, (3) continue M1 against [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) acceptance tests AT1–AT16.

---

## Review highlights (for briefing)

1. **Product is Enigma** (`martyroeters-ship-it/enigma`); institutional substance of the original directive retained.
2. **M1 = 13-step autonomous slice** (objective → … → resume), not a thin backtest-only demo — normative in IMPLEMENTATION_PLAN §4.
3. **Process success ≠ alpha success** (ADR-007): runs can COMPLETE with validation FAIL; avoids pressure to soft-pass gates.
4. **Validation is code-only**; LLM cannot set `overall` (AT11). Protocol `0.1.0-slice` is a declared subset, not silent watering-down.
5. **Orchestration = FSM + Postgres outbox** (ADR-002); multi-agent frameworks rejected for M1.
6. **Eleven workstreams** with interface contracts and dependency order; critical path named.
7. **Fixed contradictions:** M1 agents now include Intelligence + Meta-Learning; M1 packages include knowledge/comm; gate list aligned across Validation Protocol ↔ Implementation Plan; Run FSM added beside strategy FSM.
8. **Main technical risks:** holdout isolation shortcuts, backtest engine realism vs speed, ToS/egress for “real external research,” multiple-testing honesty under agent autonomy.
9. **Over-engineering consciously deferred:** Neo4j, pgvector, K8s, dual engines, Crawl4AI+Playwright, always-on 12 LLMs, Next.js blocking M1.
10. **Open for human decision:** uv vs Poetry; VectorBT vs custom; separate validation DB vs schema; venue choice; soft thresholds in `0.1.0-slice` YAML.

---

## Contradictions found & fixed

| Issue | Resolution |
|-------|------------|
| Early AGENTS.md M1 roster omitted Intelligence & Meta-Learning required by 13-step slice | Roster updated (§7) |
| ARCHITECTURE §8 told a 6-step mini-story conflicting with authorized 13-step slice | Replaced with pointer to IMPLEMENTATION_PLAN §4 + Run FSM |
| ARCHITECTURE “MVP packages” omitted `enigma-knowledge` / `enigma-comm` | Corrected to M1 critical set |
| VALIDATION_PROTOCOL listed G01–G04 for slice; IMPLEMENTATION_PLAN listed G01,G02,G03,G06,G07,G12,G16 | Protocol aligned to plan; G04 deferred to M2 |
| Risk of treating validation FAIL as run failure | ADR-007 + AT wording: process COMPLETE allowed |
| Repo naming “proposed” vs implementation already started | Docs cite `martyroeters-ship-it/enigma` |
| Knowledge schemas lacked Reflection/Lesson/NextExperiment | Added §4.7 |

---

## Technical weaknesses (still real)

1. **Threshold vacuum:** DSR/PBO/OOS floors not numerically set — implementers must not invent “PASS” thresholds without human-owned YAML.
2. **Holdout hardness vs solo-dev cost:** schema-only isolation is weaker; compensating tests help but are not equal to separate instances.
3. **“Real external research” flakiness:** live HTTP makes CI non-deterministic; need allowlisted URL + recorded content_hash / VCR policy without turning step 3 into pure fixtures.
4. **Strategy Engineering as human+tools** can bottleneck autonomy; acceptable for M1 but must not become a permanent silent dependency.
5. **Funding/perp realism** may exceed VectorBT defaults — ADR-003 spike is blocking for honest G12.
6. **Redaction policy** for ValidationReport → research agents still open (leakage via narrative summaries).
7. **Resume semantics:** idempotency keys must be specified per side effect (fetch, dataset publish, package build) or step 13 will double-spend budgets.

---

## Over-engineering risks (watch list)

| Temptation | Why resist |
|------------|------------|
| Standing up all 12 LLM personas | Contracts + on-demand sessions suffice |
| Graph DB + vector memory before retrieval metrics | SQL registry first |
| Temporal/LangGraph before outbox pain is felt | ADR-002 |
| Full gate suite + paper + portfolio in one milestone | Splits M1/M2/M3 deliberately |
| Perfect Next.js console before AT14 resume works | CLI timeline OK |

---

## Missing requirements (documented, not silently filled)

- Numeric validation thresholds  
- Venue + historical data ToS choice  
- Exact embargo function of horizon  
- Legal crawl policy beyond allowlist  
- Multi-operator IAM (single-operator assumed)  
- Paper congruence mandatory before any future live (recommended yes, not ratified)  
- Portfolio vs single-strategy paper policy for M2  

---

## Unnecessary complexity (rejected)

LangGraph-first orchestration; Neo4j; Kubernetes for M1; dual backtest engines; mandatory Crawl4AI+Playwright; MLflow parallel to registry; PyTorch/RL as default discovery path.

---

## Significant technical risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Cross-plane secret bleed | False L3 / integrity failure | AT15, separate creds, plane boot checks |
| Agent-driven multiple testing without ledger | False discoveries | Mandatory ExperimentRegistered before run |
| Sandbox escape | Host compromise | AT7, no network in strategy profile |
| Soft gates under demo pressure | Institutional credibility loss | ADR-007, AT11, human-owned protocol YAML |
| Data vendor gaps / funding mis-timestamps | Garbage hypotheses | PIT gates G01; DataIntegrityAlert |

---

## Vertical slice summary

**M1 proves the institution can run itself through research→validation→learning hooks**, with resume, provenance, sandboxing, and programmatic validation — on BTC/ETH — without live capital.

Workstreams: WS1–WS11 in IMPLEMENTATION_PLAN; critical path WS1 → (WS10∥WS2∥WS5) → WS3 → WS4 → WS6 → WS7 → WS8.

Cursor rules drafts ready to copy into `.cursor/rules/` ([CURSOR_RULES.md](./CURSOR_RULES.md)).

---

## What to approve next

1. **Constitution** as binding for the repo (copy + ratify).  
2. **ADR-002, ADR-004, ADR-005, ADR-007, ADR-010** as accepted defaults.  
3. **Spike + decide ADR-001 (uv) and ADR-003 (engine)** within M0.  
4. **Write `protocol_0.1.0-slice.yaml`** with explicit soft/hard thresholds (human).  
5. **Venue/data source** for BTC/ETH history.  
6. Continue implementation against **AT1–AT16**; do not expand to M2 until M1 exits.

---

## Spec inventory (store)

All under `/cursor/stores/bc-1382f43c-6017-49dc-b96a-c8171948182f/docs/` — see coordinator handoff for absolute paths. No platform code written by this specs agent.
