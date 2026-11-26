# M1 acceptance mapping (AT1–AT16)

Each acceptance test from `IMPLEMENTATION_PLAN.md` maps to automated tests, plus the real
end-to-end run where it applies. Offline tests run in CI (`pytest -m "not network"`). Network
tests (`-m network`) hit real public APIs and run in the non-blocking `network-smoke` job.

Real-run reference: `ENIGMA_HOME` with Binance public archives (BTCUSDT and ETHUSDT 8h perp,
spot and funding, 2020-01-01 to the sealed research cutoff 2024-10-22T08:00Z), OpenAlex,
Crossref and arXiv research, and the real strategy sandbox and validation plane. Three cycles
reached `COMPLETED` with validation `FAIL`, which is the honest outcome for these mechanisms.

| AT | Criterion | Automated coverage | Real run |
|----|-----------|--------------------|----------|
| AT1 | Objective intake | `tests/unit/test_cycle_driver.py::test_full_cycle_reaches_all_steps` (CLI `enigma research` creates the cycle; first state is `OBJECTIVE_RECEIVED`, the cycle machine's name for `OBJECTIVE_SET`) | yes |
| AT2 | Delegation | `tests/unit/test_agents_runtime.py::test_delegation_is_permissioned_and_idempotent`; full-cycle test delegates 7 tasks to 7 roles under one cycle correlation id | yes |
| AT3 | Live research | `tests/network/test_live_research.py::test_live_research_returns_real_hashed_sources` | 30 sources, ~90 evidence items per cycle, each with `retrieved_at` and content hash |
| AT4 | Provenance graph | `tests/unit/knowledge/test_knowledge_base.py::test_orphan_evidence_is_rejected` | yes |
| AT5 | Hypothesis quality | `tests/unit/test_schemas.py::test_hypothesis_requires_mechanism_and_cannot_claim_l3` | yes |
| AT6 | Dataset publish | `tests/unit/data/test_data_catalog.py` (`test_acquire_publishes_verified_version`, `test_same_data_same_hash_in_independent_roots`, integrity tests) | 10,538-row PIT-certified dataset |
| AT7 | Sandbox isolation | `tests/validation_gates/test_at7_sandbox_isolation.py` (static, runtime audit hook and network namespace) | 9 sandboxed trials per cycle |
| AT8 | Reproducibility | `tests/validation_gates/test_at8_at9_reproducibility.py::test_at8_twin_backtests_identical`; every real trial runs twice and compares results | yes |
| AT9 | Validation determinism | `test_at8_at9_reproducibility.py::test_at9_twin_validation_reports_identical` | — |
| AT10 | Leakage fail | `tests/validation_gates/test_validation_service.py::test_leaky_package_fails_g01_at10`; `tests/unit/strategy/test_mechanism_templates.py` proves the shipped templates pass truncation invariance | — |
| AT11 | No LLM gate | `tests/validation_gates/test_at11_no_llm_gate.py`; import-linter contract "validation cannot import gateway" | — |
| AT12 | Reflection and lesson | `test_full_cycle_reaches_all_steps` (lessons derive from the cycle's evidence and validation summary) | 3 lessons per cycle |
| AT13 | Next experiment | `test_full_cycle_reaches_all_steps` (proposal carries the parent cycle and lesson ids); `tests/unit/test_daemon.py::test_daemon_chains_next_experiment_up_to_daily_cap` | a child cycle was chained automatically |
| AT14 | Resume | `test_cycle_driver.py::test_resume_after_process_restart_does_not_repeat_work`, `::test_orphaned_running_task_is_requeued_and_completed`; `test_agents_runtime.py::test_bus_idempotency_and_conversation` | SIGKILL during the intelligence task, then `enigma resume` completed all 13 steps |
| AT15 | Plane secrets | `tests/validation_gates/test_at15_plane_isolation.py`, `tests/unit/test_plane_isolation.py`, `tests/unit/test_store.py::test_plane_view_blocks_research_from_validation_reports` | single host, see ADR-015 |
| AT16 | Manual demo | `tests/unit/test_dashboard.py` renders every page | dashboard walkthrough recording of a real cycle |

## Known gaps

- AT15 is enforced inside one process by plane views, tool contracts and import contracts;
  separate OS users or hosts per plane are future work (ADR-015).
- Sandbox CPU time is reported per experiment but not yet charged to the cost ledger.
