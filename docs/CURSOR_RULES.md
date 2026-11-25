# Enigma — Cursor Rules Overview

**Status:** Draft for copy into repo  
**Product:** Enigma  
**Target path in repo:** `.cursor/rules/`  
**Source drafts:** [cursor-rules/](./cursor-rules/)

---

## Purpose

These rules make the institutional constitution and engineering standards binding inside Cursor coding sessions for `martyroeters-ship-it/enigma`, so agents and humans do not “conveniently” bypass validation, isolation, or scientific integrity.

---

## Rule files (copy map)

| Store draft | Repo destination | Enforces |
|-------------|------------------|----------|
| [cursor-rules/constitution.mdc](./cursor-rules/constitution.mdc) | `.cursor/rules/constitution.mdc` | Absolute prohibitions, isolation, no live capital |
| [cursor-rules/validation-gates.mdc](./cursor-rules/validation-gates.mdc) | `.cursor/rules/validation-gates.mdc` | Programmatic validation only; no LLM pass/fail |
| [cursor-rules/security.mdc](./cursor-rules/security.mdc) | `.cursor/rules/security.mdc` | Planes, secrets, sandbox, audit |
| [cursor-rules/engineering-standards.mdc](./cursor-rules/engineering-standards.mdc) | `.cursor/rules/engineering-standards.mdc` | Typing, reproducibility, no fake ops claims |
| [cursor-rules/agent-contracts.mdc](./cursor-rules/agent-contracts.mdc) | `.cursor/rules/agent-contracts.mdc` | Authority bounds, budgets, typed comms |
| [cursor-rules/implementation-scope.mdc](./cursor-rules/implementation-scope.mdc) | `.cursor/rules/implementation-scope.mdc` | M1 slice focus; anti-overbuild |

---

## Application guidance

- Prefer **always-on** for constitution, validation-gates, security, engineering-standards.
- Apply **agent-contracts** when editing `packages/enigma-agents/**`, `registries/agents/**`, or comm schemas.
- Apply **implementation-scope** when scaffolding features or proposing new infra.

Rules complement, do not replace, docs in this folder. On conflict: Constitution > Security > Validation Protocol > ADRs > rules wording.

---

## Maintenance

- Rule changes that weaken gates or isolation require a human ADR.
- After ratification of docs, sync rule summaries if section numbers drift.
- Do not embed secrets or exchange credentials in rules.
