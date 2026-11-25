# M5 specification — plane isolation across hosts

**Status:** proposed. This document does not implement isolation and does not authorize a deploy.
**Depends on:** [ADR-015](../adr/ADR-015-sqlite-control-store-and-holdout-custody.md),
[ADR-016](../adr/ADR-016-strategy-sandbox.md), import-linter contracts in `pyproject.toml`,
`PlaneView` in `src/enigma/store/documents.py`.

## Problem

ADR-015 keeps the control registry and the holdout store in separate SQLite files on one host.
Research is blocked from `ValidationReport` by the repository view and by import-linter.
That boundary is process- and filesystem-level. A user on the same machine can still open the
holdout file. M5 is the proposal to move that boundary onto separate hosts.

## Required properties

1. The validation host is the only process that can open `$ENIGMA_VALIDATION_DIR`.
2. The research host receives `RedactedValidationSummary` and the research cutoff. It does not
   receive holdout rows, the holdout manifest body, or a path to the validation database.
3. The execution host, when it exists, reads risk clearance documents. It does not read the
   research trial ledger or the holdout file.
4. Credentials are per host. A research token cannot authenticate to the validation store.
5. Strategy code stays inside the ADR-016 sandbox: no network, no path outside the package.
6. Clocks used for `available_at` checks are recorded. A host skew larger than the embargo
   fails the run instead of widening the research window.

## Non-goals

- No Docker Compose requirement in this spec. A process boundary with separate users and
  separate machines satisfies it. Containers are one implementation, not the requirement.
- No change to protocol `0.1.0-slice` or to the opt-in `1.0.0` pin.
- No live orders and no paper session on the research host.

## Dependencies before implementation

| Dependency | Why it blocks implementation |
|---|---|
| ADR-015 single-writer SQLite | A second host cannot share the control file. M5 needs a transport, not a copied database. |
| Holdout seal in the validation service | The cutoff must stay a validation output. Research must not compute it. |
| Import-linter contracts | They must keep forbidding `enigma.validation` from importing research, execution, the model gateway, and `enigma.decisions`. |
| Human approval of the host layout | An agent must not choose production hosts or credentials. |

## Acceptance

A reviewer can show, on two machines, that a research process denied the validation credential
cannot read a holdout row, and that a redacted summary still validates a cycle. Until that
run exists, M5 is not implemented. Writing this file does not mark it implemented.
