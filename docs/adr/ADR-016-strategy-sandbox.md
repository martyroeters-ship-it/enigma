# ADR-016 — Strategy sandbox: subprocess + network namespace + rlimits

**Status:** Accepted (implementation team, 2026-11-25)

## Context
Generated strategy code must run isolated with no network and bounded resources (CONSTITUTION §2.6,
AT7). Docker is not available on the M1 host.

## Decision
Strategy packages execute in a child Python process launched as:
- `unshare --user --map-root-user --net` → empty network namespace (no interfaces besides a
  down loopback). Verified on the host: outbound `connect()` fails with `ENETUNREACH`.
- `prlimit`/`resource.setrlimit`: CPU seconds, address space, open files, file size, no core
  dumps; wall-clock timeout enforced by the parent.
- `python -I -S` (isolated mode), scrubbed environment (no secrets), working dir is a fresh temp
  directory containing only the package source and the input data slice (Parquet).
- An in-process audit hook additionally denies `socket.*`, `subprocess.*`, `os.system`, and
  writes outside the temp dir, and reports violations to the parent for the audit log.
- If `unshare` is unavailable the runner **fails closed** (refuses to execute) unless the
  operator explicitly sets `ENIGMA_SANDBOX_ALLOW_WEAK=1` for local development, which is logged.

## Alternatives
- Docker / gVisor / Firecracker: stronger, preferred when available; the runner interface keeps
  this swappable.
- RestrictedPython: too limiting for NumPy-based strategy code, and not a security boundary.

## Limitations
- Same kernel and same user as the host; this is a strong accident and network barrier, not a
  hardened multi-tenant boundary.
