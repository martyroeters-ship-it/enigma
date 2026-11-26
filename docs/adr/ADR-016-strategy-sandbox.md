# ADR-016 — Strategy sandbox: subprocess + network namespace + rlimits

**Status:** Accepted (implementation team, 2026-11-25)

## Context
Generated strategy code must run isolated with no network and bounded resources (CONSTITUTION §2.6,
AT7). The original M1 host had no Docker daemon, so the first implementation is Linux `unshare`.

## Amendment (2026-11-26) — macOS backend
macOS cannot create a Linux user and network namespace. On a Mac, the same child bootstrap runs
in a locally built Linux image, `enigma-sandbox:py312`, with `--network none`, a read-only root
filesystem, all capabilities dropped, no new privileges, pid/cpu/memory/file limits, and a single
bind mount of the temporary strategy directory. The Docker socket, host credentials, the
institution home, and sealed validation data are not mounted. Ollama, the dashboard, and the
orchestrator stay on macOS. `uv run enigma sandbox prepare` builds the image once; a research
cycle does not pull or build.

Linux hosts that can `unshare` keep that backend. `ENIGMA_SANDBOX_ALLOW_WEAK` is not a startup
mode. `enigma up`, `enigma research`, `enigma run`, and `enigma resume` refuse to start unless
the selected backend is `netns` or `docker`. An isolation failure is infrastructure: code
generation is not retried.

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
- If `unshare` is unavailable the runner tries the Docker backend in the amendment above, then
  **fails closed**. `ENIGMA_SANDBOX_ALLOW_WEAK=1` remains an explicit logged escape for tests.
  Institution startup does not accept it.

## Alternatives
- gVisor / Firecracker: stronger, and still swappable behind the runner interface. Docker is the
  macOS backend; a Linux host that can `unshare` does not use it.
- RestrictedPython: too limiting for NumPy-based strategy code, and not a security boundary.

## Limitations
- Same kernel and same user as the host; this is a strong accident and network barrier, not a
  hardened multi-tenant boundary.
