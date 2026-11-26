"""Strategy sandbox runner (ADR-016).

Each run launches one child process:

    unshare --user --map-root-user --net -- python -I -S -B _bootstrap.py

in a fresh temporary directory that contains only the bootstrap, the package source, a copy of
the strategy interface, read-only symlinks to NumPy, and an `.npz` of input arrays. The parent
applies rlimits (CPU, address space, open files, file size, no core) and a wall-clock timeout
and scrubs the environment. Inside, an audit hook denies network, process, FFI and out-of-box
file access and reports each denial. Positions come back as `.npy` files.

If `unshare` cannot create the namespaces, and this host is macOS, the same child
runs in a Linux container with `--network none` (see `sandbox_docker`). The runner
still fails closed when neither backend is available. `ENIGMA_SANDBOX_ALLOW_WEAK=1`
remains an explicit, audited escape for tests; the institution will not start on it.
"""

from __future__ import annotations

import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from enigma.core.errors import EnigmaError
from enigma.core.ids import new_id
from enigma.schemas.research import StrategyPackage
from enigma.store.audit import AuditLog
from enigma.strategy import interface as _interface
from enigma.strategy.interface import ParamValue
from enigma.strategy.package import verify_package
from enigma.strategy.sandbox_docker import (
    docker_backend_status as _docker_backend_status,
)
from enigma.strategy.sandbox_docker import (
    isolation_unavailable_message,
    spawn_docker,
)
from enigma.strategy.static_check import StrategyRejectedError, check_source

ALLOW_WEAK_ENV = "ENIGMA_SANDBOX_ALLOW_WEAK"
_CHILD_SOURCE = Path(__file__).with_name("_sandbox_child.py")
_INTERFACE_SOURCE = Path(_interface.__file__)
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_POSITION_TOLERANCE = 1e-12
_STDERR_TAIL = 4000


class SandboxUnavailableError(EnigmaError):
    """Neither a Linux network namespace nor a checked Docker sandbox is available."""


class SandboxExecutionError(EnigmaError):
    """The strategy failed, violated policy, or produced invalid output inside the sandbox."""


@dataclass(frozen=True)
class SandboxLimits:
    cpu_seconds: int = 120
    address_space_bytes: int = 4 * 1024**3
    open_files: int = 64
    file_size_bytes: int = 512 * 1024**2
    wall_seconds: float = 180.0


@dataclass(frozen=True)
class SandboxJob:
    """One `target_positions` call on `dataset`, plus re-runs on `dataset[:n]` per truncation."""

    job_id: str
    dataset: str
    params: Mapping[str, ParamValue] | None = None
    truncations: tuple[int, ...] = ()


@dataclass(frozen=True)
class SandboxResult:
    run_id: str
    package_id: str
    package_hash: str
    ok: bool
    exit_code: int
    positions: dict[str, np.ndarray]
    truncated: dict[str, dict[int, np.ndarray]]
    cpu_seconds: float
    wall_seconds: float
    violations: tuple[dict[str, str], ...]
    stderr_tail: str
    isolation: str
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def require_ok(self) -> SandboxResult:
        if not self.ok:
            raise SandboxExecutionError(
                f"sandbox run {self.run_id} failed (exit {self.exit_code}): {self.error}"
            )
        return self


@cache
def _netns_available() -> bool:
    unshare = shutil.which("unshare")
    if unshare is None:
        return False
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [unshare, "--user", "--map-root-user", "--net", "--", "/bin/true"],
            capture_output=True,
            timeout=10,
            check=False,
            env={"PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _numpy_read_roots() -> list[str]:
    numpy_dir = Path(np.__file__).resolve().parent
    roots = [str(numpy_dir)]
    libs = numpy_dir.parent / "numpy.libs"
    if libs.is_dir():
        roots.append(str(libs.resolve()))
    for key in ("stdlib", "platstdlib"):
        roots.append(str(Path(sysconfig.get_paths()[key]).resolve()))
    return sorted(set(roots))


def _datasets_to_npz(
    datasets: Mapping[str, Mapping[str, np.ndarray]], columns: Sequence[str]
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, data in datasets.items():
        if not _JOB_ID_RE.match(name):
            raise ValueError(f"invalid dataset name {name!r}")
        lengths = {len(np.asarray(data[c])) for c in columns}
        if len(lengths) != 1:
            raise ValueError(f"dataset {name!r} columns have unequal lengths")
        for column in columns:
            arr = np.ascontiguousarray(np.asarray(data[column]))
            if arr.dtype.kind not in "fiub":
                raise ValueError(f"column {column!r} must be numeric, got {arr.dtype}")
            arrays[f"{name}/{column}"] = arr
    return arrays


class SandboxRunner:
    def __init__(
        self,
        limits: SandboxLimits | None = None,
        audit: AuditLog | None = None,
        actor: str = "strategy_sandbox",
        allow_weak: bool | None = None,
    ) -> None:
        self.limits = limits or SandboxLimits()
        self.audit = audit
        self.actor = actor
        self.allow_weak = (
            os.environ.get(ALLOW_WEAK_ENV) == "1" if allow_weak is None else allow_weak
        )

    def _record(self, action: str, subject: str, details: dict[str, Any]) -> None:
        if self.audit is not None:
            self.audit.record(self.actor, action, subject, details)

    def check_isolation(self, subject_id: str) -> str:
        """Return `netns`, `docker`, or `weak`. Raise when no real backend is available.

        Linux namespaces win when they work. Docker is the macOS backend. Weak mode
        is only the explicit `ENIGMA_SANDBOX_ALLOW_WEAK=1` escape and is not offered
        as the remedy.
        """
        if _netns_available():
            return "netns"
        ready, detail = _docker_backend_status()
        if ready:
            return "docker"
        if self.allow_weak:
            self._record(
                "sandbox.weak_mode",
                subject_id,
                {"reason": "no netns or docker sandbox", "env": ALLOW_WEAK_ENV, "detail": detail},
            )
            return "weak"
        self._record(
            "sandbox.denied", subject_id, {"reason": "isolation_unavailable", "detail": detail}
        )
        raise SandboxUnavailableError(isolation_unavailable_message(detail))

    def run(
        self,
        package: StrategyPackage,
        datasets: Mapping[str, Mapping[str, np.ndarray]],
        jobs: Sequence[SandboxJob],
        *,
        static_precheck: bool = True,
    ) -> SandboxResult:
        """Execute `jobs` for `package` in one sandboxed child process.

        `static_precheck=False` exists only so tests can exercise the runtime barrier with
        source that the AST pre-check would already reject; the runtime policy is unchanged.
        """
        verify_package(package)
        if static_precheck:
            violations = check_source(package.source_code, package.entrypoint)
            if violations:
                self._record(
                    "sandbox.denied",
                    package.id,
                    {
                        "reason": "static_precheck",
                        "package_hash": package.content_hash,
                        "violations": [v.describe() for v in violations],
                    },
                )
                raise StrategyRejectedError(violations)
        for column in package.required_columns:
            if not _COLUMN_RE.match(column):
                raise ValueError(f"invalid column name {column!r}")
        for job in jobs:
            if not _JOB_ID_RE.match(job.job_id) or job.dataset not in datasets:
                raise ValueError(f"invalid job {job.job_id!r} / dataset {job.dataset!r}")
        if len({job.job_id for job in jobs}) != len(jobs):
            raise ValueError("job ids must be unique")
        isolation = self.check_isolation(package.id)
        run_id = new_id("sbx")
        with tempfile.TemporaryDirectory(prefix="enigma-sbx-") as parent_dir:
            result = self._execute(package, datasets, jobs, Path(parent_dir), run_id, isolation)
        self._record(
            "sandbox.run",
            package.id,
            {
                "run_id": run_id,
                "package_hash": package.content_hash,
                "ok": result.ok,
                "exit_code": result.exit_code,
                "isolation": isolation,
                "jobs": len(jobs),
                "cpu_seconds": round(result.cpu_seconds, 3),
                "wall_seconds": round(result.wall_seconds, 3),
                "violations": len(result.violations),
                "error": result.error,
            },
        )
        for violation in result.violations:
            self._record(
                "sandbox.denied",
                package.id,
                {"run_id": run_id, "package_hash": package.content_hash, **violation},
            )
        return result

    def _prepare_box(
        self,
        box: Path,
        package: StrategyPackage,
        datasets: Mapping[str, Mapping[str, np.ndarray]],
        jobs: Sequence[SandboxJob],
        isolation: str,
    ) -> None:
        pkg = box / "pkg"
        (pkg / "enigma" / "strategy").mkdir(parents=True)
        (pkg / "enigma" / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "enigma" / "strategy" / "__init__.py").write_text("", encoding="utf-8")
        shutil.copyfile(_INTERFACE_SOURCE, pkg / "enigma" / "strategy" / "interface.py")
        (pkg / "strategy_module.py").write_text(package.source_code, encoding="utf-8")
        lib = box / "lib"
        lib.mkdir()
        # The Docker image has its own Linux NumPy. A macOS wheel cannot be mounted in.
        if isolation != "docker":
            numpy_dir = Path(np.__file__).resolve().parent
            (lib / "numpy").symlink_to(numpy_dir, target_is_directory=True)
            if (numpy_dir.parent / "numpy.libs").is_dir():
                (lib / "numpy.libs").symlink_to(numpy_dir.parent / "numpy.libs")
        shutil.copyfile(_CHILD_SOURCE, box / "_bootstrap.py")
        arrays: dict[str, Any] = _datasets_to_npz(datasets, package.required_columns)
        np.savez(box / "inputs.npz", **arrays)
        spec = {
            "module": "strategy_module",
            "entrypoint": package.entrypoint,
            "base_params": dict(package.params),
            "jobs": [
                {
                    "id": job.job_id,
                    "dataset": job.dataset,
                    "params": None if job.params is None else dict(job.params),
                    "truncations": [int(n) for n in job.truncations],
                }
                for job in jobs
            ],
            "read_roots": [] if isolation == "docker" else _numpy_read_roots(),
            "discover_read_roots": isolation == "docker",
        }
        (box / "spec.json").write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")

    def _preexec(self) -> None:
        limits = self.limits
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
        resource.setrlimit(
            resource.RLIMIT_AS, (limits.address_space_bytes, limits.address_space_bytes)
        )
        resource.setrlimit(resource.RLIMIT_NOFILE, (limits.open_files, limits.open_files))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.file_size_bytes, limits.file_size_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def _execute(
        self,
        package: StrategyPackage,
        datasets: Mapping[str, Mapping[str, np.ndarray]],
        jobs: Sequence[SandboxJob],
        parent_dir: Path,
        run_id: str,
        isolation: str,
    ) -> SandboxResult:
        box = parent_dir / "box"
        box.mkdir()
        self._prepare_box(box, package, datasets, jobs, isolation)
        start = time.monotonic()
        if isolation == "docker":
            exit_code, timed_out = spawn_docker(
                box,
                parent_dir,
                name_seed=run_id,
                wall_seconds=self.limits.wall_seconds,
                cpu_seconds=self.limits.cpu_seconds,
                memory_bytes=self.limits.address_space_bytes,
                open_files=self.limits.open_files,
                file_size_bytes=self.limits.file_size_bytes,
            )
            cpu = 0.0
        else:
            python = str(Path(sys.base_prefix) / "bin" / "python3")
            if not Path(python).exists():
                python = sys.executable
            argv = [python, "-I", "-S", "-B", str(box / "_bootstrap.py")]
            if isolation == "netns":
                unshare = shutil.which("unshare") or "unshare"
                argv = [unshare, "--user", "--map-root-user", "--net", "--", *argv]
            env = {
                "PATH": "/usr/bin:/bin",
                "LANG": "C.UTF-8",
                "HOME": str(box),
                "TMPDIR": str(box),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            exit_code, cpu, timed_out = self._spawn(argv, box, env, parent_dir)
        wall = time.monotonic() - start
        stderr_tail = (
            (parent_dir / "stderr.log").read_bytes()[-_STDERR_TAIL:].decode("utf-8", "replace")
        )
        violations = self._read_violations(box)
        error: str | None = None
        meta: dict[str, Any] = {}
        positions: dict[str, np.ndarray] = {}
        truncated: dict[str, dict[int, np.ndarray]] = {}
        result_path = box / "out" / "result.json"
        if timed_out:
            error = f"wall-clock limit {self.limits.wall_seconds}s exceeded"
        elif exit_code < 0:
            error = f"terminated by signal {-exit_code} (resource limit?)"
        elif not result_path.exists():
            error = f"no result (exit {exit_code})"
        else:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            reported = payload.get("cpu_seconds")
            if isolation == "docker" and isinstance(reported, int | float):
                cpu = float(reported)
            if not payload.get("ok"):
                error = str(payload.get("error", "unknown error"))
            else:
                meta = {k: payload[k] for k in ("required_columns", "meta") if k in payload}
                error = self._collect(box, package, datasets, jobs, positions, truncated, meta)
        if violations and error is None:
            error = f"{len(violations)} sandbox policy violation(s)"
        ok = error is None and exit_code == 0 and not violations
        return SandboxResult(
            run_id=run_id,
            package_id=package.id,
            package_hash=package.content_hash,
            ok=ok,
            exit_code=exit_code,
            positions=positions if ok else {},
            truncated=truncated if ok else {},
            cpu_seconds=cpu,
            wall_seconds=wall,
            violations=violations,
            stderr_tail=stderr_tail,
            isolation=isolation,
            error=error,
            meta=meta,
        )

    def _spawn(
        self, argv: list[str], box: Path, env: dict[str, str], parent_dir: Path
    ) -> tuple[int, float, bool]:
        """Run the child to completion or wall timeout. Returns (exit code, cpu s, timed out)."""
        timed_out = False
        with (
            (parent_dir / "stdout.log").open("wb") as out,
            (parent_dir / "stderr.log").open("wb") as err,
        ):
            proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                argv,
                cwd=box,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                preexec_fn=self._preexec,  # noqa: PLW1509 - rlimits must apply before exec
                start_new_session=True,
                close_fds=True,
            )
            deadline = time.monotonic() + self.limits.wall_seconds
            while True:
                pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
                if pid == proc.pid:
                    break
                if time.monotonic() > deadline:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGKILL)
                    _, status, usage = os.wait4(proc.pid, 0)
                    break
                time.sleep(0.005)
            proc.returncode = os.waitstatus_to_exitcode(status)
        return int(proc.returncode), float(usage.ru_utime + usage.ru_stime), timed_out

    @staticmethod
    def _read_violations(box: Path) -> tuple[dict[str, str], ...]:
        path = box / "violations.jsonl"
        if not path.exists():
            return ()
        found: list[dict[str, str]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                found.append({"event": "unparseable", "detail": line[:200]})
                continue
            found.append({"event": str(item.get("event")), "detail": str(item.get("detail"))})
        return tuple(found)

    @staticmethod
    def _collect(
        box: Path,
        package: StrategyPackage,
        datasets: Mapping[str, Mapping[str, np.ndarray]],
        jobs: Sequence[SandboxJob],
        positions: dict[str, np.ndarray],
        truncated: dict[str, dict[int, np.ndarray]],
        meta: dict[str, Any],
    ) -> str | None:
        declared = tuple(meta.get("required_columns", ()))
        if declared != tuple(package.required_columns):
            return f"strategy required_columns {declared} != package {package.required_columns}"
        out = box / "out"
        for job in jobs:
            n = len(np.asarray(datasets[job.dataset][package.required_columns[0]]))
            arrays: dict[int | None, Path] = {None: out / f"{job.job_id}.npy"}
            arrays.update({t: out / f"{job.job_id}__t{t}.npy" for t in job.truncations})
            parts: dict[int, np.ndarray] = {}
            for trunc, path in arrays.items():
                expected = n if trunc is None else min(trunc, n)
                arr = np.load(path, allow_pickle=False)
                problem = _check_positions(arr, expected)
                if problem is not None:
                    return f"job {job.job_id} truncation {trunc}: {problem}"
                if trunc is None:
                    positions[job.job_id] = arr
                else:
                    parts[trunc] = arr
            truncated[job.job_id] = parts
        return None


def _check_positions(arr: np.ndarray, expected: int) -> str | None:
    if arr.ndim != 1 or arr.shape[0] != expected:
        return f"positions shape {arr.shape} != ({expected},)"
    if not np.all(np.isfinite(arr)):
        return "positions contain NaN/inf"
    if np.any(np.abs(arr) > 1.0 + _POSITION_TOLERANCE):
        return "positions outside [-1, 1]"
    return None
