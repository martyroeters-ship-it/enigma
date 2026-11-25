"""Sandbox child bootstrap. Copied into the sandbox directory and run as a script.

Runs under `python -I -S -B` inside an empty network namespace with rlimits applied by the
parent. Imports only the standard library and NumPy. After NumPy is imported it installs a
permanent audit hook that denies network, process, FFI, and file access outside the sandbox
directory, and reports each denial to `violations.jsonl` through a pre-opened descriptor.

Exit codes: 0 ok, 1 strategy error, 2 bootstrap error, 3 policy violation.
"""

from __future__ import annotations

import contextlib
import json
import os
import resource
import sys
import traceback
from typing import Any

BOX = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(BOX, "pkg"), os.path.join(BOX, "lib")]

import numpy as np  # noqa: E402

DENIED_EVENT_PREFIXES = (
    "socket.",
    "subprocess.",
    "os.exec",
    "os.spawn",
    "os.posix_spawn",
    "os.fork",
    "os.forkpty",
    "os.kill",
    "os.killpg",
    "os.system",
    "os.putenv",
    "os.unsetenv",
    "os.chdir",
    "os.symlink",
    "os.link",
    "os.chmod",
    "os.chown",
    "os.truncate",
    "pty.",
    "ctypes.",
    "mmap.",
    "shutil.",
    "urllib.",
    "http.",
    "ftplib.",
    "smtplib.",
    "telnetlib.",
    "webbrowser.",
    "sys._getframe",
    "sys.setprofile",
    "sys.settrace",
    "sys.addaudithook",
    "builtins.input",
    "code.__new__",
    "function.__new__",
)
DENIED_MODULES = frozenset(
    {
        "socket",
        "_socket",
        "ssl",
        "_ssl",
        "select",
        "selectors",
        "asyncio",
        "subprocess",
        "_posixsubprocess",
        "multiprocessing",
        "_multiprocessing",
        "ctypes",
        "_ctypes",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "pty",
        "shutil",
        "pickle",
        "marshal",
    }
)
PATH_EVENTS = frozenset({"os.listdir", "os.scandir", "os.remove", "os.rename", "os.rmdir"})
PATH_EVENTS_WRITE = frozenset({"os.remove", "os.rename", "os.rmdir", "os.mkdir"})

_state: dict[str, Any] = {"in_hook": False, "fd": -1, "read_roots": (), "write_root": BOX}


def _within(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(root + os.sep) for root in roots)


def _real(path: object) -> str | None:
    if isinstance(path, int):
        return None
    raw = os.fsdecode(os.fspath(path))  # type: ignore[call-overload]
    return os.path.realpath(raw if os.path.isabs(raw) else os.path.join(BOX, raw))


def _deny(event: str, detail: str) -> None:
    record = json.dumps({"event": event, "detail": detail[:300]}, sort_keys=True) + "\n"
    with contextlib.suppress(OSError):
        os.write(_state["fd"], record.encode())
    raise PermissionError(f"sandbox denied {event}: {detail[:120]}")


def _is_write(mode: object, flags: object) -> bool:
    if isinstance(mode, str) and any(c in mode for c in "wax+"):
        return True
    if isinstance(flags, int):
        return bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    return False


def _hook(event: str, args: tuple[Any, ...]) -> None:
    if _state["in_hook"]:
        return
    _state["in_hook"] = True
    try:
        _check(event, args)
    finally:
        _state["in_hook"] = False


def _check(event: str, args: tuple[Any, ...]) -> None:
    if event == "open":
        path = _real(args[0])
        if path is None:
            return
        mode = args[1] if len(args) > 1 else "r"
        flags = args[2] if len(args) > 2 else 0
        if _is_write(mode, flags):
            if not _within(path, (_state["write_root"],)):
                _deny(event, f"write {path}")
        elif not _within(path, _state["read_roots"]):
            _deny(event, f"read {path}")
        return
    if event in PATH_EVENTS or event == "os.mkdir":
        path = _real(args[0]) if args and args[0] is not None else BOX
        roots = (_state["write_root"],) if event in PATH_EVENTS_WRITE else _state["read_roots"]
        if path is not None and not _within(path, roots):
            _deny(event, str(path))
        return
    if event == "import":
        top = str(args[0]).split(".")[0]
        if top in DENIED_MODULES:
            _deny(event, str(args[0]))
        return
    if event.startswith(DENIED_EVENT_PREFIXES):
        _deny(event, repr(args)[:200])


def _write_json(name: str, payload: dict[str, Any]) -> None:
    with open(os.path.join(BOX, name), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)


def _run(spec: dict[str, Any], inputs: Any) -> dict[str, Any]:
    module = __import__(spec["module"])
    cls = getattr(module, spec["entrypoint"])
    out_dir = os.path.join(BOX, "out")

    def fresh(params: dict[str, Any]) -> Any:
        instance = cls()
        instance.configure(params)
        return instance

    probe = fresh(spec["base_params"])
    columns = tuple(str(c) for c in probe.required_columns())
    meta = probe.metadata()
    report: dict[str, Any] = {
        "required_columns": list(columns),
        "meta": {"name": str(meta.name), "version": str(meta.version)},
    }
    for job in spec["jobs"]:
        data = {c: inputs[f"{job['dataset']}/{c}"] for c in columns}
        params = job["params"] if job["params"] is not None else spec["base_params"]
        full = np.asarray(fresh(params).target_positions(data), dtype=np.float64)
        np.save(os.path.join(out_dir, f"{job['id']}.npy"), full, allow_pickle=False)
        for n in job["truncations"]:
            sliced = {c: arr[:n] for c, arr in data.items()}
            part = np.asarray(fresh(params).target_positions(sliced), dtype=np.float64)
            np.save(os.path.join(out_dir, f"{job['id']}__t{n}.npy"), part, allow_pickle=False)
    return report


def main() -> int:
    try:
        with open(os.path.join(BOX, "spec.json"), encoding="utf-8") as handle:
            spec = json.load(handle)
        os.makedirs(os.path.join(BOX, "out"), exist_ok=True)
        _state["read_roots"] = tuple(os.path.realpath(r) for r in [BOX, *spec["read_roots"]])
        _state["write_root"] = os.path.realpath(BOX)
        _state["fd"] = os.open(
            os.path.join(BOX, "violations.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
        )
        # Load inputs and exercise np.save before the hook so NumPy's lazy imports are done.
        inputs = np.load(os.path.join(BOX, "inputs.npz"), allow_pickle=False)
        np.save(os.path.join(BOX, "out", "_warmup.npy"), np.zeros(1), allow_pickle=False)
    except Exception:
        traceback.print_exc()
        return 2
    sys.addaudithook(_hook)
    try:
        report = _run(spec, inputs)
    except BaseException as exc:
        _write_json(
            "result.json",
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]},
        )
        traceback.print_exc()
        return 3 if isinstance(exc, PermissionError) else 1
    report["ok"] = True
    report["cpu_seconds"] = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    _write_json("result.json", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
