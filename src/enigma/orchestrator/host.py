"""Host resource readings for the research scheduler. Unknown readings do not pause work."""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path


def memavailable_mb(meminfo: str) -> float | None:
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[1]) / 1024.0
    return None


def available_memory_mb() -> float | None:
    system = platform.system()
    if system == "Linux":
        try:
            return memavailable_mb(Path("/proc/meminfo").read_text(encoding="utf-8"))
        except OSError:
            return None
    if system == "Darwin":
        return _darwin_available_mb()
    return None


def _darwin_available_mb() -> float | None:
    binary = shutil.which("vm_stat")
    if binary is None:
        return None
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [binary],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    page = 4096
    counts: dict[str, int] = {}
    for line in done.stdout.splitlines():
        if "page size of" in line:
            tail = line.split("page size of", 1)[-1]
            digits = "".join(ch for ch in tail if ch.isdigit())
            if digits:
                page = int(digits)
        if ":" not in line:
            continue
        label, raw = line.split(":", 1)
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits:
            counts[label.strip().lower()] = int(digits)
    free = counts.get("pages free", 0) + counts.get("pages inactive", 0)
    if free <= 0:
        return None
    return free * page / (1024.0 * 1024.0)
