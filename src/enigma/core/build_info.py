"""Code provenance for experiment records (reproducibility: code SHA + config hash)."""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

import enigma

_REPO_ROOT = Path(__file__).resolve().parents[3]


@functools.cache
def code_sha() -> str:
    """Git commit of the running code, suffixed `-dirty` when the tree has local changes."""
    try:
        sha = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(  # noqa: S603
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return f"enigma-{enigma.__version__}-nogit"
