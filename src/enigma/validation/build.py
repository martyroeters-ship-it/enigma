"""Identify the validator build that produced a report."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from functools import cache
from importlib import metadata
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SRC_ROOT.parents[1]
_HASHED_PACKAGES = ("validation", "backtest", "strategy")


def validation_source_hash() -> str:
    digest = hashlib.sha256()
    for package in _HASHED_PACKAGES:
        for path in sorted((_SRC_ROOT / package).rglob("*.py")):
            digest.update(path.relative_to(_SRC_ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@cache
def validator_build_sha() -> str:
    """`git:<sha>` of the checkout if available, else `enigma-<version>+src.<hash>`."""
    git = shutil.which("git")
    if git is not None and (_REPO_ROOT / ".git").exists():
        try:
            done = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [git, "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            done = None
        if done is not None and done.returncode == 0 and done.stdout.strip():
            return f"git:{done.stdout.strip()}"
    try:
        version = metadata.version("enigma")
    except metadata.PackageNotFoundError:
        version = "unknown"
    return f"enigma-{version}+src.{validation_source_hash()[:16]}"
