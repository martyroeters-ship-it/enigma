"""Runtime settings. Values come from environment variables (`ENIGMA_*`) with safe defaults.

Secrets are never stored in settings objects that get logged or hashed; providers read their
own credentials from the environment at call time (see `enigma.gateway`).
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    home: Path = Field(description="Root directory for all Enigma state")
    research_dir: Path
    validation_dir: Path
    execution_dir: Path
    db_path: Path
    registries_dir: Path
    max_daily_api_spend_usd: float = 0.0
    http_user_agent: str = "Enigma-Research/0.1 (+https://github.com/martyroeters-ship-it/enigma)"

    @classmethod
    def from_env(cls) -> Settings:
        home = _env_path("ENIGMA_HOME", Path.cwd() / ".enigma")
        repo_root = Path(__file__).resolve().parents[3]
        return cls(
            home=home,
            research_dir=_env_path("ENIGMA_RESEARCH_DIR", home / "research"),
            validation_dir=_env_path("ENIGMA_VALIDATION_DIR", home / "validation"),
            execution_dir=_env_path("ENIGMA_EXECUTION_DIR", home / "execution"),
            db_path=_env_path("ENIGMA_DB_PATH", home / "control" / "enigma.sqlite3"),
            registries_dir=_env_path("ENIGMA_REGISTRIES_DIR", repo_root / "registries"),
            max_daily_api_spend_usd=float(os.environ.get("ENIGMA_MAX_DAILY_API_SPEND_USD", "0")),
        )

    def ensure_dirs(self) -> None:
        for path in (
            self.research_dir,
            self.validation_dir,
            self.execution_dir,
            self.db_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)
