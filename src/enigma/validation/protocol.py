"""Validation protocol loader. Registry YAML files are human-owned and pinned by sha256.

Editing a protocol YAML without a new version and a matching pin change here makes the
validation service refuse to run.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from enigma.core.errors import IntegrityError

_REG_DIR = Path(__file__).resolve().parents[3] / "registries" / "validation"

# Default remains the M1 slice until operators explicitly opt into 1.0.0.
PROTOCOL_VERSION = "0.1.0-slice"
PROTOCOL_SHA256 = "ccaa3d0f26b8474320b8be8e518c2a3abfed687afb523018782fdbde6f7efc8c"
DEFAULT_PROTOCOL_PATH = _REG_DIR / "protocol_0.1.0-slice.yaml"

PROTOCOL_REGISTRY: dict[str, tuple[Path, str]] = {
    "0.1.0-slice": (
        _REG_DIR / "protocol_0.1.0-slice.yaml",
        "ccaa3d0f26b8474320b8be8e518c2a3abfed687afb523018782fdbde6f7efc8c",
    ),
    "1.0.0": (
        _REG_DIR / "protocol_1.0.0.yaml",
        "cc80810e833b08f599afc847eb77361c23bb42c28ee3311de3d7c81dc476a785",
    ),
    # Sampling-frequency protocols. They do not replace the sealed 8h protocol.
    "0.2.0-1h": (
        _REG_DIR / "protocol_0.2.0-1h.yaml",
        "c2cadcd397b958c4fc9dc9f07717fd2c1307fedfaffa7bfa743822b9bb0e46c7",
    ),
    "0.2.0-1d": (
        _REG_DIR / "protocol_0.2.0-1d.yaml",
        "62e7293626fb59ec9f32816e58c0e1c9193e599cfede6033cc3a095980b45947",
    ),
}

REQUIRED_KEYS = ("protocol_version", "required_gates", "annualization_periods_per_year", "gates")


@dataclass(frozen=True)
class ValidationProtocol:
    version: str
    required_gates: tuple[str, ...]
    periods_per_year: float
    gates: dict[str, dict[str, Any]]
    sha256: str

    def gate(self, gate_id: str) -> dict[str, Any]:
        try:
            return self.gates[gate_id]
        except KeyError as exc:
            raise IntegrityError(f"protocol {self.version} has no gate {gate_id}") from exc


def resolve_protocol_version(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("ENIGMA_VALIDATION_PROTOCOL", "").strip()
    return env or PROTOCOL_VERSION


def load_protocol(
    path: Path | None = None,
    expected_sha256: str | None = None,
    *,
    version: str | None = None,
) -> ValidationProtocol:
    """Load and verify a protocol. Fails closed on any hash or structure mismatch."""
    chosen = resolve_protocol_version(version)
    if path is None:
        if chosen not in PROTOCOL_REGISTRY:
            raise IntegrityError(f"unsupported validation protocol {chosen!r}")
        target, pinned = PROTOCOL_REGISTRY[chosen]
        expected = expected_sha256 or pinned
    else:
        target = path
        # When a path is supplied (tests), default pin is the slice pin unless overridden.
        expected = expected_sha256 or PROTOCOL_SHA256
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise IntegrityError(f"validation protocol unreadable: {target.name}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected:
        raise IntegrityError(
            f"validation protocol {target.name} sha256 {digest} does not match pinned"
            f" {expected}; protocol changes need a new version and pin"
        )
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or any(k not in data for k in REQUIRED_KEYS):
        raise IntegrityError("validation protocol is missing required keys")
    file_version = str(data["protocol_version"])
    if path is None and file_version != chosen:
        raise IntegrityError(f"protocol_version {file_version!r} != requested {chosen!r}")
    if path is not None and expected_sha256 is None and file_version != PROTOCOL_VERSION:
        # Legacy test path: only the default slice is accepted without an explicit pin.
        raise IntegrityError(f"protocol_version {file_version!r} != supported {PROTOCOL_VERSION!r}")
    required = tuple(str(g) for g in data["required_gates"])
    gates = {str(k): dict(v) for k, v in data["gates"].items()}
    missing = [g for g in required if g not in gates]
    if not required or missing:
        raise IntegrityError(f"required gates without configuration: {missing}")
    return ValidationProtocol(
        version=file_version,
        required_gates=required,
        periods_per_year=float(data["annualization_periods_per_year"]),
        gates=gates,
        sha256=digest,
    )


def protocol_for_sampling(frequency: str, current: ValidationProtocol) -> ValidationProtocol:
    """Protocol for a dataset's sampling frequency.

    Eight-hour research keeps the protocol already loaded on the service, so an
    existing sealed holdout and any historical 8h report stay on that version.
    One-hour and one-day research use their own pinned protocols. The label
    horizon is not a frequency: callers still pass it separately as bars.
    """
    if frequency == "1h":
        return load_protocol(version="0.2.0-1h")
    if frequency == "1d":
        return load_protocol(version="0.2.0-1d")
    if frequency in {"8h", ""}:
        return current
    raise IntegrityError(f"no pinned validation protocol for sampling frequency {frequency!r}")
