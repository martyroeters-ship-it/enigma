"""Validation plane: independent, deterministic validation gates and holdout custody.

Nothing in this package may import `enigma.gateway` (no model may influence pass/fail); an
import-linter contract enforces it.
"""

from enigma.validation.build import validator_build_sha
from enigma.validation.gates import GATE_FAILURE_KIND, INFORMATIVE_GATES
from enigma.validation.holdout import (
    HoldoutCustody,
    HoldoutSource,
    InMemoryHoldoutSource,
    frame_content_hash,
)
from enigma.validation.protocol import (
    PROTOCOL_SHA256,
    PROTOCOL_VERSION,
    ValidationProtocol,
    load_protocol,
)
from enigma.validation.service import ValidationService, render_summary

__all__ = [
    "GATE_FAILURE_KIND",
    "INFORMATIVE_GATES",
    "PROTOCOL_SHA256",
    "PROTOCOL_VERSION",
    "HoldoutCustody",
    "HoldoutSource",
    "InMemoryHoldoutSource",
    "ValidationProtocol",
    "ValidationService",
    "frame_content_hash",
    "load_protocol",
    "render_summary",
    "validator_build_sha",
]
