"""Credential lookup and secret redaction.

Keys are read from environment variables at call time only. They are never stored on
registry objects, logged, persisted, or included in error messages.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

REDACTED = "[REDACTED]"

_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"gsk_[A-Za-z0-9]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"),
    re.compile(
        r"(?i)((?:x-api-key|x-goog-api-key|api[_-]?key|authorization)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}&]{6,}"
    ),
    re.compile(r"(?i)([?&]key=)[^&\s\"']+"),
)


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Remove known secret values and anything shaped like an API key."""
    for secret in sorted({s for s in secrets if s and len(s) >= 6}, key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    for pattern in _KEY_PATTERNS:
        text = pattern.sub(
            lambda m: (m.group(1) if m.lastindex else "") + REDACTED,
            text,
        )
    return text


def first_present(environ: Mapping[str, str], names: Iterable[str]) -> str | None:
    for name in names:
        value = environ.get(name, "").strip()
        if value:
            return value
    return None


def secret_values(environ: Mapping[str, str], names: Iterable[str]) -> list[str]:
    return [v for n in names if (v := environ.get(n, "").strip())]
