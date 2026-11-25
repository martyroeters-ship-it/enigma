"""Error taxonomy. Governance errors fail closed and must never be swallowed silently."""

from __future__ import annotations


class EnigmaError(Exception):
    """Base class for all Enigma errors."""


class ConstitutionViolationError(EnigmaError):
    """An action would breach the institutional constitution. Treated as a security incident."""


class PermissionDeniedError(ConstitutionViolationError):
    """An actor attempted an operation outside its authority."""


class BudgetExceededError(EnigmaError):
    """A computational or financial budget ceiling would be exceeded."""


class InvalidTransitionError(EnigmaError):
    """A state machine transition is not permitted from the current state or by this actor."""


class IntegrityError(EnigmaError):
    """Data, hash-chain, or artifact integrity check failed."""


class ProviderUnavailableError(EnigmaError):
    """An external provider (model, search, data) is not configured or unreachable."""
