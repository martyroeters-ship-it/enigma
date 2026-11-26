"""Gateway error types. All derive from the core taxonomy so callers can fail closed."""

from __future__ import annotations

from enigma.core.errors import EnigmaError, ProviderUnavailableError


class ProviderCallError(ProviderUnavailableError):
    """A provider call failed. Messages are always redacted before construction."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        retryable: bool = False,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after_s = retry_after_s
        self.attempts = 1


class MissingCredentialsError(ProviderUnavailableError):
    """The provider needs environment variables that are not set."""

    def __init__(self, provider: str, env_vars: tuple[str, ...]) -> None:
        super().__init__(f"provider {provider!r} not configured; set one of: {', '.join(env_vars)}")
        self.provider = provider
        self.env_vars = env_vars


class StructuredOutputError(EnigmaError):
    """The model's output did not validate against the requested schema, even after repair."""


class RegistryError(EnigmaError):
    """The model/pricing/routing registry is invalid."""
