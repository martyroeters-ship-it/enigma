"""Polite HTTP access for public market-data endpoints: rate limit, retries, request counting."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

import httpx

from enigma.core.errors import ProviderUnavailableError

DEFAULT_USER_AGENT = "Enigma-Research/0.1 (+https://github.com/martyroeters-ship-it/enigma)"

_RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_BLOCKED_STATUS = frozenset({401, 403, 418, 451})


class PoliteHttp:
    """Synchronous GET client with a minimum interval between requests and bounded retries.

    `requests` counts every HTTP request actually sent (including retries), so callers can
    record a `CostRecord(category="http", http_requests=...)`.
    """

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        min_interval_s: float = 0.2,
        max_retries: int = 4,
        backoff_s: float = 1.0,
        timeout_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.user_agent = user_agent
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.requests = 0
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request: float | None = None
        self._client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout_s,
            transport=transport,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self.min_interval_s - (self._monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._monotonic()

    def get(self, url: str, params: Mapping[str, str] | None = None) -> httpx.Response | None:
        """GET `url`. Returns None on 404; raises `ProviderUnavailableError` when exhausted."""
        last_error = ""
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.requests += 1
            try:
                response = self._client.get(url, params=params)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 404:
                    return None
                if response.status_code in _BLOCKED_STATUS:
                    raise ProviderUnavailableError(
                        f"{url} refused access (HTTP {response.status_code})"
                    )
                if response.status_code < 400:
                    return response
                last_error = f"HTTP {response.status_code}"
                if response.status_code not in _RETRY_STATUS:
                    break
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit() and attempt < self.max_retries:
                    self._sleep(float(retry_after))
                    continue
            if attempt < self.max_retries:
                self._sleep(self.backoff_s * (2**attempt))
        raise ProviderUnavailableError(f"GET {url} failed: {last_error}")
