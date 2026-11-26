"""Polite, budgeted HTTP access for the Intelligence plane.

Every request is charged against a shared `HttpBudget` (hard ceiling, fail closed), rate limited
per host, retried with exponential backoff on 429/5xx/transport errors, capped in size, and the
raw bytes of every successful response are stored content-addressed on disk. Generic page fetches
honour robots.txt (including Crawl-delay). Redirects are followed manually so that every hop is
re-checked against the egress guard and robots.txt.
"""

from __future__ import annotations

import ipaddress
import json
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from enigma.core.errors import (
    BudgetExceededError,
    EnigmaError,
    PermissionDeniedError,
    ProviderUnavailableError,
)
from enigma.core.hashing import sha256_hex
from enigma.core.logging import get_logger
from enigma.core.paths import StorageRoots, bind_uri, join_uri, within

log = get_logger(__name__)

DEFAULT_USER_AGENT = "Enigma-Research/0.1 (+https://github.com/martyroeters-ship-it/enigma)"
DEFAULT_MAX_BYTES = 8 * 1024 * 1024
MAX_REDIRECTS = 5
MAX_CRAWL_DELAY_SECONDS = 20.0

# Documented provider etiquette: arXiv asks for >= 3 s between API calls.
DEFAULT_HOST_INTERVALS: dict[str, float] = {
    "export.arxiv.org": 3.0,
    "arxiv.org": 3.0,
    "api.github.com": 2.0,
    "api.semanticscholar.org": 1.5,
}


class HttpStatusError(EnigmaError):
    """Non-retryable HTTP failure (4xx other than 429)."""

    def __init__(self, url: str, status: int) -> None:
        super().__init__(f"HTTP {status} for {url}")
        self.url = url
        self.status = status


class ResponseTooLargeError(EnigmaError):
    """Response exceeded the configured byte cap; the body was discarded."""


class RobotsDisallowedError(PermissionDeniedError):
    """robots.txt for the host disallows fetching this URL for our user agent."""


class HttpBudget:
    """Shared request counter with a hard ceiling (`Budget.max_http_requests`)."""

    def __init__(self, max_requests: int) -> None:
        if max_requests < 0:
            raise ValueError("max_requests must be >= 0")
        self.max_requests = max_requests
        self._used = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.max_requests - self._used)

    def charge(self, url: str) -> None:
        with self._lock:
            if self._used >= self.max_requests:
                raise BudgetExceededError(
                    f"HTTP budget of {self.max_requests} requests exhausted (next: {url})"
                )
            self._used += 1


class RawStore:
    """Content-addressed raw byte store: `<root>/<hh>/<sha256hex><suffix>`."""

    def __init__(self, root: Path, roots: StorageRoots | None = None) -> None:
        self.root = root
        self.roots = roots

    def put(self, data: bytes, suffix: str = "") -> tuple[str, str]:
        """Store bytes; returns `(uri, "sha256:<hex>")`. Idempotent for identical content."""
        digest = sha256_hex(data)
        path = self.root / digest[:2] / f"{digest}{suffix}"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return bind_uri(path, self.roots, catalog_root=self.root), "sha256:" + digest

    def resolve(self, uri: str) -> Path:
        """Path inside this store. Missing files still return a path under the root."""
        candidate = join_uri(uri, self.roots, catalog_root=self.root)
        if candidate is None or not within(candidate, self.root):
            raise PermissionDeniedError(f"{uri} is outside the raw store")
        return candidate

    def read(self, uri: str) -> bytes:
        return self.resolve(uri).read_bytes()


@dataclass(frozen=True)
class HttpResponse:
    url: str
    status: int
    headers: Mapping[str, str]
    content: bytes
    raw_uri: str
    content_hash: str

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").split(";")[0].strip().lower()

    @property
    def text(self) -> str:
        charset = "utf-8"
        for part in self.headers.get("content-type", "").split(";")[1:]:
            key, _, value = part.strip().partition("=")
            if key.lower() == "charset" and value:
                charset = value.strip("\"'")
        try:
            return self.content.decode(charset, errors="replace")
        except LookupError:
            return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.content)


@dataclass
class HostStats:
    requests: Counter[str] = field(default_factory=Counter)
    retries: Counter[str] = field(default_factory=Counter)
    errors: Counter[str] = field(default_factory=Counter)


def _suffix_for(content_type: str) -> str:
    return {
        "application/json": ".json",
        "application/atom+xml": ".xml",
        "application/xml": ".xml",
        "text/xml": ".xml",
        "text/html": ".html",
        "application/pdf": ".pdf",
        "text/plain": ".txt",
    }.get(content_type, ".bin")


def check_egress(url: str, allowed_hosts: frozenset[str] | None) -> str:
    """Return the lower-cased host if `url` may be fetched; raise PermissionDeniedError otherwise.

    Blocks non-HTTP(S) schemes, embedded credentials, localhost, and private/loopback/link-local
    IP literals (basic SSRF guard), then applies the optional host allowlist.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise PermissionDeniedError(f"scheme {parts.scheme!r} not allowed: {url}")
    if parts.username or parts.password:
        raise PermissionDeniedError(f"URLs with credentials are not allowed: {url}")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host or host == "localhost" or host.endswith(".localhost"):
        raise PermissionDeniedError(f"host not allowed: {url}")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and not ip.is_global:
        raise PermissionDeniedError(f"non-public address not allowed: {url}")
    if allowed_hosts is not None and not any(
        host == h or host.endswith("." + h) for h in allowed_hosts
    ):
        raise PermissionDeniedError(f"host {host} is not on the egress allowlist")
    return host


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


class HttpClient:
    """Budgeted, polite HTTP GET client. One instance is shared by all providers of a run."""

    def __init__(
        self,
        budget: HttpBudget,
        raw_store: RawStore,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: float = 20.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        max_backoff_seconds: float = 30.0,
        max_bytes: int = DEFAULT_MAX_BYTES,
        default_host_interval: float = 1.0,
        host_intervals: Mapping[str, float] | None = None,
        allowed_hosts: frozenset[str] | None = None,
        transport: httpx.BaseTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.budget = budget
        self.raw_store = raw_store
        self.user_agent = user_agent
        self.max_retries = max_retries
        self.backoff_base = backoff_base_seconds
        self.max_backoff = max_backoff_seconds
        self.max_bytes = max_bytes
        self.default_host_interval = default_host_interval
        self.host_intervals = dict(
            DEFAULT_HOST_INTERVALS if host_intervals is None else host_intervals
        )
        self.allowed_hosts = allowed_hosts
        self.stats = HostStats()
        self._sleep = sleeper
        self._monotonic = monotonic
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser | None] = {}
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            headers={"User-Agent": user_agent},
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _throttle(self, host: str) -> None:
        interval = self.host_intervals.get(host, self.default_host_interval)
        last = self._last_request.get(host)
        if last is not None:
            wait = interval - (self._monotonic() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_request[host] = self._monotonic()

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        delay = self.backoff_base * float(2**attempt)
        if retry_after is not None:
            delay = max(delay, retry_after)
        return min(delay, self.max_backoff)

    def _read_capped(self, response: httpx.Response, cap: int) -> bytes:
        declared = response.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > cap:
            raise ResponseTooLargeError(f"{response.url}: declared {declared} bytes > cap {cap}")
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > cap:
                raise ResponseTooLargeError(f"{response.url}: body exceeds cap of {cap} bytes")
            chunks.append(chunk)
        return b"".join(chunks)

    def _attempt(
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        json_body: Any,
        cap: int,
    ) -> tuple[int, httpx.Headers, bytes, str]:
        with self._client.stream(
            method, url, params=params, headers=headers, json=json_body
        ) as response:
            status = response.status_code
            if status >= 300:
                return status, response.headers, b"", str(response.url)
            return status, response.headers, self._read_capped(response, cap), str(response.url)

    def get(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        *,
        max_bytes: int | None = None,
        respect_robots: bool = False,
        store: bool = True,
    ) -> HttpResponse:
        """GET with budget, throttling, retries, size cap, and raw storage of the final body."""
        return self.request(
            "GET",
            url,
            params,
            headers,
            max_bytes=max_bytes,
            respect_robots=respect_robots,
            store=store,
        )

    def post_json(
        self, url: str, body: Mapping[str, Any], headers: Mapping[str, str] | None = None
    ) -> HttpResponse:
        """POST a JSON body to a search API (no redirects followed for non-GET requests)."""
        return self.request("POST", url, None, headers, json_body=dict(body))

    def request(
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        *,
        json_body: Any = None,
        max_bytes: int | None = None,
        respect_robots: bool = False,
        store: bool = True,
    ) -> HttpResponse:
        cap = self.max_bytes if max_bytes is None else max_bytes
        current = url
        current_params = params
        for _hop in range(MAX_REDIRECTS + 1):
            host = check_egress(current, self.allowed_hosts)
            if respect_robots and not self.robots_allowed(current):
                raise RobotsDisallowedError(f"robots.txt disallows {current}")
            status, resp_headers, body, final_url = self._with_retries(
                method, current, host, current_params, headers, json_body, cap
            )
            if 300 <= status < 400:
                if method != "GET":
                    raise HttpStatusError(current, status)
                location = resp_headers.get("location")
                if not location:
                    raise HttpStatusError(current, status)
                current = urljoin(final_url, location)
                current_params = None
                continue
            header_map = {k.lower(): v for k, v in resp_headers.items()}
            if store:
                suffix = _suffix_for(header_map.get("content-type", "").split(";")[0].strip())
                raw_uri, content_hash = self.raw_store.put(body, suffix)
            else:
                raw_uri, content_hash = "", "sha256:" + sha256_hex(body)
            return HttpResponse(
                url=final_url,
                status=status,
                headers=header_map,
                content=body,
                raw_uri=raw_uri,
                content_hash=content_hash,
            )
        raise HttpStatusError(url, 310)

    def _with_retries(
        self,
        method: str,
        url: str,
        host: str,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        json_body: Any,
        cap: int,
    ) -> tuple[int, httpx.Headers, bytes, str]:
        last_error = "unknown"
        for attempt in range(self.max_retries + 1):
            self.budget.charge(url)
            self._throttle(host)
            self.stats.requests[host] += 1
            retry_after: float | None = None
            try:
                status, resp_headers, body, final_url = self._attempt(
                    method, url, params, headers, json_body, cap
                )
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if status == 429 or status >= 500:
                    last_error = f"HTTP {status}"
                    retry_after = _retry_after_seconds(resp_headers.get("retry-after"))
                elif status >= 400:
                    self.stats.errors[host] += 1
                    raise HttpStatusError(url, status)
                else:
                    return status, resp_headers, body, final_url
            if attempt < self.max_retries:
                self.stats.retries[host] += 1
                delay = self._backoff(attempt, retry_after)
                log.info(
                    "http.retry", host=host, attempt=attempt + 1, delay=delay, error=last_error
                )
                self._sleep(delay)
        self.stats.errors[host] += 1
        raise ProviderUnavailableError(f"{host} unavailable after retries: {last_error}")

    def robots_allowed(self, url: str) -> bool:
        """Check robots.txt for `url` (cached per host). Unreachable robots.txt fails closed."""
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            self._robots[origin] = self._load_robots(origin)
        parser = self._robots[origin]
        if parser is None:
            return False
        return parser.can_fetch(self.user_agent, url)

    def _load_robots(self, origin: str) -> RobotFileParser | None:
        robots_url = origin + "/robots.txt"
        parser = RobotFileParser(robots_url)
        try:
            response = self.get(robots_url, max_bytes=512 * 1024, store=False)
        except HttpStatusError as exc:
            if exc.status in (401, 403):
                return None
            parser.parse([])  # 404 and other 4xx: no restrictions published
            return parser
        except (ProviderUnavailableError, ResponseTooLargeError, PermissionDeniedError):
            return None
        parser.parse(response.text.splitlines())
        delay = parser.crawl_delay(self.user_agent)
        if delay is not None:
            host = (urlsplit(origin).hostname or "").lower()
            current = self.host_intervals.get(host, self.default_host_interval)
            self.host_intervals[host] = max(current, min(float(delay), MAX_CRAWL_DELAY_SECONDS))
        return parser
