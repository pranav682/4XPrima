"""Market-data interface for the fast loop, plus the OANDA v20 adapter.

This module owns:

- The :class:`PriceProvider` Protocol — a quote source. Stable; the rest of
  the fast loop (paper broker, future strategies) only ever depends on this.
- The :class:`CandleProvider` Protocol — a bar source. Optional capability;
  some implementations provide both, others only one.
- :class:`StubPriceProvider` for tests.
- :class:`OandaPriceProvider` implementing both Protocols against the OANDA
  v20 REST API. Verified against developer.oanda.com on 2026-06-24:
    * pricing: ``GET /v3/accounts/{accountID}/pricing?instruments=...``
    * candles: ``GET /v3/instruments/{instrument}/candles``
  with header ``Authorization: Bearer <token>``. Practice and live base URLs
  resolved by :class:`core.config.OandaSettings`. 120 req/s rate limit,
  HTTP 429 on excess.

This module makes NO order-execution calls. Orders still flow exclusively
through :class:`core.execution.ExecutionEngine`.
"""

from __future__ import annotations

import logging
import random
import time as _time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Final, Protocol

import httpx
import structlog

from core.config import OandaSettings
from core.models import Candle, Granularity, Quote

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MarketDataError(Exception):
    """Raised on unrecoverable market-data failures (after retries exhausted,
    non-retryable HTTP status, malformed payload, transport timeout).

    :class:`core.execution.ExecutionEngine` trips the kill switch on any
    exception from a broker call. The same fail-safe applies if a price
    provider raises this — the *upstream* caller (the future strategy
    runner) is expected to catch it and decide whether to halt.
    """


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class PriceProvider(Protocol):
    """Source of current bid/ask for a pair.

    Implementations are responsible for freshness (the caller can read the
    :attr:`Quote.timestamp` to decide whether a quote is stale enough to
    refuse to trade on it — that policy lives upstream, not here).
    """

    def get_quote(self, pair: str) -> Quote:
        """Return the most recent quote for ``pair``.

        Raises:
            KeyError: if the provider has no quote for ``pair``.
            MarketDataError: if the provider hit a transport/parse failure.
        """
        ...


class CandleProvider(Protocol):
    """Source of historical OHLCV bars. Distinct from :class:`PriceProvider`
    because not every quote source provides bars."""

    def get_candles(
        self,
        pair: str,
        *,
        granularity: Granularity,
        count: int | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ) -> list[Candle]:
        """Return bars for ``pair``. Either ``count`` OR ``from_time``/``to_time``
        should be specified — implementations may reject combinations.

        Raises:
            MarketDataError: on any unrecoverable failure.
        """
        ...


# ---------------------------------------------------------------------------
# Test stub
# ---------------------------------------------------------------------------


class StubPriceProvider:
    """Trivial in-memory provider, for tests.

    Holds a single quote per pair. ``set_quote`` overwrites; ``get_quote``
    raises ``KeyError`` if the pair was never set — implementations of the
    real feed will surface a similar error when an unknown pair is requested.
    """

    def __init__(self, quotes: dict[str, Quote] | None = None) -> None:
        self._quotes: dict[str, Quote] = {}
        if quotes:
            for pair, quote in quotes.items():
                self.set_quote(pair, quote)

    def set_quote(self, pair: str, quote: Quote) -> None:
        if pair.upper() != quote.pair:
            raise ValueError(
                f"quote pair {quote.pair!r} does not match key {pair!r}"
            )
        self._quotes[quote.pair] = quote

    def set_price(
        self,
        pair: str,
        *,
        bid: Decimal,
        ask: Decimal,
        timestamp: datetime | None = None,
    ) -> None:
        """Convenience: build the Quote and stash it in one call."""
        self.set_quote(
            pair,
            Quote(
                pair=pair,
                bid=bid,
                ask=ask,
                timestamp=timestamp or datetime.now(UTC),
            ),
        )

    def get_quote(self, pair: str) -> Quote:
        return self._quotes[pair.upper()]


# ---------------------------------------------------------------------------
# Pair-format helpers
# ---------------------------------------------------------------------------


def to_oanda_instrument(pair: str) -> str:
    """Convert any of ``EUR/USD``, ``EURUSD``, ``EUR_USD`` to OANDA's
    canonical ``EUR_USD`` instrument naming.

    Rule: strip separators (``/`` ``-``), upper-case, then re-insert the
    underscore at the midpoint. Forex pairs are always 6 letters split 3/3;
    anything else is rejected.
    """
    cleaned = pair.replace("/", "").replace("_", "").replace("-", "").upper()
    if not cleaned.isalpha() or len(cleaned) != 6:
        raise ValueError(
            f"unrecognised pair format {pair!r}; expected EUR/USD, EURUSD, or EUR_USD"
        )
    return f"{cleaned[:3]}_{cleaned[3:]}"


def to_canonical_pair(oanda_instrument: str) -> str:
    """Map ``EUR_USD`` → ``EURUSD`` for our :class:`Quote.pair` /
    :class:`Candle.pair` fields (which require ``isalpha()``)."""
    return oanda_instrument.replace("_", "").upper()


# ---------------------------------------------------------------------------
# OANDA timestamp parsing
# ---------------------------------------------------------------------------


def _parse_oanda_time(s: str) -> datetime:
    """Parse OANDA's RFC3339 timestamps, which include nanosecond precision.

    Examples we must handle:
      ``2016-06-22T18:41:36.201836422Z``  (RFC3339 w/ nanos)
      ``2016-06-22T18:41:36Z``            (no fractional)
      ``2016-06-22T18:41:36.123456Z``     (microsecond)

    Python's ``datetime.fromisoformat`` accepts microseconds at most. We
    truncate any sub-microsecond digits before parsing, never round, so
    we don't accidentally bump a candle to the next second.
    """
    if not s.endswith("Z"):
        # OANDA always returns Z-suffixed UTC. If something else turns up,
        # surface it — silently coercing would mask a bug.
        raise ValueError(f"expected Z-suffixed UTC timestamp, got {s!r}")
    body = s[:-1]
    if "." in body:
        head, frac = body.split(".", 1)
        frac = frac[:6]  # microseconds is pydantic / stdlib's ceiling
        body = f"{head}.{frac}"
    return datetime.fromisoformat(body).replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# OANDA provider
# ---------------------------------------------------------------------------


_DEFAULT_TIMEOUT_SECONDS: Final[float] = 10.0
_DEFAULT_RETRIES: Final[int] = 3
_DEFAULT_BASE_BACKOFF_SECONDS: Final[float] = 0.25
_RETRYABLE_STATUSES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})


class OandaPriceProvider:
    """OANDA v20 REST adapter — read-only.

    Implements :class:`PriceProvider` and :class:`CandleProvider`. No order
    placement, no streaming yet — polling only. Suitable for the fast loop's
    price refresh and for ad-hoc historical-bar pulls.

    Thread safety: the underlying :class:`httpx.Client` is reused across
    calls; a single instance is intended for single-threaded use (the fast
    loop is single-threaded). For concurrent use, construct one per thread.
    """

    def __init__(
        self,
        settings: OandaSettings,
        *,
        client: httpx.Client | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        retries: int = _DEFAULT_RETRIES,
        base_backoff_seconds: float = _DEFAULT_BASE_BACKOFF_SECONDS,
        logger: structlog.stdlib.BoundLogger | None = None,
        sleep: object | None = None,
    ) -> None:
        self._settings = settings
        self._timeout = timeout_seconds
        self._retries = retries
        self._base_backoff = base_backoff_seconds
        # `sleep` is injectable so tests don't actually wait.
        self._sleep = sleep if sleep is not None else _time.sleep
        # The Authorization header carries the secret. We build the client
        # here so the secret is set ONCE; nothing else in this module ever
        # touches `api_token.get_secret_value()`.
        if client is None:
            self._client = httpx.Client(
                base_url=settings.base_url,
                headers={
                    "Authorization": f"Bearer {settings.api_token.get_secret_value()}",
                    "Accept-Datetime-Format": "RFC3339",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(timeout_seconds),
            )
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False
        self._logger = (
            logger if logger is not None else _default_logger()
        ).bind(component="oanda_price_provider", env=settings.env.value)

    # ---------------------------------------------------------- lifecycle

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OandaPriceProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --------------------------------------------------------------- API

    def get_quote(self, pair: str) -> Quote:
        instrument = to_oanda_instrument(pair)
        canonical = to_canonical_pair(instrument)

        path = f"/v3/accounts/{self._settings.account_id}/pricing"
        params = {"instruments": instrument}
        payload = self._request_json(path, params=params)

        prices = payload.get("prices") or []
        match = next(
            (p for p in prices if p.get("instrument") == instrument),
            None,
        )
        if match is None:
            raise MarketDataError(
                f"no price entry for {instrument!r} in OANDA response"
            )

        try:
            bid = Decimal(str(match["closeoutBid"]))
            ask = Decimal(str(match["closeoutAsk"]))
            ts = _parse_oanda_time(match["time"])
        except (KeyError, ValueError) as e:
            raise MarketDataError(f"malformed pricing payload for {instrument}: {e}") from e

        return Quote(pair=canonical, bid=bid, ask=ask, timestamp=ts)

    def get_candles(
        self,
        pair: str,
        *,
        granularity: Granularity,
        count: int | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ) -> list[Candle]:
        instrument = to_oanda_instrument(pair)
        canonical = to_canonical_pair(instrument)

        if count is None and from_time is None and to_time is None:
            raise ValueError("must supply count or from_time/to_time")
        if count is not None and (from_time is not None or to_time is not None):
            raise ValueError("supply EITHER count OR from_time/to_time, not both")

        params: dict[str, str] = {"price": "M", "granularity": granularity.value}
        if count is not None:
            params["count"] = str(count)
        if from_time is not None:
            params["from"] = _to_oanda_time(from_time)
        if to_time is not None:
            params["to"] = _to_oanda_time(to_time)

        path = f"/v3/instruments/{instrument}/candles"
        payload = self._request_json(path, params=params)

        raw_candles = payload.get("candles") or []
        out: list[Candle] = []
        for raw in raw_candles:
            try:
                mid = raw["mid"]
                candle = Candle(
                    pair=canonical,
                    granularity=granularity,
                    time=_parse_oanda_time(raw["time"]),
                    open=Decimal(str(mid["o"])),
                    high=Decimal(str(mid["h"])),
                    low=Decimal(str(mid["l"])),
                    close=Decimal(str(mid["c"])),
                    volume=int(raw.get("volume", 0)),
                    complete=bool(raw.get("complete", False)),
                )
            except (KeyError, ValueError) as e:
                raise MarketDataError(
                    f"malformed candle for {instrument}: {e}"
                ) from e
            out.append(candle)
        return out

    # -------------------------------------------------------- HTTP plumbing

    def _request_json(
        self, path: str, *, params: dict[str, str] | None = None
    ) -> dict[str, object]:
        """GET ``path`` with exponential backoff on retryable failures.

        Retry on HTTP 429/5xx and on transport errors (connect/read timeouts).
        Respect OANDA's ``Retry-After`` header when present — fall back to
        ``base_backoff * 2**attempt`` with jitter otherwise.
        """
        last_exc: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                response = self._client.get(path, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                self._logger.warning(
                    "oanda_request_transport_error",
                    attempt=attempt,
                    path=path,
                    error=str(e),
                )
                if attempt == self._retries:
                    raise MarketDataError(
                        f"transport error after {self._retries + 1} attempts: {e}"
                    ) from e
                self._sleep(self._backoff_seconds(attempt, retry_after=None))
                continue

            if response.status_code in _RETRYABLE_STATUSES and attempt < self._retries:
                self._logger.warning(
                    "oanda_request_retryable_status",
                    attempt=attempt,
                    path=path,
                    status=response.status_code,
                )
                self._sleep(
                    self._backoff_seconds(
                        attempt,
                        retry_after=_parse_retry_after(response.headers.get("Retry-After")),
                    )
                )
                continue

            if response.status_code >= 400:
                # Surface OANDA's errorMessage if it sent one; do not include
                # the request path in the message in case it contains a token
                # somewhere unexpected.
                detail = _safe_error_detail(response)
                raise MarketDataError(
                    f"OANDA returned HTTP {response.status_code}: {detail}"
                )

            try:
                return response.json()
            except ValueError as e:
                raise MarketDataError(f"non-JSON response from OANDA: {e}") from e

        # The loop only falls through here when retries are exhausted on a
        # retryable status. We need a final raise for that case.
        raise MarketDataError(
            f"OANDA request failed after {self._retries + 1} attempts: "
            f"last error {last_exc!r}"
        )

    def _backoff_seconds(self, attempt: int, *, retry_after: float | None) -> float:
        if retry_after is not None:
            return max(0.0, retry_after)
        # Exponential with jitter to avoid thundering-herd retries.
        # `random` is fine here — backoff jitter is not security-sensitive.
        base = self._base_backoff * (2**attempt)
        return base + random.uniform(0, base)  # noqa: S311


# ---------------------------------------------------------------------------
# OANDA timestamp encoding (for outgoing query strings)
# ---------------------------------------------------------------------------


def _to_oanda_time(dt: datetime) -> str:
    """Encode a datetime to OANDA's accepted RFC3339 format (Z-suffix UTC)."""
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware UTC")
    # OANDA accepts microseconds at most. Replace tz to make sure we always
    # emit the Z form rather than +00:00.
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_retry_after(value: str | None) -> float | None:
    """Parse the Retry-After header into seconds. Accepts numeric form only
    (OANDA convention); HTTP-date form would be valid too but we don't see
    it from OANDA so it's not worth the parsing cost."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _safe_error_detail(response: httpx.Response) -> str:
    """Best-effort error detail extraction that never re-includes the
    request URL (which could in theory carry a leaked token), never includes
    request headers, and is safe to log."""
    try:
        body = response.json()
        if isinstance(body, dict):
            msg = body.get("errorMessage") or body.get("error") or ""
            if isinstance(msg, str) and msg:
                return msg
    except ValueError:
        pass
    text = response.text or ""
    return text[:200]  # avoid dumping huge HTML error pages


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _default_logger() -> structlog.stdlib.BoundLogger:
    if not structlog.is_configured():
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    return structlog.get_logger("core.market_data")


__all__ = [
    "CandleProvider",
    "MarketDataError",
    "OandaPriceProvider",
    "PriceProvider",
    "StubPriceProvider",
    "to_canonical_pair",
    "to_oanda_instrument",
]
