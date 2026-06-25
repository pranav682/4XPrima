"""Context-data adapters: economic calendar, macro time series, news.

Three deterministic, read-only providers that distil external feeds into
the typed models in :mod:`core.models`. **No LLM, no agents, no
interpretation** — these adapters do parsing and dedup only; the future
:mod:`market_context_agent` is the layer that reads them.

Each provider sits behind its own Protocol so the source can swap:

- :class:`EconomicCalendarProvider` — scheduled releases with actual /
  forecast / previous. Default implementation:
  :class:`ForexFactoryCalendarProvider` against the public Forex Factory
  community feed (no API key, forex-focused). Could be replaced with
  Finnhub free tier or any other source without touching upstream code.
- :class:`MacroDataProvider` — FRED time series.
  :class:`FredMacroDataProvider` is the default; named series map
  (:data:`FRED_SERIES`) gives callers human names for the handful that
  matter for FX.
- :class:`NewsProvider` — GDELT Doc 2.0 article list.
  :class:`GdeltNewsProvider` is the default. GDELT tone is dictionary-based
  and weaker than transformer sentiment; it's good for *detecting and
  locating* events, not for final sentiment. The agent will interpret.

Reuses the retry / backoff / timeout / typed-error discipline from the
OANDA adapter: :class:`ContextDataError` is the unrecoverable failure
exception; all secrets stay in :class:`pydantic.SecretStr`; error messages
never include request URLs or headers, so a leaked key can't end up in a
traceback.
"""

from __future__ import annotations

import logging
import random
import re
import time as _time
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Final, Protocol

import httpx
import structlog

from core.config import FredSettings
from core.models import (
    EconomicEvent,
    ImpactLevel,
    MacroSeriesPoint,
    NewsEvent,
)

# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class ContextDataError(Exception):
    """Raised on unrecoverable context-data failures.

    Analogous to :class:`core.market_data.MarketDataError`. Upstream callers
    (the future strategy runner / agent orchestrator) should catch this and
    decide whether to halt or degrade.
    """


# ---------------------------------------------------------------------------
# Shared HTTP helper (retry + backoff)
# ---------------------------------------------------------------------------


_DEFAULT_TIMEOUT_SECONDS: Final[float] = 10.0
_DEFAULT_RETRIES: Final[int] = 3
_DEFAULT_BASE_BACKOFF_SECONDS: Final[float] = 0.25
_RETRYABLE_STATUSES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})


def _http_get_json(
    client: httpx.Client,
    path: str,
    *,
    params: dict[str, str] | None,
    retries: int,
    base_backoff: float,
    sleep: Callable[[float], None],
    logger: structlog.stdlib.BoundLogger,
    error_cls: type[Exception] = ContextDataError,
) -> object:
    """GET ``path`` with retries on transport errors and 429/5xx.

    Returns parsed JSON (dict OR list — feeds vary). Raises ``error_cls`` on
    unrecoverable failure. Error messages never include the request URL or
    headers, so a leaked key can't end up in a traceback.
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.get(path, params=params)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_exc = e
            logger.warning(
                "context_data_request_transport_error",
                attempt=attempt,
                path=path,
                error=str(e),
            )
            if attempt == retries:
                raise error_cls(
                    f"transport error after {retries + 1} attempts: {e}"
                ) from e
            sleep(_backoff_seconds(attempt, base_backoff, retry_after=None))
            continue

        if response.status_code in _RETRYABLE_STATUSES and attempt < retries:
            logger.warning(
                "context_data_request_retryable_status",
                attempt=attempt,
                path=path,
                status=response.status_code,
            )
            sleep(
                _backoff_seconds(
                    attempt,
                    base_backoff,
                    retry_after=_parse_retry_after(response.headers.get("Retry-After")),
                )
            )
            continue

        if response.status_code >= 400:
            detail = _safe_error_detail(response)
            raise error_cls(f"HTTP {response.status_code}: {detail}")

        try:
            return response.json()
        except ValueError as e:
            raise error_cls(f"non-JSON response: {e}") from e

    raise error_cls(
        f"request failed after {retries + 1} attempts: last error {last_exc!r}"
    )


def _backoff_seconds(
    attempt: int, base_backoff: float, *, retry_after: float | None
) -> float:
    if retry_after is not None:
        return max(0.0, retry_after)
    base = base_backoff * (2**attempt)
    # `random` is fine — backoff jitter is not security-sensitive.
    return base + random.uniform(0, base)  # noqa: S311


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _safe_error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict):
            for key in ("errorMessage", "error_message", "error", "message"):
                msg = body.get(key)
                if isinstance(msg, str) and msg:
                    return msg
    except ValueError:
        pass
    text = response.text or ""
    return text[:200]


# ---------------------------------------------------------------------------
# Numeric parsing (Forex Factory uses suffixes like 170K, 3.5%, $5.2B)
# ---------------------------------------------------------------------------


_SUFFIX_MULTIPLIERS: Final[dict[str, Decimal]] = {
    "K": Decimal("1000"),
    "M": Decimal("1000000"),
    "B": Decimal("1000000000"),
    "T": Decimal("1000000000000"),
}
_NUMERIC_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*([+-]?\d[\d,]*\.?\d*)\s*([KMBT]?)\s*$",
    re.IGNORECASE,
)


def parse_economic_value(raw: str | None) -> Decimal | None:
    """Parse a Forex-Factory-style numeric value into ``Decimal``.

    Handles ``"170K"`` -> ``170000``, ``"3.5%"`` -> ``3.5``,
    ``"$5.2B"`` -> ``5200000000``, ``"1,234"`` -> ``1234``, ``"-18.6K"`` ->
    ``-18600``. Returns ``None`` for empty / unparseable values
    (``""``, ``None``, ``"Tentative"``, ``"All Day"``).

    The ``%`` is silently stripped: the percent semantic is in the field
    name (e.g. "CPI m/m"), not the number — agents that need to know it's a
    percent already know from the event title.
    """
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    # Strip currency markers and percent sign.
    cleaned = cleaned.replace("$", "").replace("%", "").replace("+", "")
    m = _NUMERIC_RE.match(cleaned)
    if not m:
        return None
    body, suffix = m.group(1), m.group(2).upper()
    body = body.replace(",", "")
    try:
        value = Decimal(body)
    except InvalidOperation:
        return None
    if suffix:
        value *= _SUFFIX_MULTIPLIERS[suffix]
    return value


def _parse_impact(raw: str | None) -> ImpactLevel:
    if not raw:
        return ImpactLevel.UNKNOWN
    try:
        return ImpactLevel(raw.lower())
    except ValueError:
        return ImpactLevel.UNKNOWN


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class EconomicCalendarProvider(Protocol):
    """Source of scheduled economic releases."""

    def upcoming(
        self,
        window: timedelta,
        *,
        currencies: Iterable[str] | None = None,
        min_impact: ImpactLevel | None = None,
    ) -> list[EconomicEvent]:
        """Events scheduled in the next ``window``, optionally filtered."""
        ...

    def recent(
        self,
        window: timedelta,
        *,
        currencies: Iterable[str] | None = None,
        min_impact: ImpactLevel | None = None,
    ) -> list[EconomicEvent]:
        """Events that occurred in the past ``window``."""
        ...


class MacroDataProvider(Protocol):
    """Source of macro time series (FRED-shaped)."""

    def get_series(
        self,
        series_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[MacroSeriesPoint]:
        ...


class NewsProvider(Protocol):
    """Source of news articles relevant to a query/theme over a window."""

    def search(
        self,
        query: str,
        *,
        start: datetime,
        end: datetime,
        max_results: int = 75,
    ) -> list[NewsEvent]:
        ...


# ---------------------------------------------------------------------------
# Forex Factory calendar provider
# ---------------------------------------------------------------------------

# Why Forex Factory (community JSON feed at faireconomy.media) and not
# Finnhub: no API key needed, forex-focused by construction, and stable
# field shapes (title/country/date/impact/forecast/previous/actual). Finnhub
# free tier would also work but adds a key and a rate-limit ceiling we'd
# have to honour. The Protocol is source-agnostic so this can swap.
_FF_BASE_URL: Final[str] = "https://nfs.faireconomy.media"
_FF_PATHS: Final[dict[str, str]] = {
    "last_week": "/ff_calendar_lastweek.json",
    "this_week": "/ff_calendar_thisweek.json",
    "next_week": "/ff_calendar_nextweek.json",
}
_IMPACT_ORDER: Final[dict[ImpactLevel, int]] = {
    ImpactLevel.UNKNOWN: 0,
    ImpactLevel.HOLIDAY: 0,
    ImpactLevel.LOW: 1,
    ImpactLevel.MEDIUM: 2,
    ImpactLevel.HIGH: 3,
}


class ForexFactoryCalendarProvider:
    """Reads the public Forex Factory weekly JSON feeds (no API key).

    The feed groups events by trading week. ``upcoming`` and ``recent``
    pull this/next or this/last week as needed, then filter by absolute
    time window relative to ``now()``.
    """

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        retries: int = _DEFAULT_RETRIES,
        base_backoff_seconds: float = _DEFAULT_BASE_BACKOFF_SECONDS,
        logger: structlog.stdlib.BoundLogger | None = None,
        sleep: Callable[[float], None] = _time.sleep,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._retries = retries
        self._base_backoff = base_backoff_seconds
        self._sleep = sleep
        self._now_fn = now_fn or (lambda: datetime.now(UTC))
        if client is None:
            self._client = httpx.Client(
                base_url=_FF_BASE_URL,
                timeout=httpx.Timeout(timeout_seconds),
                headers={"Accept": "application/json"},
            )
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False
        self._logger = (
            logger if logger is not None else _default_logger()
        ).bind(component="forex_factory_calendar")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> ForexFactoryCalendarProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------- API

    def upcoming(
        self,
        window: timedelta,
        *,
        currencies: Iterable[str] | None = None,
        min_impact: ImpactLevel | None = None,
    ) -> list[EconomicEvent]:
        if window <= timedelta(0):
            raise ValueError("window must be positive")
        now = self._now_fn()
        cutoff = now + window
        weeks = ["this_week"]
        if cutoff > now + timedelta(days=4):
            weeks.append("next_week")
        events = self._fetch_weeks(weeks)
        return _filter_events(
            events, start=now, end=cutoff, currencies=currencies, min_impact=min_impact
        )

    def recent(
        self,
        window: timedelta,
        *,
        currencies: Iterable[str] | None = None,
        min_impact: ImpactLevel | None = None,
    ) -> list[EconomicEvent]:
        if window <= timedelta(0):
            raise ValueError("window must be positive")
        now = self._now_fn()
        floor = now - window
        weeks = ["this_week"]
        if floor < now - timedelta(days=4):
            weeks.append("last_week")
        events = self._fetch_weeks(weeks)
        return _filter_events(
            events, start=floor, end=now, currencies=currencies, min_impact=min_impact
        )

    # -------------------------------------------------------- internals

    def _fetch_weeks(self, weeks: Iterable[str]) -> list[EconomicEvent]:
        seen: set[tuple[str, datetime, str]] = set()
        out: list[EconomicEvent] = []
        for week in weeks:
            payload = _http_get_json(
                self._client,
                _FF_PATHS[week],
                params=None,
                retries=self._retries,
                base_backoff=self._base_backoff,
                sleep=self._sleep,
                logger=self._logger,
                error_cls=ContextDataError,
            )
            if not isinstance(payload, list):
                raise ContextDataError("ff calendar payload was not a list")
            for raw in payload:
                event = _parse_ff_event(raw)
                if event is None:
                    continue
                key = (event.currency, event.when, event.name)
                if key in seen:
                    continue
                seen.add(key)
                out.append(event)
        out.sort(key=lambda e: e.when)
        return out


def _parse_ff_event(raw: dict) -> EconomicEvent | None:
    """Parse one Forex Factory JSON entry. Returns None on un-parseable input
    (e.g. missing required fields) rather than raising — the feed
    occasionally drops fields and we don't want one bad row to kill a fetch.
    """
    try:
        currency = raw.get("country")
        name = raw.get("title")
        raw_date = raw.get("date")
        if not currency or not name or not raw_date:
            return None
        # date is ISO-8601 with timezone, e.g. "2026-06-22T08:30:00-04:00"
        dt = datetime.fromisoformat(raw_date)
        if dt.tzinfo is None:
            return None
        when = dt.astimezone(UTC)
    except (TypeError, ValueError):
        return None

    raw_actual = raw.get("actual")
    raw_forecast = raw.get("forecast")
    raw_previous = raw.get("previous")

    return EconomicEvent(
        when=when,
        currency=currency,
        name=name,
        impact=_parse_impact(raw.get("impact")),
        raw_actual=raw_actual or None,
        raw_forecast=raw_forecast or None,
        raw_previous=raw_previous or None,
        actual=parse_economic_value(raw_actual),
        forecast=parse_economic_value(raw_forecast),
        previous=parse_economic_value(raw_previous),
    )


def _filter_events(
    events: Iterable[EconomicEvent],
    *,
    start: datetime,
    end: datetime,
    currencies: Iterable[str] | None,
    min_impact: ImpactLevel | None,
) -> list[EconomicEvent]:
    ccy_set = {c.upper() for c in currencies} if currencies is not None else None
    floor = _IMPACT_ORDER[min_impact] if min_impact is not None else None
    out: list[EconomicEvent] = []
    for e in events:
        if not (start <= e.when <= end):
            continue
        if ccy_set is not None and e.currency not in ccy_set:
            continue
        if floor is not None and _IMPACT_ORDER[e.impact] < floor:
            continue
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# FRED macro provider
# ---------------------------------------------------------------------------

# Curated handful that matters for FX. Callers reference these names rather
# than ids so a typo doesn't silently fetch the wrong series.
FRED_SERIES: Final[dict[str, str]] = {
    "US_CPI": "CPIAUCSL",            # CPI All Urban Consumers, SA
    "US_CORE_CPI": "CPILFESL",       # Core CPI (ex food and energy)
    "US_UNEMPLOYMENT": "UNRATE",     # Unemployment rate
    "US_GDP": "GDP",                 # Nominal GDP
    "US_REAL_GDP": "GDPC1",          # Real GDP
    "US_FED_FUNDS_RATE": "DFF",      # Effective Fed Funds rate (daily)
    "US_2Y_YIELD": "DGS2",
    "US_10Y_YIELD": "DGS10",
    "US_DOLLAR_INDEX": "DTWEXBGS",   # Broad trade-weighted USD
}


class FredMacroDataProvider:
    """FRED series-observations adapter.

    The API key is stored in :class:`FredSettings.api_key` (SecretStr) and
    only injected into the query string at request time. Errors never
    re-include the full URL, so the key can't leak through tracebacks.
    """

    def __init__(
        self,
        settings: FredSettings,
        *,
        client: httpx.Client | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        retries: int = _DEFAULT_RETRIES,
        base_backoff_seconds: float = _DEFAULT_BASE_BACKOFF_SECONDS,
        logger: structlog.stdlib.BoundLogger | None = None,
        sleep: Callable[[float], None] = _time.sleep,
    ) -> None:
        self._settings = settings
        self._retries = retries
        self._base_backoff = base_backoff_seconds
        self._sleep = sleep
        if client is None:
            self._client = httpx.Client(
                base_url=settings.base_url,
                timeout=httpx.Timeout(timeout_seconds),
                headers={"Accept": "application/json"},
            )
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False
        self._logger = (
            logger if logger is not None else _default_logger()
        ).bind(component="fred_macro")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> FredMacroDataProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def get_series(
        self,
        series_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[MacroSeriesPoint]:
        params: dict[str, str] = {
            "series_id": series_id,
            "api_key": self._settings.api_key.get_secret_value(),
            "file_type": "json",
        }
        if start is not None:
            params["observation_start"] = start.isoformat()
        if end is not None:
            params["observation_end"] = end.isoformat()

        payload = _http_get_json(
            self._client,
            "/fred/series/observations",
            params=params,
            retries=self._retries,
            base_backoff=self._base_backoff,
            sleep=self._sleep,
            logger=self._logger,
            error_cls=ContextDataError,
        )
        if not isinstance(payload, dict):
            raise ContextDataError("FRED payload was not a JSON object")

        out: list[MacroSeriesPoint] = []
        for raw in payload.get("observations") or []:
            try:
                obs_date = date.fromisoformat(raw["date"])
                raw_value = raw.get("value", "")
            except (KeyError, ValueError) as e:
                raise ContextDataError(f"malformed FRED observation: {e}") from e

            # FRED encodes missing observations as ".".
            value: Decimal | None
            if raw_value in (".", "", None):
                value = None
            else:
                try:
                    value = Decimal(str(raw_value))
                except InvalidOperation:
                    value = None
            out.append(MacroSeriesPoint(series_id=series_id, date=obs_date, value=value))
        return out


# ---------------------------------------------------------------------------
# GDELT news provider
# ---------------------------------------------------------------------------

# GDELT Doc 2.0 ArticleList mode returns basic metadata only — no per-article
# tone or themes. Adding tone/themes would require joining the GKG dataset,
# which is out of scope for this stage. We still expose tone / themes /
# entities on NewsEvent so a future GKG-aware provider can fill them in
# without breaking the schema. GDELT tone (dictionary-based) is good for
# detecting and locating events, not for final sentiment — the agent will
# interpret later (see this module's docstring).
_GDELT_BASE_URL: Final[str] = "https://api.gdeltproject.org"
_GDELT_DOC_PATH: Final[str] = "/api/v2/doc/doc"
_GDELT_TIME_FMT: Final[str] = "%Y%m%d%H%M%S"


class GdeltNewsProvider:
    """GDELT Doc 2.0 ArticleList adapter.

    Dedupes articles by exact URL — GDELT frequently surfaces syndicated
    copies of the same story across multiple domains. We keep the first
    occurrence in the order GDELT returned it (sort=DateDesc keeps that
    "first" = "most recent" by default).
    """

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        retries: int = _DEFAULT_RETRIES,
        base_backoff_seconds: float = _DEFAULT_BASE_BACKOFF_SECONDS,
        logger: structlog.stdlib.BoundLogger | None = None,
        sleep: Callable[[float], None] = _time.sleep,
    ) -> None:
        self._retries = retries
        self._base_backoff = base_backoff_seconds
        self._sleep = sleep
        if client is None:
            self._client = httpx.Client(
                base_url=_GDELT_BASE_URL,
                timeout=httpx.Timeout(timeout_seconds),
                headers={"Accept": "application/json"},
            )
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False
        self._logger = (
            logger if logger is not None else _default_logger()
        ).bind(component="gdelt_news")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> GdeltNewsProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def search(
        self,
        query: str,
        *,
        start: datetime,
        end: datetime,
        max_results: int = 75,
    ) -> list[NewsEvent]:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start/end must be timezone-aware")
        if end <= start:
            raise ValueError("end must be after start")
        if not (1 <= max_results <= 250):
            raise ValueError("max_results must be in [1, 250] (GDELT cap)")

        params: dict[str, str] = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "sort": "DateDesc",
            "maxrecords": str(max_results),
            "startdatetime": start.astimezone(UTC).strftime(_GDELT_TIME_FMT),
            "enddatetime": end.astimezone(UTC).strftime(_GDELT_TIME_FMT),
        }
        payload = _http_get_json(
            self._client,
            _GDELT_DOC_PATH,
            params=params,
            retries=self._retries,
            base_backoff=self._base_backoff,
            sleep=self._sleep,
            logger=self._logger,
            error_cls=ContextDataError,
        )
        if not isinstance(payload, dict):
            raise ContextDataError("GDELT payload was not a JSON object")

        seen_urls: set[str] = set()
        out: list[NewsEvent] = []
        for raw in payload.get("articles") or []:
            event = _parse_gdelt_article(raw)
            if event is None:
                continue
            if event.url in seen_urls:
                continue
            seen_urls.add(event.url)
            out.append(event)
        return out


def _parse_gdelt_article(raw: dict) -> NewsEvent | None:
    """Parse one GDELT article. Returns ``None`` on missing required fields.

    GDELT's outgoing query `startdatetime` / `enddatetime` use the no-T form
    ``YYYYMMDDHHMMSS``, but the incoming ``seendate`` is reported with the T
    separator: ``20260624T080000Z``. We accept both by stripping any ``T``
    before parsing — same logical instant, different surface form.
    """
    try:
        url = raw.get("url")
        title = raw.get("title")
        seendate = raw.get("seendate")
        if not url or not title or not seendate:
            return None
        clean = seendate.replace("T", "")
        ts = datetime.strptime(clean, "%Y%m%d%H%M%SZ").replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None

    source = raw.get("domain") or ""
    tone_raw = raw.get("tone")
    tone: Decimal | None = None
    if tone_raw not in (None, ""):
        try:
            tone = Decimal(str(tone_raw))
        except InvalidOperation:
            tone = None

    return NewsEvent(
        timestamp=ts,
        title=title,
        source=source,
        url=url,
        tone=tone,
        themes=(),
        entities=(),
    )


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
    return structlog.get_logger("core.context_data")


__all__ = [
    "FRED_SERIES",
    "ContextDataError",
    "EconomicCalendarProvider",
    "ForexFactoryCalendarProvider",
    "FredMacroDataProvider",
    "GdeltNewsProvider",
    "MacroDataProvider",
    "NewsProvider",
    "parse_economic_value",
]
