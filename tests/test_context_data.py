"""Tests for the context-data adapters — fully hermetic.

No network, no API keys. Recorded JSON fixtures + httpx.MockTransport.
Optional live smoke tests at the bottom, skipped unless RUN_LIVE_TESTS=1
(and FRED_API_KEY for the FRED smoke).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from core.config import FredSettings
from core.context_data import (
    FRED_SERIES,
    ContextDataError,
    ForexFactoryCalendarProvider,
    FredMacroDataProvider,
    GdeltNewsProvider,
    parse_economic_value,
)
from core.models import ImpactLevel

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> object:
    return json.loads((FIXTURES / name).read_text())


def _client(transport: httpx.MockTransport, *, base_url: str) -> httpx.Client:
    return httpx.Client(
        base_url=base_url,
        transport=transport,
        timeout=httpx.Timeout(5.0),
        headers={"Accept": "application/json"},
    )


# ---------------------------------------------------------------------------
# parse_economic_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("170K", Decimal("170000")),
        ("1.2M", Decimal("1200000.0")),
        ("5.2B", Decimal("5200000000.0")),
        ("-18.6K", Decimal("-18600.0")),
        ("3.5%", Decimal("3.5")),
        ("+0.7%", Decimal("0.7")),
        ("$5.2B", Decimal("5200000000.0")),
        ("1,234", Decimal("1234")),
        ("0", Decimal("0")),
        ("", None),
        ("   ", None),
        (None, None),
        ("Tentative", None),
        ("All Day", None),
    ],
)
def test_parse_economic_value(raw: str | None, expected: Decimal | None) -> None:
    assert parse_economic_value(raw) == expected


# ---------------------------------------------------------------------------
# Forex Factory calendar
# ---------------------------------------------------------------------------


def _ff_provider(handler) -> ForexFactoryCalendarProvider:
    client = _client(httpx.MockTransport(handler), base_url="https://nfs.faireconomy.media")
    # Pin "now" inside the THIS-WEEK window so only this_week is fetched.
    return ForexFactoryCalendarProvider(
        client=client,
        sleep=lambda *_: None,
        base_backoff_seconds=0.0,
        now_fn=lambda: datetime(2026, 6, 22, 8, 0, tzinfo=UTC),
    )


def test_ff_upcoming_parses_and_filters() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(200, json=_load("ff_calendar_thisweek.json"))

    with _ff_provider(handler) as p:
        events = p.upcoming(timedelta(days=3))

    assert captured["path"] == "/ff_calendar_thisweek.json"
    # In a 3-day window starting 2026-06-22 08:00 UTC, we expect: CAD CPI,
    # USD Flash PMI, AUD Employment Change. NOT the EUR speech (before
    # window) and NOT JPY/USD Friday events.
    titles = [(e.currency, e.name) for e in events]
    assert ("CAD", "CPI m/m") in titles
    assert ("USD", "Flash Manufacturing PMI") in titles
    assert ("AUD", "Employment Change") in titles


def test_ff_surprise_when_actual_present() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_load("ff_calendar_thisweek.json"))

    with _ff_provider(handler) as p:
        events = p.upcoming(timedelta(days=3))

    cpi = next(e for e in events if e.currency == "CAD" and e.name == "CPI m/m")
    assert cpi.actual == Decimal("0.8")
    assert cpi.forecast == Decimal("0.7")
    # 0.8 - 0.7 = 0.1
    assert cpi.surprise == Decimal("0.1")


def test_ff_surprise_is_none_when_actual_missing() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_load("ff_calendar_thisweek.json"))

    with _ff_provider(handler) as p:
        events = p.upcoming(timedelta(days=3))

    pmi = next(e for e in events if e.currency == "USD" and e.name == "Flash Manufacturing PMI")
    assert pmi.actual is None
    assert pmi.forecast == Decimal("54.6")
    assert pmi.surprise is None


def test_ff_currency_and_impact_filters() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_load("ff_calendar_thisweek.json"))

    with _ff_provider(handler) as p:
        events = p.upcoming(
            timedelta(days=7),
            currencies=["USD"],
            min_impact=ImpactLevel.HIGH,
        )
    # USD Retail Sales is past day-7 of our pinned "now" actually within 4 days
    # so let's just check filter semantics — only USD, only HIGH or above.
    assert all(e.currency == "USD" for e in events)
    assert all(e.impact == ImpactLevel.HIGH for e in events)


def test_ff_recent_window_uses_this_week() -> None:
    captured_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_paths.append(request.url.path)
        return httpx.Response(200, json=_load("ff_calendar_thisweek.json"))

    with _ff_provider(handler) as p:
        events = p.recent(timedelta(days=1))

    assert "/ff_calendar_thisweek.json" in captured_paths
    # Within 1 day before pinned now (2026-06-22 08:00 UTC) → only the EUR
    # speech at 2026-06-22T05:00-04:00 = 09:00 UTC is AFTER now, so skipped.
    # The event list may be empty here, which is fine — the assertion is on
    # the fetch path.
    for e in events:
        assert e.when <= datetime(2026, 6, 22, 8, 0, tzinfo=UTC)


def test_ff_rejects_non_positive_window() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    with _ff_provider(handler) as p:
        with pytest.raises(ValueError, match="window must be positive"):
            p.upcoming(timedelta(0))
        with pytest.raises(ValueError, match="window must be positive"):
            p.recent(timedelta(0))


def test_ff_skips_malformed_rows() -> None:
    payload = [
        {"title": "Good", "country": "USD", "date": "2026-06-22T10:00:00-04:00", "impact": "High"},
        {"title": "Bad", "country": "USD"},  # missing date
        {"title": "Bad date", "country": "USD", "date": "not a date", "impact": "High"},
        {"title": "Naive date", "country": "USD", "date": "2026-06-22T10:00:00", "impact": "High"},
    ]

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with _ff_provider(handler) as p:
        events = p.upcoming(timedelta(days=7))
    # Only the first row should survive.
    assert len(events) == 1
    assert events[0].name == "Good"


def test_ff_retries_on_503_then_succeeds() -> None:
    calls: list[int] = []

    def handler(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) <= 2:
            return httpx.Response(503, json={"message": "down"})
        return httpx.Response(200, json=_load("ff_calendar_thisweek.json"))

    with _ff_provider(handler) as p:
        events = p.upcoming(timedelta(days=3))
    assert len(events) >= 1
    assert len(calls) == 3


def test_ff_transport_timeout_raises_context_data_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("simulated")

    with _ff_provider(handler) as p:
        with pytest.raises(ContextDataError, match="transport error"):
            p.upcoming(timedelta(days=3))


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------


def _fred_settings() -> FredSettings:
    return FredSettings(api_key=SecretStr("test-fred-key-not-real"))


def _fred_provider(handler) -> FredMacroDataProvider:
    client = _client(httpx.MockTransport(handler), base_url="https://api.stlouisfed.org")
    return FredMacroDataProvider(
        _fred_settings(),
        client=client,
        sleep=lambda *_: None,
        base_backoff_seconds=0.0,
    )


def test_fred_get_series_parses_fixture() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["query"] = dict(request.url.params)
        return httpx.Response(200, json=_load("fred_observations_cpi.json"))

    with _fred_provider(handler) as p:
        series = p.get_series(
            FRED_SERIES["US_CPI"],
            start=date(2026, 1, 1),
            end=date(2026, 6, 24),
        )

    assert captured["path"] == "/fred/series/observations"
    q = captured["query"]
    assert q["series_id"] == "CPIAUCSL"
    assert q["file_type"] == "json"
    assert q["observation_start"] == "2026-01-01"
    assert q["observation_end"] == "2026-06-24"
    # The api_key IS sent — verify it ends up where it should and only there.
    assert q["api_key"] == "test-fred-key-not-real"

    assert len(series) == 5
    assert series[0].series_id == "CPIAUCSL"
    assert series[0].date == date(2026, 1, 1)
    assert series[0].value == Decimal("319.123")
    # Missing observation ("." in FRED) → value=None, not zero.
    assert series[2].date == date(2026, 3, 1)
    assert series[2].value is None


def test_fred_error_does_not_leak_api_key() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error_message": "bad request"})

    with _fred_provider(handler) as p:
        with pytest.raises(ContextDataError) as excinfo:
            p.get_series("CPIAUCSL")
    assert "test-fred-key-not-real" not in str(excinfo.value)


def test_fred_settings_repr_does_not_leak_key() -> None:
    s = FredSettings(api_key=SecretStr("supersecret"))
    assert "supersecret" not in repr(s)


def test_fred_malformed_observation_raises() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"observations": [{"date": "not-a-date", "value": "1.0"}]},
        )

    with _fred_provider(handler) as p:
        with pytest.raises(ContextDataError, match="malformed FRED observation"):
            p.get_series("CPIAUCSL")


def test_fred_non_object_payload_raises() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    with _fred_provider(handler) as p:
        with pytest.raises(ContextDataError, match="not a JSON object"):
            p.get_series("CPIAUCSL")


def test_fred_persistent_429_exhausts_retries() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": "0"},
            json={"error_message": "rate limit"},
        )

    client = _client(
        httpx.MockTransport(handler), base_url="https://api.stlouisfed.org"
    )
    p = FredMacroDataProvider(
        _fred_settings(),
        client=client,
        sleep=lambda *_: None,
        base_backoff_seconds=0.0,
        retries=2,
    )
    with pytest.raises(ContextDataError, match="HTTP 429"):
        p.get_series("CPIAUCSL")


# ---------------------------------------------------------------------------
# GDELT
# ---------------------------------------------------------------------------


def _gdelt_provider(handler) -> GdeltNewsProvider:
    client = _client(
        httpx.MockTransport(handler), base_url="https://api.gdeltproject.org"
    )
    return GdeltNewsProvider(
        client=client,
        sleep=lambda *_: None,
        base_backoff_seconds=0.0,
    )


def test_gdelt_search_parses_and_dedupes_by_url() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["query"] = dict(request.url.params)
        return httpx.Response(200, json=_load("gdelt_doc_search.json"))

    start = datetime(2026, 6, 24, 0, 0, tzinfo=UTC)
    end = datetime(2026, 6, 24, 23, 59, 59, tzinfo=UTC)
    with _gdelt_provider(handler) as p:
        events = p.search("oil OR eurusd", start=start, end=end, max_results=10)

    assert captured["path"] == "/api/v2/doc/doc"
    q = captured["query"]
    assert q["mode"] == "ArtList"
    assert q["format"] == "json"
    assert q["startdatetime"] == "20260624000000"
    assert q["enddatetime"] == "20260624235959"
    assert q["sort"] == "DateDesc"

    # 5 articles in fixture; the duplicate-URL example.com link is deduped,
    # the empty-title row is skipped → 3 events.
    assert len(events) == 3
    urls = [e.url for e in events]
    assert "https://example.com/oil-prices-rise-2026-06-24" in urls
    assert urls.count("https://example.com/oil-prices-rise-2026-06-24") == 1
    # First occurrence kept (the one with the real title, not "(duplicate)").
    first = next(e for e in events if e.url.startswith("https://example.com/"))
    assert "duplicate" not in first.title.lower()

    # Tone passed through when present.
    tonal = next(e for e in events if e.url.startswith("https://another-news"))
    assert tonal.tone == Decimal("-2.5")


def test_gdelt_rejects_naive_datetimes() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"articles": []})

    with _gdelt_provider(handler) as p:
        with pytest.raises(ValueError, match="timezone-aware"):
            p.search(
                "x",
                start=datetime(2026, 6, 24),  # naive
                end=datetime(2026, 6, 25, tzinfo=UTC),
            )


def test_gdelt_rejects_reversed_window() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"articles": []})

    start = datetime(2026, 6, 24, tzinfo=UTC)
    with _gdelt_provider(handler) as p:
        with pytest.raises(ValueError, match="end must be after"):
            p.search("x", start=start, end=start)


def test_gdelt_rejects_max_results_out_of_range() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"articles": []})

    start = datetime(2026, 6, 24, tzinfo=UTC)
    end = datetime(2026, 6, 25, tzinfo=UTC)
    with _gdelt_provider(handler) as p:
        with pytest.raises(ValueError, match="GDELT cap"):
            p.search("x", start=start, end=end, max_results=0)
        with pytest.raises(ValueError, match="GDELT cap"):
            p.search("x", start=start, end=end, max_results=300)


def test_gdelt_non_object_payload_raises() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    start = datetime(2026, 6, 24, tzinfo=UTC)
    end = datetime(2026, 6, 25, tzinfo=UTC)
    with _gdelt_provider(handler) as p:
        with pytest.raises(ContextDataError, match="not a JSON object"):
            p.search("x", start=start, end=end)


def test_gdelt_retries_on_429_then_succeeds() -> None:
    calls: list[int] = []

    def handler(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
                json={"message": "rate limit"},
            )
        return httpx.Response(200, json=_load("gdelt_doc_search.json"))

    start = datetime(2026, 6, 24, tzinfo=UTC)
    end = datetime(2026, 6, 25, tzinfo=UTC)
    with _gdelt_provider(handler) as p:
        events = p.search("x", start=start, end=end)
    assert len(events) == 3
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Optional live smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="RUN_LIVE_TESTS=1 not set",
)
def test_live_forex_factory_upcoming() -> None:
    with ForexFactoryCalendarProvider() as p:
        events = p.upcoming(timedelta(days=2))
    # The feed may or may not have events in the next 48h depending on the
    # week, so we only assert the call succeeds and returns a list of typed
    # events.
    assert isinstance(events, list)


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1" or not os.environ.get("FRED_API_KEY"),
    reason="RUN_LIVE_TESTS=1 and FRED_API_KEY must be set",
)
def test_live_fred_get_cpi() -> None:
    with FredMacroDataProvider(FredSettings()) as p:
        series = p.get_series(
            FRED_SERIES["US_CPI"],
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
        )
    assert len(series) > 0
    assert any(s.value is not None for s in series)


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="RUN_LIVE_TESTS=1 not set",
)
def test_live_gdelt_search() -> None:
    end = datetime.now(UTC)
    start = end - timedelta(hours=6)
    with GdeltNewsProvider() as p:
        events = p.search("eurusd", start=start, end=end, max_results=5)
    assert isinstance(events, list)
