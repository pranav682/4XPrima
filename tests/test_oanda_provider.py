"""Tests for OandaPriceProvider — no network, no real token.

Uses recorded JSON fixtures and httpx's ``MockTransport`` so the suite is
fully hermetic. The optional live smoke test at the bottom is skipped
unless ``RUN_LIVE_TESTS=1`` and ``OANDA_API_TOKEN`` are both set, so CI
never depends on the network.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from core.config import OandaEnv, OandaSettings
from core.market_data import (
    MarketDataError,
    OandaPriceProvider,
    to_canonical_pair,
    to_oanda_instrument,
)
from core.models import Granularity

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _settings() -> OandaSettings:
    # Fake values — the MockTransport intercepts before any real network
    # call would happen, so the token never leaves the test process.
    return OandaSettings(
        api_token=SecretStr("test-token-not-real"),
        account_id="101-001-00000000-001",
        env=OandaEnv.PRACTICE,
    )


def _provider(transport: httpx.MockTransport, *, retries: int = 3) -> OandaPriceProvider:
    settings = _settings()
    # We construct the httpx.Client manually so we can inject MockTransport,
    # and ALSO provide the same headers the production code sets.
    client = httpx.Client(
        base_url=settings.base_url,
        transport=transport,
        headers={
            "Authorization": f"Bearer {settings.api_token.get_secret_value()}",
            "Accept-Datetime-Format": "RFC3339",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(5.0),
    )
    # `sleep=lambda *_: None` so backoff doesn't actually sleep in tests.
    return OandaPriceProvider(
        settings,
        client=client,
        retries=retries,
        base_backoff_seconds=0.0,
        sleep=lambda *_: None,
    )


# ---------------------------------------------------------------------------
# Pair-format mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_, expected",
    [
        ("EUR/USD", "EUR_USD"),
        ("EURUSD", "EUR_USD"),
        ("EUR_USD", "EUR_USD"),
        ("eur/usd", "EUR_USD"),
        ("Eur-Usd", "EUR_USD"),
    ],
)
def test_to_oanda_instrument(input_: str, expected: str) -> None:
    assert to_oanda_instrument(input_) == expected


def test_to_oanda_instrument_rejects_bad_format() -> None:
    with pytest.raises(ValueError):
        to_oanda_instrument("EURUSDX")
    with pytest.raises(ValueError):
        to_oanda_instrument("EUR")
    with pytest.raises(ValueError):
        to_oanda_instrument("EUR/123")


def test_to_canonical_pair_strips_underscore() -> None:
    assert to_canonical_pair("EUR_USD") == "EURUSD"


# ---------------------------------------------------------------------------
# get_quote — parsing
# ---------------------------------------------------------------------------


def test_get_quote_parses_fixture() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["query"] = dict(request.url.params)
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=_load("oanda_pricing_eurusd.json"))

    provider = _provider(httpx.MockTransport(handler))
    quote = provider.get_quote("EUR/USD")

    assert captured["path"] == "/v3/accounts/101-001-00000000-001/pricing"
    assert captured["query"] == {"instruments": "EUR_USD"}
    # The token IS sent to OANDA — verify the header is shaped correctly.
    assert captured["auth"] == "Bearer test-token-not-real"

    assert quote.pair == "EURUSD"          # canonical, isalpha()
    assert quote.bid == Decimal("1.08470")
    assert quote.ask == Decimal("1.08500")
    assert quote.timestamp.tzinfo is UTC
    # Nanosecond suffix truncated to microseconds (no rounding).
    assert quote.timestamp == datetime(2026, 6, 24, 8, 0, 0, 123456, tzinfo=UTC)


def test_get_quote_raises_when_instrument_missing_from_response() -> None:
    payload = _load("oanda_pricing_eurusd.json")
    payload["prices"] = []  # broker returned nothing for our pair

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    provider = _provider(httpx.MockTransport(handler))
    with pytest.raises(MarketDataError, match="no price entry"):
        provider.get_quote("EUR/USD")


def test_get_quote_raises_on_malformed_payload() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"prices": [{"instrument": "EUR_USD"}]})

    provider = _provider(httpx.MockTransport(handler))
    with pytest.raises(MarketDataError, match="malformed pricing payload"):
        provider.get_quote("EUR/USD")


# ---------------------------------------------------------------------------
# get_candles — parsing
# ---------------------------------------------------------------------------


def test_get_candles_parses_fixture() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["query"] = dict(request.url.params)
        return httpx.Response(200, json=_load("oanda_candles_eurusd_h1.json"))

    provider = _provider(httpx.MockTransport(handler))
    candles = provider.get_candles("EURUSD", granularity=Granularity.H1, count=4)

    assert captured["path"] == "/v3/instruments/EUR_USD/candles"
    assert captured["query"] == {"price": "M", "granularity": "H1", "count": "4"}

    assert len(candles) == 4
    first = candles[0]
    assert first.pair == "EURUSD"
    assert first.granularity is Granularity.H1
    assert first.time == datetime(2026, 6, 24, 5, 0, tzinfo=UTC)
    assert first.open == Decimal("1.08350")
    assert first.high == Decimal("1.08495")
    assert first.low == Decimal("1.08340")
    assert first.close == Decimal("1.08470")
    assert first.volume == 5421
    assert first.complete is True

    # Forming bar at the tail.
    assert candles[-1].complete is False


def test_get_candles_uses_from_to_when_provided() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["query"] = dict(request.url.params)
        return httpx.Response(200, json=_load("oanda_candles_eurusd_h1.json"))

    provider = _provider(httpx.MockTransport(handler))
    f = datetime(2026, 6, 24, 5, 0, tzinfo=UTC)
    t = datetime(2026, 6, 24, 9, 0, tzinfo=UTC)
    provider.get_candles("EURUSD", granularity=Granularity.H1, from_time=f, to_time=t)

    q = captured["query"]
    assert q["granularity"] == "H1"
    assert q["price"] == "M"
    assert "from" in q and q["from"].startswith("2026-06-24T05:00:00")
    assert "to" in q and q["to"].startswith("2026-06-24T09:00:00")
    assert "count" not in q


def test_get_candles_rejects_no_window_spec() -> None:
    provider = _provider(httpx.MockTransport(lambda _: httpx.Response(200, json={})))
    with pytest.raises(ValueError, match="count or from_time"):
        provider.get_candles("EURUSD", granularity=Granularity.H1)


def test_get_candles_rejects_mixed_window_spec() -> None:
    provider = _provider(httpx.MockTransport(lambda _: httpx.Response(200, json={})))
    with pytest.raises(ValueError, match="not both"):
        provider.get_candles(
            "EURUSD",
            granularity=Granularity.H1,
            count=10,
            from_time=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_get_candles_raises_on_malformed_candle() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candles": [{"time": "2026-06-24T05:00:00Z", "complete": True}]},
        )

    provider = _provider(httpx.MockTransport(handler))
    with pytest.raises(MarketDataError, match="malformed candle"):
        provider.get_candles("EURUSD", granularity=Granularity.H1, count=1)


# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------


def test_retries_on_429_then_succeeds() -> None:
    calls: list[int] = []
    payload = _load("oanda_pricing_eurusd.json")

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
                json={"errorMessage": "rate limit"},
            )
        return httpx.Response(200, json=payload)

    provider = _provider(httpx.MockTransport(handler))
    quote = provider.get_quote("EUR/USD")
    assert quote.bid == Decimal("1.08470")
    assert len(calls) == 2  # one retry then success


def test_retries_on_503_then_succeeds() -> None:
    calls: list[int] = []
    payload = _load("oanda_pricing_eurusd.json")

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) <= 2:
            return httpx.Response(503, json={"errorMessage": "service unavailable"})
        return httpx.Response(200, json=payload)

    provider = _provider(httpx.MockTransport(handler))
    quote = provider.get_quote("EUR/USD")
    assert quote.bid == Decimal("1.08470")
    assert len(calls) == 3


def test_retries_exhausted_on_persistent_429() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": "0"},
            json={"errorMessage": "rate limit"},
        )

    provider = _provider(httpx.MockTransport(handler), retries=2)
    with pytest.raises(MarketDataError, match="HTTP 429"):
        provider.get_quote("EUR/USD")


def test_non_retryable_4xx_raises_immediately() -> None:
    calls: list[int] = []

    def handler(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"errorMessage": "unauthorized"})

    provider = _provider(httpx.MockTransport(handler))
    with pytest.raises(MarketDataError, match="HTTP 401"):
        provider.get_quote("EUR/USD")
    assert len(calls) == 1  # no retry on auth failure


def test_transport_timeout_eventually_raises_market_data_error() -> None:
    calls: list[int] = []

    def handler(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectTimeout("simulated timeout")

    provider = _provider(httpx.MockTransport(handler), retries=2)
    with pytest.raises(MarketDataError, match="transport error"):
        provider.get_quote("EUR/USD")
    # 1 initial + 2 retries = 3 attempts
    assert len(calls) == 3


def test_non_json_response_raises() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    provider = _provider(httpx.MockTransport(handler))
    with pytest.raises(MarketDataError, match="non-JSON"):
        provider.get_quote("EUR/USD")


# ---------------------------------------------------------------------------
# Secret hygiene — the token must never appear in repr or error messages
# ---------------------------------------------------------------------------


def test_settings_repr_does_not_leak_token() -> None:
    s = OandaSettings(
        api_token=SecretStr("supersecret"),
        account_id="101-001-00000000-001",
        env=OandaEnv.PRACTICE,
    )
    assert "supersecret" not in repr(s)


def test_provider_error_message_does_not_include_token() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"errorMessage": "unauthorized"})

    provider = _provider(httpx.MockTransport(handler))
    try:
        provider.get_quote("EUR/USD")
    except MarketDataError as e:
        assert "test-token-not-real" not in str(e)
    else:
        pytest.fail("expected MarketDataError")


# ---------------------------------------------------------------------------
# Optional live smoke test (against OANDA practice)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1" or not os.environ.get("OANDA_API_TOKEN"),
    reason="RUN_LIVE_TESTS=1 and OANDA_API_TOKEN must be set",
)
def test_live_practice_get_quote() -> None:
    """Real-network smoke test against the practice endpoint. Off by default."""
    settings = OandaSettings()  # reads from environment / .env
    with OandaPriceProvider(settings) as provider:
        quote = provider.get_quote("EUR/USD")
    assert quote.pair == "EURUSD"
    assert quote.bid > 0
    assert quote.ask >= quote.bid
