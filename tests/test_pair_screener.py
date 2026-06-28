"""Tests for the deterministic structural pair screener.

Hermetic: synthetic candles + in-memory provider stubs, no network. The
optional live smoke at the bottom is skipped unless RUN_LIVE_TESTS=1 and
OANDA_API_TOKEN are set.

Two themes run through these tests:
- metric correctness on known inputs (vol, ATR, autocorrelation, variance
  ratio, correlation, gaps), and
- the selection is STRUCTURAL — it drops correlated pairs and prefers low
  cost-to-move, and exposes NO historical-return / profitability field
  anywhere (the guard against selection bias).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from core.analysis.pair_screener import (
    CorrelationMatrix,
    ExclusionEntry,
    PairScreener,
    ScreenConfig,
    ScreeningReport,
    ShortlistEntry,
    autocorrelation,
    average_true_range,
    detect_gaps,
    pearson,
    realized_volatility,
    simple_returns,
    variance_ratio,
)
from core.market_data import StubPriceProvider
from core.models import Candle, Granularity, PairProfile

BASE = datetime(2024, 1, 1, tzinfo=UTC)  # a Monday


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _candle(
    pair: str,
    time: datetime,
    close: Decimal | str,
    *,
    open: Decimal | str | None = None,
    high: Decimal | str | None = None,
    low: Decimal | str | None = None,
) -> Candle:
    c = Decimal(str(close))
    o = c if open is None else Decimal(str(open))
    hi = (max(o, c) + Decimal("0.002")) if high is None else Decimal(str(high))
    lo = (min(o, c) - Decimal("0.002")) if low is None else Decimal(str(low))
    return Candle(
        pair=pair,
        granularity=Granularity.D,
        time=time,
        open=o,
        high=hi,
        low=lo,
        close=c,
        volume=1000,
        complete=True,
    )


def _closes_from_returns(start: float, returns: list[float]) -> list[Decimal]:
    px = Decimal(str(start))
    out = [px]
    for r in returns:
        px = px * (Decimal("1") + Decimal(str(r)))
        out.append(px)
    return out


def _daily(pair: str, closes: list[Decimal], *, start: datetime = BASE) -> list[Candle]:
    return [_candle(pair, start + timedelta(days=i), c) for i, c in enumerate(closes)]


class StubCandleProvider:
    """In-memory CandleProvider keyed by canonical pair code."""

    def __init__(self, candles_by_pair: dict[str, list[Candle]]) -> None:
        self._d = candles_by_pair

    def get_candles(
        self,
        pair: str,
        *,
        granularity: Granularity,
        count: int | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ) -> list[Candle]:
        canon = pair.replace("/", "").replace("_", "").replace("-", "").upper()
        candles = self._d.get(canon, [])
        if count is not None:
            return candles[-count:]
        return candles


def _prices(spreads: dict[str, tuple[str, str]]) -> StubPriceProvider:
    """Build a StubPriceProvider from {pair: (bid, ask)}."""
    p = StubPriceProvider()
    for pair, (bid, ask) in spreads.items():
        p.set_price(pair, bid=Decimal(bid), ask=Decimal(ask), timestamp=BASE)
    return p


# ---------------------------------------------------------------------------
# Metric correctness (known inputs)
# ---------------------------------------------------------------------------


def test_simple_returns_known() -> None:
    rs = simple_returns([Decimal("100"), Decimal("110"), Decimal("121")])
    assert rs == pytest.approx([0.1, 0.1])


def test_realized_volatility_known() -> None:
    # returns [.1,-.1,.1,-.1]: mean 0, std 0.1; * sqrt(4) = 0.2
    vol = realized_volatility([0.1, -0.1, 0.1, -0.1], 4.0)
    assert vol == pytest.approx(0.2)
    assert realized_volatility([0.1], 4.0) == 0.0  # too short


def test_average_true_range_known() -> None:
    candles = [
        _candle("EURUSD", BASE, "9", high="10", low="8"),
        _candle("EURUSD", BASE + timedelta(days=1), "10", high="11", low="9"),
        _candle("EURUSD", BASE + timedelta(days=2), "11.5", high="12", low="11"),
    ]
    # TRs: bar0 = 10-8 = 2; bar1 = max(2, |11-9|, |9-9|) = 2;
    #      bar2 = max(1, |12-10|, |11-10|) = 2  -> ATR = 2
    assert average_true_range(candles) == Decimal("2")
    assert average_true_range([]) == Decimal("0")


def test_autocorrelation_known() -> None:
    alt = [1.0, -1.0] * 4  # n=8
    assert autocorrelation(alt, 1) == pytest.approx(-7 / 8)
    block = [1.0, 1.0, -1.0, -1.0] * 2
    assert autocorrelation(block, 1) == pytest.approx(0.125)
    assert autocorrelation([0.01] * 6, 1) == 0.0  # zero variance
    assert autocorrelation([1.0, 2.0], 1) == 0.0  # too short


def test_variance_ratio_known() -> None:
    assert variance_ratio([1.0, -1.0] * 4, 2) == pytest.approx(0.0)  # mean-reverting
    assert variance_ratio([1.0, 1.0, -1.0, -1.0] * 2, 2) == pytest.approx(8 / 7)  # >1
    assert variance_ratio([1.0, -1.0], 5) == 1.0  # too short -> neutral


def test_pearson_known() -> None:
    assert pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert pearson([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]) == pytest.approx(-1.0)
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0  # zero variance -> 0


def test_detect_gaps_counts_holes_not_weekends() -> None:
    # Consecutive daily bars: no gaps.
    closes = [Decimal("1.10")] * 6
    candles = _daily("EURUSD", closes)
    assert detect_gaps(candles, Granularity.D) == (0, 0)

    # Inject a 5-day hole mid-Wednesday (not a weekend): one gap, 4 missing bars.
    holed = [
        _candle("EURUSD", BASE, "1.10"),  # Mon
        _candle("EURUSD", BASE + timedelta(days=1), "1.10"),  # Tue
        _candle("EURUSD", BASE + timedelta(days=2), "1.10"),  # Wed
        _candle("EURUSD", BASE + timedelta(days=7), "1.10"),  # +5 days
        _candle("EURUSD", BASE + timedelta(days=8), "1.10"),
    ]
    assert detect_gaps(holed, Granularity.D) == (1, 4)

    # Thu -> Fri -> Mon: the Fri->Mon weekend is NOT a gap.
    weekend = [
        _candle("EURUSD", BASE + timedelta(days=3), "1.10"),  # Thu
        _candle("EURUSD", BASE + timedelta(days=4), "1.10"),  # Fri
        _candle("EURUSD", BASE + timedelta(days=7), "1.10"),  # Mon
    ]
    assert detect_gaps(weekend, Granularity.D) == (0, 0)


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------


def _three_pair_setup() -> tuple[StubCandleProvider, StubPriceProvider]:
    """EURUSD == GBPUSD (corr 1.0); AUDUSD decorrelated (~0.33)."""
    eur_r = [0.01 if i % 2 == 0 else -0.01 for i in range(120)]
    aud_r = [0.01 if (i // 3) % 2 == 0 else -0.01 for i in range(120)]
    candles = {
        "EURUSD": _daily("EURUSD", _closes_from_returns(1.10, eur_r)),
        "GBPUSD": _daily("GBPUSD", _closes_from_returns(1.27, eur_r)),  # identical moves
        "AUDUSD": _daily("AUDUSD", _closes_from_returns(0.66, aud_r)),
    }
    prices = _prices(
        {
            "EURUSD": ("1.10000", "1.10010"),  # cheapest
            "GBPUSD": ("1.27000", "1.27030"),  # pricier than EUR
            "AUDUSD": ("0.66000", "0.66012"),
        }
    )
    return StubCandleProvider(candles), prices


def test_correlation_matrix_structure() -> None:
    candles, prices = _three_pair_setup()
    report = PairScreener(candles, price_provider=prices).screen(
        ScreenConfig(
            pairs=("EURUSD", "GBPUSD", "AUDUSD"),
            lookback_count=120,
            min_candles=100,
        ),
        as_of=BASE,
    )
    corr = report.correlation
    # Diagonal is 1.0, matrix symmetric.
    for p in corr.pairs:
        assert corr.get(p, p) == pytest.approx(1.0)
    assert corr.get("EURUSD", "GBPUSD") == pytest.approx(corr.get("GBPUSD", "EURUSD"))
    # Identical moves -> ~1.0; decorrelated -> well under the 0.8 threshold.
    assert corr.get("EURUSD", "GBPUSD") == pytest.approx(1.0, abs=1e-6)
    assert abs(corr.get("EURUSD", "AUDUSD")) < 0.8


# ---------------------------------------------------------------------------
# Shortlist: drops correlated pairs, prefers low cost-to-move
# ---------------------------------------------------------------------------


def test_shortlist_drops_correlated_and_prefers_low_cost() -> None:
    candles, prices = _three_pair_setup()
    report = PairScreener(candles, price_provider=prices).screen(
        ScreenConfig(
            pairs=("EURUSD", "GBPUSD", "AUDUSD"),
            lookback_count=120,
            shortlist_size=3,
            min_candles=100,
            max_correlation=0.80,
        ),
        as_of=BASE,
    )
    picked = [e.pair for e in report.shortlist]
    excluded = {e.pair: e.reason for e in report.excluded}

    # EURUSD (cheapest) chosen first; AUDUSD kept (decorrelated); GBPUSD dropped
    # because it is ~perfectly correlated with the already-picked EURUSD.
    assert picked == ["EURUSD", "AUDUSD"]
    assert "GBPUSD" in excluded
    assert "correlation" in excluded["GBPUSD"].lower()
    # EURUSD beat GBPUSD purely on cost-to-move (lower spread/ATR), not return.
    assert report.shortlist[0].pair == "EURUSD"
    assert report.shortlist[0].selection_rank == 1


def test_selection_ignores_return_prefers_cheaper_pair() -> None:
    """Anti-selection-bias: a strong-uptrend (high return) but expensive pair is
    NOT preferred over a flat-but-cheap pair. Cost-to-move decides, not return."""
    trend_r = [0.01] * 120  # large positive cumulative return
    flat_r = [0.004 if i % 2 == 0 else -0.004 for i in range(120)]  # ~zero net return
    candles = StubCandleProvider(
        {
            "EURUSD": _daily("EURUSD", _closes_from_returns(1.10, trend_r)),
            "AUDUSD": _daily("AUDUSD", _closes_from_returns(0.66, flat_r)),
        }
    )
    # EURUSD made big money historically but is EXPENSIVE to move; AUDUSD is flat
    # but CHEAP. With one slot, the cheap one wins.
    prices = _prices(
        {
            "EURUSD": ("1.10000", "1.20000"),  # absurdly wide spread -> expensive
            "AUDUSD": ("0.66000", "0.66010"),  # tight -> cheap
        }
    )
    report = PairScreener(candles, price_provider=prices).screen(
        ScreenConfig(
            pairs=("EURUSD", "AUDUSD"),
            lookback_count=120,
            shortlist_size=1,
            min_candles=100,
            max_spread_to_atr=None,  # don't filter; let cost ORDER decide
        ),
        as_of=BASE,
    )
    assert [e.pair for e in report.shortlist] == ["AUDUSD"]
    # The high-return pair is NOT selected.
    assert "EURUSD" not in {e.pair for e in report.shortlist}


# ---------------------------------------------------------------------------
# Missing / short data handled gracefully
# ---------------------------------------------------------------------------


def test_short_and_missing_data_handled_gracefully() -> None:
    candles = StubCandleProvider(
        {
            "EURUSD": _daily("EURUSD", _closes_from_returns(1.10, [0.001] * 120)),
            "GBPUSD": _daily("GBPUSD", [Decimal("1.27"), Decimal("1.271")]),  # 2 bars
            # AUDUSD: no data at all (provider returns []).
        }
    )
    report = PairScreener(candles, price_provider=None).screen(
        ScreenConfig(
            pairs=("EURUSD", "GBPUSD", "AUDUSD"),
            lookback_count=120,
            min_candles=100,
        ),
        as_of=BASE,
    )
    profiles = {p.pair: p for p in report.profiles}
    excluded = {e.pair: e.reason for e in report.excluded}

    # No crash; every candidate produced a profile.
    assert set(profiles) == {"EURUSD", "GBPUSD", "AUDUSD"}
    assert profiles["AUDUSD"].candle_count == 0
    assert profiles["GBPUSD"].candle_count == 2
    # Short / empty pairs are excluded for insufficient data.
    assert "insufficient data" in excluded["GBPUSD"]
    assert "insufficient data" in excluded["AUDUSD"]
    # The descriptor degrades gracefully, not crashes.
    assert profiles["AUDUSD"].behavior_descriptor == "insufficient data"


def test_empty_candidate_set_does_not_crash() -> None:
    report = PairScreener(StubCandleProvider({})).screen(
        ScreenConfig(pairs=("EURUSD",), lookback_count=120),
        as_of=BASE,
    )
    assert report.shortlist == ()
    assert report.correlation.pairs == ()
    assert "PAIR SCREEN" in report.render()  # render still works


# ---------------------------------------------------------------------------
# Selection-bias guard: NO return / profitability field anywhere
# ---------------------------------------------------------------------------


def test_no_return_or_profitability_field_exposed() -> None:
    """Hard guard against selection bias: none of the screener's data structures
    may expose a historical-return or profit/performance ranking field."""
    forbidden = (
        "return",
        "profit",
        "pnl",
        "sharpe",
        "sortino",
        "alpha",
        "expectancy",
        "momentum",
        "performance",
        "gain",
        "yield",
        "score",
        "winrate",
        "win_rate",
        "cumulative",
    )
    models = (
        PairProfile,
        ScreenConfig,
        CorrelationMatrix,
        ShortlistEntry,
        ExclusionEntry,
        ScreeningReport,
    )
    for model in models:
        for field_name in model.model_fields:
            low = field_name.lower()
            for bad in forbidden:
                assert bad not in low, f"{model.__name__}.{field_name} looks profit-derived"

    # And the structural fields we DO rely on are present.
    assert "spread_to_atr" in PairProfile.model_fields
    assert "coverage_ratio" in PairProfile.model_fields
    assert "variance_ratio" in PairProfile.model_fields
    assert "cost_to_move" in ShortlistEntry.model_fields


def test_report_renders_and_serialises() -> None:
    candles, prices = _three_pair_setup()
    report = PairScreener(candles, price_provider=prices).screen(
        ScreenConfig(pairs=("EURUSD", "GBPUSD", "AUDUSD"), lookback_count=120),
        as_of=BASE,
    )
    text = report.render()
    assert "PROFILES" in text
    assert "RETURN CORRELATION MATRIX" in text
    assert "SHORTLIST" in text
    # JSON-serialisable for saving to samples/.
    dumped = report.model_dump(mode="json")
    assert dumped["candidate_pairs"] == ["EURUSD", "GBPUSD", "AUDUSD"]


# ---------------------------------------------------------------------------
# Optional live smoke (skipped in CI)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1" or not os.environ.get("OANDA_API_TOKEN"),
    reason="RUN_LIVE_TESTS=1 and OANDA_API_TOKEN must be set",
)
def test_live_smoke_oanda() -> None:
    from core.config import OandaSettings
    from core.market_data import OandaPriceProvider

    settings = OandaSettings()  # type: ignore[call-arg]
    with OandaPriceProvider(settings) as provider:
        report = PairScreener(provider, price_provider=provider).screen(
            ScreenConfig(
                pairs=("EURUSD", "USDJPY", "GBPUSD"),
                granularity=Granularity.D,
                lookback_count=120,
                shortlist_size=2,
            ),
            as_of=datetime.now(UTC),
        )
    assert len(report.profiles) == 3
    assert all(p.candle_count > 0 for p in report.profiles)
    assert len(report.shortlist) <= 2
