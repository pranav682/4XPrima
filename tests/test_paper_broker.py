"""Tests for the in-memory paper broker.

Verifies fill math (spread + commission), round-trip P&L, equity / peak-equity
/ drawdown bookkeeping, and the boundary edge cases the user enumerated.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from core.market_data import StubPriceProvider
from core.models import Direction, OrderRequest, Position
from core.paper_broker import PaperBroker, PaperBrokerConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 6, 24, 12, 0, tzinfo=UTC)


@pytest.fixture
def stub_now(now: datetime):
    """A callable that returns a fixed `now` — keeps fill timestamps stable
    so assertions don't have to chase wall-clock drift."""

    def _now() -> datetime:
        return now

    return _now


@pytest.fixture
def prices() -> StubPriceProvider:
    pp = StubPriceProvider()
    # 2-pip market spread on EURUSD.
    pp.set_price("EURUSD", bid=Decimal("1.0848"), ask=Decimal("1.0850"))
    return pp


@pytest.fixture
def broker(prices: StubPriceProvider, stub_now) -> PaperBroker:
    cfg = PaperBrokerConfig(
        starting_balance=Decimal("10000"),
        commission_per_unit=Decimal("0.0001"),    # 0.01 per 100 units
        extra_half_spread=Decimal("0.0001"),      # 1 pip markup per side
    )
    return PaperBroker(cfg, prices, now_fn=stub_now)


def make_order(
    *,
    pair: str = "EURUSD",
    direction: Direction = Direction.LONG,
    size: Decimal = Decimal("10000"),
    entry: Decimal = Decimal("1.0850"),
    stop: Decimal = Decimal("1.0800"),
) -> OrderRequest:
    return OrderRequest(
        pair=pair,
        direction=direction,
        size=size,
        entry_price=entry,
        stop_price=stop,
    )


# ---------------------------------------------------------------------------
# Fill math
# ---------------------------------------------------------------------------


def test_long_fill_price_is_ask_plus_extra_half_spread(
    broker: PaperBroker, stub_now
) -> None:
    fill = broker.place_order(make_order(direction=Direction.LONG))
    # ask = 1.0850, extra_half_spread = 0.0001 → 1.0851
    assert fill.fill_price == Decimal("1.0851")
    assert fill.direction is Direction.LONG
    assert fill.timestamp == stub_now()


def test_short_fill_price_is_bid_minus_extra_half_spread(
    broker: PaperBroker,
) -> None:
    fill = broker.place_order(make_order(direction=Direction.SHORT))
    # bid = 1.0848, extra_half_spread = 0.0001 → 1.0847
    assert fill.fill_price == Decimal("1.0847")
    assert fill.direction is Direction.SHORT


def test_commission_is_size_times_per_unit(broker: PaperBroker) -> None:
    fill = broker.place_order(make_order(size=Decimal("12345")))
    # commission_per_unit = 0.0001 → 12345 * 0.0001 = 1.2345
    assert fill.commission == Decimal("1.23450000")
    # And cash dropped by exactly that.
    expected_cash = Decimal("10000") - Decimal("1.23450000")
    assert broker.cash == expected_cash


def test_open_then_close_long_round_trip_realizes_pnl(
    broker: PaperBroker, prices: StubPriceProvider, now: datetime
) -> None:
    # Open: ask 1.0850 + half_spread 0.0001 → entry 1.0851
    open_fill = broker.place_order(
        make_order(direction=Direction.LONG, size=Decimal("10000"))
    )
    assert open_fill.fill_price == Decimal("1.0851")

    # Price moves up 50 pips: bid 1.0898, ask 1.0900.
    prices.set_price(
        "EURUSD", bid=Decimal("1.0898"), ask=Decimal("1.0900"), timestamp=now
    )
    position = broker.get_open_positions()[0]
    close_fill = broker.close_position(position)
    # Close-side for a long = bid - half_spread → 1.0898 - 0.0001 = 1.0897
    assert close_fill.fill_price == Decimal("1.0897")
    assert close_fill.direction is Direction.SHORT

    # P&L = (close - open) * size - 2 * commission
    #     = (1.0897 - 1.0851) * 10_000 - 2 * (10_000 * 0.0001)
    #     = 0.0046 * 10_000 - 2.0
    #     = 46 - 2 = 44
    expected_pnl_after_costs = Decimal("44")
    expected_cash = Decimal("10000") + expected_pnl_after_costs
    assert broker.cash == expected_cash
    # And the position is gone.
    assert broker.get_open_positions() == []


def test_open_then_close_short_round_trip_realizes_pnl(
    broker: PaperBroker, prices: StubPriceProvider, now: datetime
) -> None:
    # Open short: bid 1.0848 - 0.0001 = 1.0847
    broker.place_order(make_order(direction=Direction.SHORT, size=Decimal("10000")))

    # Price moves DOWN 50 pips → favourable for a short.
    prices.set_price(
        "EURUSD", bid=Decimal("1.0798"), ask=Decimal("1.0800"), timestamp=now
    )
    position = broker.get_open_positions()[0]
    close_fill = broker.close_position(position)
    # Close-side for short = ask + half_spread → 1.0800 + 0.0001 = 1.0801
    assert close_fill.fill_price == Decimal("1.0801")
    assert close_fill.direction is Direction.LONG

    # P&L = (entry - close) * size - 2 * commission
    #     = (1.0847 - 1.0801) * 10_000 - 2
    #     = 0.0046 * 10_000 - 2 = 44
    assert broker.cash == Decimal("10044")


# ---------------------------------------------------------------------------
# Account-state arithmetic
# ---------------------------------------------------------------------------


def test_account_state_initial_snapshot(broker: PaperBroker) -> None:
    state = broker.get_account_state()
    assert state.balance == Decimal("10000")
    assert state.equity == Decimal("10000")
    assert state.unrealized_pnl == Decimal("0")
    assert state.peak_equity == Decimal("10000")
    assert state.day_start_equity == Decimal("10000")
    assert state.open_positions == ()


def test_unrealized_pnl_uses_close_side_marks(
    broker: PaperBroker, prices: StubPriceProvider, now: datetime
) -> None:
    broker.place_order(make_order(direction=Direction.LONG, size=Decimal("10000")))
    # entry = 1.0851 (ask + half_spread)
    # Move price up; mark-to-bid (1.0898) is the conservative number.
    prices.set_price(
        "EURUSD", bid=Decimal("1.0898"), ask=Decimal("1.0900"), timestamp=now
    )
    state = broker.get_account_state()
    # unrealized = (bid - entry) * size = (1.0898 - 1.0851) * 10_000 = 47
    assert state.unrealized_pnl == Decimal("47.0000")
    # equity = cash (10_000 - 1 commission) + unrealized 47 = 10_046
    assert state.equity == Decimal("10046.0000")


def test_peak_equity_advances_only_on_new_highs(
    broker: PaperBroker, prices: StubPriceProvider, now: datetime
) -> None:
    broker.place_order(make_order(direction=Direction.LONG, size=Decimal("10000")))
    # Push up → new peak.
    prices.set_price(
        "EURUSD", bid=Decimal("1.0898"), ask=Decimal("1.0900"), timestamp=now
    )
    high_state = broker.get_account_state()
    high_peak = high_state.peak_equity
    assert high_peak == high_state.equity

    # Drop back → peak does not retreat.
    prices.set_price(
        "EURUSD", bid=Decimal("1.0848"), ask=Decimal("1.0850"), timestamp=now
    )
    low_state = broker.get_account_state()
    assert low_state.equity < high_peak
    assert low_state.peak_equity == high_peak
    assert low_state.drawdown_pct > 0


def test_drawdown_is_zero_at_or_above_peak(broker: PaperBroker) -> None:
    state = broker.get_account_state()
    assert state.drawdown_pct == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_closing_position_not_open_raises(
    broker: PaperBroker, now: datetime
) -> None:
    bogus = Position(
        pair="EURUSD",
        direction=Direction.LONG,
        size=Decimal("100"),
        entry_price=Decimal("1.0"),
        stop_price=Decimal("0.99"),
    )
    with pytest.raises(ValueError, match="position not open"):
        broker.close_position(bogus)


def test_insufficient_balance_for_commission_raises(
    prices: StubPriceProvider, stub_now
) -> None:
    # Tiny balance, large position → commission > cash.
    cfg = PaperBrokerConfig(
        starting_balance=Decimal("1"),
        commission_per_unit=Decimal("0.001"),
        extra_half_spread=Decimal("0"),
    )
    poor = PaperBroker(cfg, prices, now_fn=stub_now)
    with pytest.raises(ValueError, match="insufficient cash"):
        poor.place_order(make_order(size=Decimal("100000")))


def test_zero_quote_construction_raises() -> None:
    # The Quote model itself rejects non-positive prices — Field(gt=0).
    from pydantic import ValidationError

    from core.models import Quote

    with pytest.raises(ValidationError):
        Quote(
            pair="EURUSD",
            bid=Decimal("0"),
            ask=Decimal("0"),
            timestamp=datetime.now(UTC),
        )


def test_quote_inversion_raises() -> None:
    from pydantic import ValidationError

    from core.models import Quote

    with pytest.raises(ValidationError, match="inverted quote"):
        Quote(
            pair="EURUSD",
            bid=Decimal("1.0850"),
            ask=Decimal("1.0840"),
            timestamp=datetime.now(UTC),
        )


def test_short_close_with_negative_effective_price_raises(
    prices: StubPriceProvider, stub_now
) -> None:
    # Construct a setup where extra_half_spread would push a SELL fill below
    # zero. Using a tiny price.
    prices.set_price("EURUSD", bid=Decimal("0.0001"), ask=Decimal("0.0002"))
    cfg = PaperBrokerConfig(
        starting_balance=Decimal("10000"),
        commission_per_unit=Decimal("0"),
        extra_half_spread=Decimal("0.001"),  # bigger than bid → effective < 0
    )
    rb = PaperBroker(cfg, prices, now_fn=stub_now)
    with pytest.raises(ValueError, match="non-positive"):
        rb.place_order(
            make_order(
                direction=Direction.SHORT, entry=Decimal("0.0002"), stop=Decimal("0.0003")
            )
        )


def test_stub_price_provider_unknown_pair_raises(prices: StubPriceProvider) -> None:
    with pytest.raises(KeyError):
        prices.get_quote("GBPUSD")


def test_stub_price_provider_key_pair_mismatch_raises() -> None:
    from core.models import Quote

    pp = StubPriceProvider()
    q = Quote(
        pair="EURUSD",
        bid=Decimal("1.0848"),
        ask=Decimal("1.0850"),
        timestamp=datetime.now(UTC),
    )
    with pytest.raises(ValueError, match="does not match"):
        pp.set_quote("GBPUSD", q)
