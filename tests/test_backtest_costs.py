"""Tests for the backtest cost model and the BacktestBroker's cost accounting.

A backtest that ignores costs is a lie. These tests pin the fill-price
arithmetic, commission, and swap, then prove at the broker level that a
zero-cost round trip and a costed round trip differ by *exactly* the modelled
cost — no more, no less.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from core.backtest.costs import (
    CostModel,
    buy_fill_price,
    commission_for,
    sell_fill_price,
    swap_per_unit_per_day,
)
from core.backtest.engine import BacktestBroker
from core.models import Candle, Direction, OrderRequest

# ---------------------------------------------------------------------------
# Fill-price / commission / swap helpers
# ---------------------------------------------------------------------------


def test_buy_and_sell_fill_prices_move_adversely(costed_model: CostModel) -> None:
    mid = Decimal("1.10000")
    edge = costed_model.half_spread + costed_model.slippage_per_unit
    assert buy_fill_price(mid, costed_model) == mid + edge
    assert sell_fill_price(mid, costed_model) == mid - edge


def test_sell_fill_rejects_nonpositive_price() -> None:
    aggressive = CostModel(half_spread=Decimal("0.6"), slippage_per_unit=Decimal("0.6"))
    with pytest.raises(ValueError):
        sell_fill_price(Decimal("1.0"), aggressive)


def test_commission_scales_with_size(costed_model: CostModel) -> None:
    assert commission_for(Decimal("100000"), costed_model) == Decimal("1.00000")


def test_swap_sign_convention(costed_model: CostModel) -> None:
    # Long pays (negative), short receives a rebate (positive) in this model.
    assert swap_per_unit_per_day(Direction.LONG, costed_model) < 0
    assert swap_per_unit_per_day(Direction.SHORT, costed_model) > 0


# ---------------------------------------------------------------------------
# Broker round-trip: zero-cost vs costed differ by EXACTLY the modelled cost
# ---------------------------------------------------------------------------


def _round_trip(
    model: CostModel,
    *,
    size: Decimal,
    ref_in: Decimal,
    ref_out: Decimal,
) -> tuple[Decimal, Decimal]:
    """Open a long at ``ref_in`` then close it at ``ref_out``.

    Returns ``(ending_cash, total_cost)``. Times are kept inside one day so no
    swap rollover is crossed.
    """
    broker = BacktestBroker(starting_balance=Decimal("10000"), cost_model=model)
    open_bar = Candle(
        pair="EURUSD",
        granularity="H1",
        time=datetime(2025, 1, 6, 10, tzinfo=UTC),
        open=ref_in,
        high=ref_in + Decimal("0.001"),
        low=ref_in - Decimal("0.001"),
        close=ref_in,
        volume=1000,
        complete=True,
    )
    close_bar = open_bar.model_copy(
        update={
            "time": datetime(2025, 1, 6, 12, tzinfo=UTC),
            "open": ref_out,
            "high": ref_out + Decimal("0.001"),
            "low": ref_out - Decimal("0.001"),
            "close": ref_out,
        }
    )
    order = OrderRequest(
        pair="EURUSD",
        direction=Direction.LONG,
        size=size,
        entry_price=ref_in,
        stop_price=ref_in - Decimal("0.01"),
    )
    lot, _ = broker.open_at_next_open(order, open_bar, trade_id="t000000")
    broker.close_at_next_open(lot, close_bar)
    return broker.cash, broker.cost_breakdown.total


def test_zero_cost_round_trip_charges_nothing(zero_cost_model) -> None:
    cash, total = _round_trip(
        zero_cost_model,
        size=Decimal("50000"),
        ref_in=Decimal("1.1000"),
        ref_out=Decimal("1.1100"),
    )
    assert total == Decimal("0")
    # Pure mid-to-mid PnL: (1.11 - 1.10) * 50000 = 500.
    assert cash == Decimal("10000") + Decimal("500.000")


def test_costed_round_trip_differs_by_exactly_the_modelled_cost(
    zero_cost_model, costed_model
) -> None:
    size, ref_in, ref_out = Decimal("50000"), Decimal("1.1000"), Decimal("1.1100")

    zero_cash, zero_total = _round_trip(zero_cost_model, size=size, ref_in=ref_in, ref_out=ref_out)
    costed_cash, costed_total = _round_trip(costed_model, size=size, ref_in=ref_in, ref_out=ref_out)

    assert zero_total == Decimal("0")
    assert costed_total > 0
    # The whole difference in ending cash is the modelled cost — exactly.
    assert zero_cash - costed_cash == costed_total

    # And that total decomposes into the expected components (two fills each).
    expected_spread = 2 * size * costed_model.half_spread
    expected_slippage = 2 * size * costed_model.slippage_per_unit
    expected_commission = 2 * commission_for(size, costed_model)
    breakdown_total = expected_spread + expected_slippage + expected_commission
    assert costed_total == breakdown_total


# ---------------------------------------------------------------------------
# Swap / rollover
# ---------------------------------------------------------------------------


def _open_long(broker: BacktestBroker, size: Decimal, at: datetime) -> None:
    bar = Candle(
        pair="EURUSD",
        granularity="H1",
        time=at,
        open=Decimal("1.1000"),
        high=Decimal("1.1010"),
        low=Decimal("1.0990"),
        close=Decimal("1.1000"),
        volume=1000,
        complete=True,
    )
    order = OrderRequest(
        pair="EURUSD",
        direction=Direction.LONG,
        size=size,
        entry_price=Decimal("1.1000"),
        stop_price=Decimal("1.0900"),
    )
    broker.open_at_next_open(order, bar, trade_id="t000000")


def test_swap_charged_once_when_crossing_one_rollover(costed_model: CostModel) -> None:
    broker = BacktestBroker(
        starting_balance=Decimal("10000"),
        cost_model=costed_model,
        day_rollover_utc_hour=21,
    )
    size = Decimal("10000")
    _open_long(broker, size, datetime(2025, 1, 6, 19, tzinfo=UTC))  # Monday
    cash_before = broker.cash

    # 20:00 -> 22:00 Monday crosses the 21:00 rollover exactly once, no weekend.
    swap = broker.apply_swap_if_rollover(
        Candle(
            pair="EURUSD",
            granularity="H1",
            time=datetime(2025, 1, 6, 20, tzinfo=UTC),
            open=Decimal("1.1"),
            high=Decimal("1.1"),
            low=Decimal("1.1"),
            close=Decimal("1.1"),
            volume=1,
            complete=True,
        ),
        Candle(
            pair="EURUSD",
            granularity="H1",
            time=datetime(2025, 1, 6, 22, tzinfo=UTC),
            open=Decimal("1.1"),
            high=Decimal("1.1"),
            low=Decimal("1.1"),
            close=Decimal("1.1"),
            volume=1,
            complete=True,
        ),
    )
    # Long pays: size * per_unit * 1 rollover * 1 (no weekend).
    expected = size * costed_model.swap_long_per_unit_per_day
    assert swap == expected
    assert broker.cash == cash_before + expected
    assert broker.cost_breakdown.swap == expected


def test_wednesday_rollover_applies_weekend_multiplier(costed_model: CostModel) -> None:
    broker = BacktestBroker(
        starting_balance=Decimal("10000"),
        cost_model=costed_model,
        day_rollover_utc_hour=21,
    )
    size = Decimal("10000")
    _open_long(broker, size, datetime(2025, 1, 8, 19, tzinfo=UTC))  # Wednesday

    swap = broker.apply_swap_if_rollover(
        Candle(
            pair="EURUSD",
            granularity="H1",
            time=datetime(2025, 1, 8, 20, tzinfo=UTC),
            open=Decimal("1.1"),
            high=Decimal("1.1"),
            low=Decimal("1.1"),
            close=Decimal("1.1"),
            volume=1,
            complete=True,
        ),
        Candle(
            pair="EURUSD",
            granularity="H1",
            time=datetime(2025, 1, 8, 22, tzinfo=UTC),
            open=Decimal("1.1"),
            high=Decimal("1.1"),
            low=Decimal("1.1"),
            close=Decimal("1.1"),
            volume=1,
            complete=True,
        ),
    )
    expected = (
        size
        * costed_model.swap_long_per_unit_per_day
        * Decimal(costed_model.weekend_swap_multiplier)
    )
    assert swap == expected


def test_no_swap_without_open_positions(costed_model: CostModel) -> None:
    broker = BacktestBroker(starting_balance=Decimal("10000"), cost_model=costed_model)
    swap = broker.apply_swap_if_rollover(
        Candle(
            pair="EURUSD",
            granularity="H1",
            time=datetime(2025, 1, 6, 20, tzinfo=UTC),
            open=Decimal("1.1"),
            high=Decimal("1.1"),
            low=Decimal("1.1"),
            close=Decimal("1.1"),
            volume=1,
            complete=True,
        ),
        Candle(
            pair="EURUSD",
            granularity="H1",
            time=datetime(2025, 1, 6, 20, tzinfo=UTC) + timedelta(hours=2),
            open=Decimal("1.1"),
            high=Decimal("1.1"),
            low=Decimal("1.1"),
            close=Decimal("1.1"),
            volume=1,
            complete=True,
        ),
    )
    assert swap == Decimal("0")
