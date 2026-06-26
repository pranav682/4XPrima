"""Tests for the strategy contract: PointInTimeView look-ahead defence and the
MovingAverageCrossover reference strategy.

The look-ahead defence is structural — the view physically stores only the
visible slice, so reaching past bar ``t`` raises rather than silently
returning future data. These tests pin that down directly; the engine-level
end-to-end look-ahead test lives in ``test_backtest_engine.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from core.models import AccountState, Candle, Direction
from core.strategy import (
    LookAheadError,
    MovingAverageCrossover,
    PointInTimeView,
)

# ---------------------------------------------------------------------------
# PointInTimeView — look-ahead defence
# ---------------------------------------------------------------------------


def test_view_exposes_only_visible_slice(make_bars: Callable[..., list[Candle]]) -> None:
    bars = make_bars(["1.10", "1.11", "1.12", "1.13", "1.14"])
    view = PointInTimeView(bars, end_index=2)

    assert len(view) == 3
    assert view.latest is bars[2]
    assert view[0] is bars[0]
    assert view[2] is bars[2]
    assert view[-1] is bars[2]


def test_view_raises_on_read_beyond_bar_t(
    make_bars: Callable[..., list[Candle]],
) -> None:
    bars = make_bars(["1.10", "1.11", "1.12", "1.13", "1.14"])
    view = PointInTimeView(bars, end_index=2)

    # The next bar exists in `bars` but NOT in the view — reading it is the
    # exact look-ahead the structure forbids.
    with pytest.raises(LookAheadError):
        _ = view[3]
    with pytest.raises(LookAheadError):
        _ = view[len(view)]
    with pytest.raises(LookAheadError):
        _ = view[-4]


def test_lookahead_error_is_index_error(
    make_bars: Callable[..., list[Candle]],
) -> None:
    # Subclassing IndexError means existing try/except IndexError branches keep
    # catching look-ahead violations.
    bars = make_bars(["1.10", "1.11", "1.12"])
    view = PointInTimeView(bars, end_index=1)
    assert issubclass(LookAheadError, IndexError)
    with pytest.raises(IndexError):
        _ = view[5]


def test_view_construction_rejects_out_of_range_end_index(
    make_bars: Callable[..., list[Candle]],
) -> None:
    bars = make_bars(["1.10", "1.11", "1.12"])
    with pytest.raises(ValueError):
        PointInTimeView(bars, end_index=3)  # == len(bars)
    with pytest.raises(ValueError):
        PointInTimeView(bars, end_index=-1)


def test_view_lookback_and_closes_cap_at_window(
    make_bars: Callable[..., list[Candle]],
) -> None:
    bars = make_bars(["1.10", "1.11", "1.12", "1.13"])
    view = PointInTimeView(bars, end_index=2)  # visible: 0,1,2

    assert view.closes(2) == (Decimal("1.11"), Decimal("1.12"))
    assert view.closes() == (Decimal("1.10"), Decimal("1.11"), Decimal("1.12"))
    # lookback never reaches past the visible window even when asked for more.
    assert len(view.lookback(10)) == 3
    assert view.lookback(0) == ()


def test_view_cannot_grow_new_attributes(
    make_bars: Callable[..., list[Candle]],
) -> None:
    # __slots__ means a strategy can't smuggle in extra state (e.g. a stashed
    # reference to the full bar list) to widen its visibility.
    bars = make_bars(["1.10", "1.11", "1.12"])
    view = PointInTimeView(bars, end_index=1)
    with pytest.raises(AttributeError):
        view.all_bars = bars  # type: ignore[attr-defined]
    # Backing store is a tuple — not extendable in place.
    assert isinstance(view[:], tuple)


# ---------------------------------------------------------------------------
# MovingAverageCrossover — reference strategy unit behaviour
# ---------------------------------------------------------------------------


def test_ma_crossover_validates_params() -> None:
    with pytest.raises(ValueError):
        MovingAverageCrossover(
            pair="EURUSD",
            fast_period=8,
            slow_period=3,  # fast >= slow
            size=Decimal("1000"),
            stop_distance=Decimal("0.005"),
        )
    with pytest.raises(ValueError):
        MovingAverageCrossover(
            pair="EURUSD",
            fast_period=3,
            slow_period=8,
            size=Decimal("0"),  # non-positive size
            stop_distance=Decimal("0.005"),
        )


def test_ma_crossover_silent_until_enough_bars(
    make_bars: Callable[..., list[Candle]],
) -> None:
    strat = MovingAverageCrossover(
        pair="EURUSD",
        fast_period=2,
        slow_period=4,
        size=Decimal("1000"),
        stop_distance=Decimal("0.005"),
    )
    bars = make_bars(["1.10", "1.11", "1.12"])  # fewer than slow_period
    view = PointInTimeView(bars, end_index=2)
    account = AccountState(
        balance=Decimal("10000"),
        equity=Decimal("10000"),
        peak_equity=Decimal("10000"),
        day_start_equity=Decimal("10000"),
        as_of=datetime(2025, 1, 6, tzinfo=UTC),
    )
    assert strat.decide(view, account, as_of=account.as_of) == []


def test_ma_crossover_emits_long_then_short_on_crosses(
    make_bars: Callable[..., list[Candle]],
) -> None:
    """Drive a clean up-cross then down-cross and assert the signal directions.

    Series rises (fast pulls above slow → LONG) then falls (fast drops below
    slow → SHORT). We feed bars one at a time through fresh views, mirroring
    how the engine calls the strategy.
    """
    closes = [
        "1.100",
        "1.100",
        "1.100",
        "1.100",  # flat warm-up
        "1.110",
        "1.120",
        "1.130",  # rising → up-cross
        "1.110",
        "1.090",
        "1.070",  # falling → down-cross
    ]
    bars = make_bars(closes)
    strat = MovingAverageCrossover(
        pair="EURUSD",
        fast_period=2,
        slow_period=4,
        size=Decimal("1000"),
        stop_distance=Decimal("0.005"),
    )
    account = AccountState(
        balance=Decimal("10000"),
        equity=Decimal("10000"),
        peak_equity=Decimal("10000"),
        day_start_equity=Decimal("10000"),
        as_of=datetime(2025, 1, 6, tzinfo=UTC),
    )

    directions: list[Direction] = []
    for t in range(len(bars)):
        view = PointInTimeView(bars, end_index=t)
        for order in strat.decide(view, account, as_of=bars[t].time):
            directions.append(order.direction)

    assert Direction.LONG in directions
    assert Direction.SHORT in directions
    # First signal is the up-cross (LONG), and a SHORT comes after it.
    assert directions.index(Direction.LONG) < directions.index(Direction.SHORT)
