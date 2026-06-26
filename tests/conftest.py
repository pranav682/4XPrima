"""Shared fixtures for the backtester + strategy test suite.

All factories are hermetic — they build synthetic candles and risk configs in
memory. No network, no recorded broker data needed. Prices are ``Decimal``
throughout so cost/PnL assertions are exact (never float-fuzzy).

Factory fixtures return *callables* so each test can shape its own data.
Names are deliberately distinct from the per-module fixtures in the existing
suite (``risk_config``, ``account``, ``now``) so nothing is shadowed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from core.backtest.costs import CostModel
from core.backtest.types import EquityPoint
from core.models import Candle, Direction, Granularity, OrderRequest, RiskConfig

# A Monday, so callers can reason about weekday-dependent behaviour (the
# Wednesday swap multiplier) by adding whole days.
BASE_TIME = datetime(2025, 1, 6, 0, 0, tzinfo=UTC)


def _dec(value: Decimal | str | int | float) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


@pytest.fixture
def make_candle() -> Callable[..., Candle]:
    """Build one OHLCV bar with consistent OHLC.

    ``open`` defaults to ``close``; high/low are padded out so the bar always
    satisfies ``low <= open,close <= high``.
    """

    def _make(
        close: Decimal | str | int | float,
        time: datetime,
        *,
        pair: str = "EURUSD",
        open: Decimal | str | int | float | None = None,
        granularity: Granularity = Granularity.H1,
        volume: int = 1000,
        pad: Decimal | str = Decimal("0.0005"),
        complete: bool = True,
    ) -> Candle:
        c = _dec(close)
        o = c if open is None else _dec(open)
        p = _dec(pad)
        return Candle(
            pair=pair,
            granularity=granularity,
            time=time,
            open=o,
            high=max(o, c) + p,
            low=min(o, c) - p,
            close=c,
            volume=volume,
            complete=complete,
        )

    return _make


@pytest.fixture
def make_bars(make_candle: Callable[..., Candle]) -> Callable[..., list[Candle]]:
    """Build a list of hourly bars from a sequence of closes.

    Optionally pass ``opens`` (same length) to make the next-bar-open fill
    price observably different from the prior bar's close.
    """

    def _make(
        closes: Sequence[Decimal | str | int | float],
        *,
        start: datetime = BASE_TIME,
        step: timedelta = timedelta(hours=1),
        opens: Sequence[Decimal | str | int | float] | None = None,
        pair: str = "EURUSD",
        granularity: Granularity = Granularity.H1,
    ) -> list[Candle]:
        bars: list[Candle] = []
        for i, close in enumerate(closes):
            bars.append(
                make_candle(
                    close,
                    start + i * step,
                    pair=pair,
                    open=None if opens is None else opens[i],
                    granularity=granularity,
                )
            )
        return bars

    return _make


@pytest.fixture
def make_equity_point() -> Callable[..., EquityPoint]:
    """Build an EquityPoint; metrics only read ``time``/``equity``/``open_positions``."""

    def _make(
        bar_index: int,
        equity: Decimal | str | int | float,
        *,
        time: datetime | None = None,
        open_positions: int = 0,
        step: timedelta = timedelta(hours=1),
    ) -> EquityPoint:
        eq = _dec(equity)
        return EquityPoint(
            bar_index=bar_index,
            time=time if time is not None else BASE_TIME + bar_index * step,
            balance=eq,
            equity=eq,
            drawdown_pct=Decimal("0"),
            open_positions=open_positions,
        )

    return _make


@pytest.fixture
def make_order() -> Callable[..., OrderRequest]:
    """Build an OrderRequest with a stop placed ``stop_distance`` away."""

    def _make(
        *,
        pair: str = "EURUSD",
        direction: Direction = Direction.LONG,
        size: Decimal | str | int | float = "1000",
        entry_price: Decimal | str | int | float = "1.1000",
        stop_distance: Decimal | str | int | float = "0.0050",
    ) -> OrderRequest:
        entry = _dec(entry_price)
        dist = _dec(stop_distance)
        stop = entry - dist if direction is Direction.LONG else entry + dist
        return OrderRequest(
            pair=pair,
            direction=direction,
            size=_dec(size),
            entry_price=entry,
            stop_price=stop,
        )

    return _make


@pytest.fixture
def roomy_risk_config() -> RiskConfig:
    """Caps roomy enough that ordinary reference-strategy orders pass cleanly.

    Individual tests tighten a single cap to exercise a specific rejection or
    resize path.
    """
    return RiskConfig(
        max_risk_per_trade_pct=Decimal("0.10"),
        max_portfolio_risk_pct=Decimal("0.50"),
        max_concurrent_positions=10,
        max_exposure_per_pair_pct=Decimal("100"),
        max_correlated_exposure_pct=Decimal("100"),
        correlation_groups={"USD_QUOTE": ("EURUSD", "GBPUSD", "AUDUSD")},
        daily_loss_limit_pct=Decimal("0.50"),
        max_drawdown_pct=Decimal("0.50"),
    )


@pytest.fixture
def zero_cost_model() -> CostModel:
    """A cost model that charges nothing — fills land exactly at the mid."""
    return CostModel(
        half_spread=Decimal("0"),
        commission_per_unit=Decimal("0"),
        slippage_per_unit=Decimal("0"),
        swap_long_per_unit_per_day=Decimal("0"),
        swap_short_per_unit_per_day=Decimal("0"),
        weekend_swap_multiplier=3,
    )


@pytest.fixture
def costed_model() -> CostModel:
    """A realistic, round-numbered cost model for EURUSD-class pairs."""
    return CostModel(
        half_spread=Decimal("0.0001"),
        commission_per_unit=Decimal("0.00001"),
        slippage_per_unit=Decimal("0.00005"),
        swap_long_per_unit_per_day=Decimal("-0.00001"),
        swap_short_per_unit_per_day=Decimal("0.00001"),
        weekend_swap_multiplier=3,
    )
