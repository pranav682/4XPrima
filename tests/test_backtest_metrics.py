"""Tests for the risk-adjusted metric pack.

Metrics are verified against hand-computable equity curves and trade logs so
the math is pinned, not just smoke-tested. Sharpe/Sortino annualisation is
checked for sign and the clean edge cases (no downside => inf); max drawdown,
profit factor, win rate, total return, average trade, and exposure are checked
to exact values.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal

from core.backtest.metrics import compute_metrics, infer_periods_per_year
from core.backtest.types import EquityPoint, TradeRecord
from core.models import Direction


def _trade(pnl: Decimal | None, *, tid: str = "t") -> TradeRecord:
    """A closed (or open, if pnl is None) trade carrying only what metrics read."""
    return TradeRecord(
        trade_id=tid,
        pair="EURUSD",
        direction=Direction.LONG,
        size=Decimal("1000"),
        entry_time=datetime(2025, 1, 6, tzinfo=UTC),
        entry_price=Decimal("1.1000"),
        entry_commission=Decimal("0"),
        exit_time=None if pnl is None else datetime(2025, 1, 6, 1, tzinfo=UTC),
        exit_price=None if pnl is None else Decimal("1.1100"),
        exit_commission=None if pnl is None else Decimal("0"),
        realized_pnl=pnl,
    )


def test_empty_curve_returns_zeroed_metrics(
    make_equity_point: Callable[..., EquityPoint],
) -> None:
    m = compute_metrics([make_equity_point(0, "100")], [])
    assert m.trade_count == 0
    assert m.sharpe_ratio == 0.0
    assert m.total_return_pct == Decimal("0")


def test_total_return_and_max_drawdown_exact(
    make_equity_point: Callable[..., EquityPoint],
) -> None:
    # Peak 120 then trough 60 => drawdown 0.5; ends at 90 => total return -0.10.
    curve = [
        make_equity_point(0, "100"),
        make_equity_point(1, "120"),
        make_equity_point(2, "60"),
        make_equity_point(3, "90"),
    ]
    m = compute_metrics(curve, [])
    assert m.max_drawdown_pct == Decimal("0.5")
    assert m.total_return_pct == Decimal(str((90 - 100) / 100))


def test_monotonic_increase_has_zero_drawdown_and_positive_sharpe(
    make_equity_point: Callable[..., EquityPoint],
) -> None:
    curve = [
        make_equity_point(0, "100"),
        make_equity_point(1, "101"),
        make_equity_point(2, "103"),
        make_equity_point(3, "104"),
    ]
    m = compute_metrics(curve, [], periods_per_year=252.0)
    assert m.max_drawdown_pct == Decimal("0")
    assert m.sharpe_ratio > 0
    # No negative returns + positive mean => Sortino is infinite by definition.
    assert m.sortino_ratio == float("inf")


def test_declining_curve_has_negative_sharpe(
    make_equity_point: Callable[..., EquityPoint],
) -> None:
    curve = [
        make_equity_point(0, "100"),
        make_equity_point(1, "99"),
        make_equity_point(2, "97"),
        make_equity_point(3, "96"),
    ]
    m = compute_metrics(curve, [], periods_per_year=252.0)
    assert m.sharpe_ratio < 0


def test_profit_factor_win_rate_and_avg_trade_exact(
    make_equity_point: Callable[..., EquityPoint],
) -> None:
    curve = [make_equity_point(0, "100"), make_equity_point(1, "110")]
    trades = [
        _trade(Decimal("30"), tid="a"),
        _trade(Decimal("-10"), tid="b"),
        _trade(Decimal("20"), tid="c"),
        _trade(Decimal("-10"), tid="d"),
        _trade(None, tid="open"),  # open trade is ignored by trade metrics
    ]
    m = compute_metrics(curve, trades)
    assert m.trade_count == 4
    assert m.win_rate == 0.5
    # gross wins 50 / gross losses 20 = 2.5
    assert m.profit_factor == 2.5
    # (30 - 10 + 20 - 10) / 4 = 7.5
    assert m.avg_trade_pnl == Decimal("30") / Decimal("4")


def test_profit_factor_infinite_when_no_losses(
    make_equity_point: Callable[..., EquityPoint],
) -> None:
    curve = [make_equity_point(0, "100"), make_equity_point(1, "110")]
    m = compute_metrics(curve, [_trade(Decimal("10")), _trade(Decimal("5"))])
    assert m.profit_factor == float("inf")
    assert m.win_rate == 1.0


def test_exposure_is_share_of_bars_with_open_positions(
    make_equity_point: Callable[..., EquityPoint],
) -> None:
    curve = [
        make_equity_point(0, "100", open_positions=0),
        make_equity_point(1, "101", open_positions=1),
        make_equity_point(2, "102", open_positions=2),
        make_equity_point(3, "103", open_positions=0),
    ]
    m = compute_metrics(curve, [])
    assert m.exposure_pct == 0.5


def test_infer_periods_per_year_for_hourly_bars(
    make_equity_point: Callable[..., EquityPoint],
) -> None:
    # Hourly samples => 252 trading days * 24h = 6048 periods/year.
    curve = [make_equity_point(i, "100") for i in range(5)]
    assert infer_periods_per_year(curve) == 252.0 * 24.0
