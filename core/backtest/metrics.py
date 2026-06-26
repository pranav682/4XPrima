"""Backtest metrics — RISK-ADJUSTED FIRST.

Total return is the headline number but never the *primary* ranking
number. Sharpe, Sortino, max drawdown, and profit factor come first. A
strategy that doubled equity by sitting through a 70% drawdown is not a
strategy; it's a coin flip with a marketing department.

Annualisation: derived from the average bar interval — H1 ≈ 6048
bars/year (252 trading days * 24 hours), D1 ≈ 252. Override
``periods_per_year`` if your fixture lies about its bar interval.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Risk-adjusted metrics first, raw return second.

    All percent fields are fractions (0.05 = 5%). The *_ratio fields are
    annualised. ``inf`` is a legitimate value for ratios when the
    denominator is zero (no downside / no losing trade) — caller can
    decide how to render that.
    """

    total_return_pct: Decimal
    annualised_return_pct: Decimal
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: Decimal
    win_rate: float  # fraction of closed trades that were positive
    profit_factor: float  # gross wins / gross losses (|.|)
    trade_count: int
    avg_trade_pnl: Decimal
    exposure_pct: float  # share of bars with at least one open position


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def infer_periods_per_year(times: Sequence[object]) -> float:
    """Estimate bars-per-year from a sequence of equity-point times.

    Uses a 252-trading-day year (standard for forex / equities). Falls
    back to 252 (daily) for too-few-bars inputs.
    """
    if len(times) < 2:
        return 252.0
    first = getattr(times[0], "time", times[0])
    last = getattr(times[-1], "time", times[-1])
    delta_seconds = (last - first).total_seconds()
    if delta_seconds <= 0:
        return 252.0
    avg_seconds = delta_seconds / (len(times) - 1)
    if avg_seconds <= 0:
        return 252.0
    seconds_per_year = 252.0 * 24.0 * 3600.0
    return seconds_per_year / avg_seconds


def _floats(values: Sequence[Decimal]) -> list[float]:
    return [float(v) for v in values]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_metrics(
    equity_curve: Sequence,  # EquityPoint sequence
    trade_log: Sequence,  # TradeRecord sequence
    *,
    periods_per_year: float | None = None,
) -> BacktestMetrics:
    """Compute the risk-adjusted metric pack.

    The equity-curve sequence carries ``time`` + ``equity`` per sample;
    trade-log carries ``realized_pnl`` per closed trade. Both are taken
    from the engine's :class:`BacktestResult`.

    ``periods_per_year``: inferred from ``equity_curve`` bar intervals
    when omitted.
    """
    if len(equity_curve) < 2:
        return _empty_metrics()

    equities = _floats([p.equity for p in equity_curve])
    if equities[0] <= 0:
        # No principal → no metric is meaningful. Return empty.
        return _empty_metrics()

    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(equity_curve)

    # 1. Per-bar log-of-returns (well-behaved when equity passes zero).
    #    Use simple returns; for backtests of mostly-bounded trajectories
    #    these are fine and read cleanly.
    returns: list[float] = []
    for i in range(1, len(equities)):
        prev = equities[i - 1]
        cur = equities[i]
        if prev > 0:
            returns.append((cur - prev) / prev)

    # 2. Total + annualised return.
    total_return = (equities[-1] - equities[0]) / equities[0]
    if returns:
        annualised = (1.0 + total_return) ** (periods_per_year / len(returns)) - 1.0
    else:
        annualised = 0.0

    # 3. Sharpe / Sortino — annualised, zero risk-free.
    if returns:
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        std_r = math.sqrt(variance)
        sharpe = (mean_r / std_r) * math.sqrt(periods_per_year) if std_r > 0 else 0.0

        negs = [r for r in returns if r < 0]
        if negs:
            downside_var = sum(r**2 for r in negs) / len(returns)
            downside_std = math.sqrt(downside_var)
            sortino = (
                (mean_r / downside_std) * math.sqrt(periods_per_year) if downside_std > 0 else 0.0
            )
        else:
            sortino = float("inf") if mean_r > 0 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    # 4. Max drawdown from equity curve (independent of returns).
    peak = equities[0]
    max_dd = 0.0
    for e in equities:
        if e > peak:
            peak = e
        if peak > 0:
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd

    # 5. Trade metrics from the closed trades.
    closed = [t for t in trade_log if t.realized_pnl is not None]
    if closed:
        wins = [t for t in closed if t.realized_pnl > 0]
        losses = [t for t in closed if t.realized_pnl < 0]
        win_rate = len(wins) / len(closed)
        gross_wins = sum((t.realized_pnl for t in wins), Decimal("0"))
        gross_losses_abs = abs(sum((t.realized_pnl for t in losses), Decimal("0")))
        if gross_losses_abs > 0:
            profit_factor = float(gross_wins / gross_losses_abs)
        else:
            profit_factor = float("inf") if gross_wins > 0 else 0.0
        avg_trade = sum((t.realized_pnl for t in closed), Decimal("0")) / Decimal(len(closed))
        trade_count = len(closed)
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade = Decimal("0")
        trade_count = 0

    # 6. Exposure: share of equity-curve samples with at least one open.
    if equity_curve:
        bars_with_open = sum(1 for p in equity_curve if p.open_positions > 0)
        exposure = bars_with_open / len(equity_curve)
    else:
        exposure = 0.0

    return BacktestMetrics(
        total_return_pct=Decimal(str(total_return)),
        annualised_return_pct=Decimal(str(annualised)),
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=Decimal(str(max_dd)),
        win_rate=win_rate,
        profit_factor=profit_factor,
        trade_count=trade_count,
        avg_trade_pnl=avg_trade,
        exposure_pct=exposure,
    )


def _empty_metrics() -> BacktestMetrics:
    return BacktestMetrics(
        total_return_pct=Decimal("0"),
        annualised_return_pct=Decimal("0"),
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown_pct=Decimal("0"),
        win_rate=0.0,
        profit_factor=0.0,
        trade_count=0,
        avg_trade_pnl=Decimal("0"),
        exposure_pct=0.0,
    )


# Tag unused imports for static analysers; we deliberately keep the
# `timedelta` import for type-narrowing in `infer_periods_per_year`.
_ = timedelta


__all__ = [
    "BacktestMetrics",
    "compute_metrics",
    "infer_periods_per_year",
]
