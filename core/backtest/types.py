"""Shared data shapes for the backtest engine.

Kept in a single module so :mod:`engine`, :mod:`metrics`,
:mod:`walkforward`, and :mod:`report` can depend on the same frozen
structures without circular imports.

Determinism: every type here is hashable and serialisable; ``BacktestResult``
carries the ``config_hash`` of the inputs so two identical runs are easy to
compare and CI can verify "same inputs → same hash".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from core.models import Direction

# ---------------------------------------------------------------------------
# Per-bar equity samples
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """One sample of the equity curve, taken at end-of-bar (after
    mark-to-market, before any fills queued for the next bar)."""

    bar_index: int
    time: datetime  # UTC, the closing time of bar ``bar_index``
    balance: Decimal  # cash balance
    equity: Decimal  # balance + unrealized PnL
    drawdown_pct: Decimal  # versus rolling peak
    open_positions: int


# ---------------------------------------------------------------------------
# Trade lifecycle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """One trade across its full life: entry, optional exit, and the
    costs paid for both ends + any overnight swap accumulated."""

    trade_id: str
    pair: str
    direction: Direction
    size: Decimal
    entry_time: datetime
    entry_price: Decimal  # actual fill price (not the strategy's estimate)
    entry_commission: Decimal
    exit_time: datetime | None = None
    exit_price: Decimal | None = None
    exit_commission: Decimal | None = None
    swap_paid: Decimal = field(default_factory=lambda: Decimal("0"))
    realized_pnl: Decimal | None = None  # populated on exit

    @property
    def is_closed(self) -> bool:
        return self.exit_time is not None


# ---------------------------------------------------------------------------
# Full result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """The output of one :meth:`BacktestEngine.run` call.

    Deterministic: any two runs with identical inputs produce identical
    ``config_hash`` and identical other fields. The hash is computed from
    a canonical JSON of (strategy params + risk config + cost model +
    bar count + start/end times + starting balance).
    """

    config_hash: str
    pair: str
    bars_total: int
    bars_processed: int  # less than total if halted mid-run
    start_time: datetime
    end_time: datetime

    starting_balance: Decimal
    ending_balance: Decimal
    ending_equity: Decimal
    peak_equity: Decimal
    max_drawdown_pct: Decimal

    n_signals_proposed: int
    n_signals_accepted: int
    n_signals_rejected: int

    halted_due_to_kill_switch: bool
    halted_at_bar_index: int | None
    halt_reason: str | None

    metrics: BacktestMetrics
    cost_breakdown: CostBreakdown
    equity_curve: tuple[EquityPoint, ...]
    trade_log: tuple[TradeRecord, ...]


# Avoid circular imports for the type hints used above by referring via
# forward strings; the real classes live in costs / metrics modules.
from core.backtest.costs import CostBreakdown  # noqa: E402
from core.backtest.metrics import BacktestMetrics  # noqa: E402

__all__ = [
    "BacktestResult",
    "EquityPoint",
    "TradeRecord",
]
