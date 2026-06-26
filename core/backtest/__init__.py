"""Deterministic backtesting engine.

Ground-truth for the slow-loop improvement cycle: backtest results are the
only signal that promotes a strategy. Predictive correctness lives here,
not in any LLM judge.

Two cardinal sins the engine refuses to commit:

1. **Look-ahead bias.** At bar ``t`` a strategy sees a
   :class:`core.strategy.PointInTimeView` that physically stores only
   bars ``0..t``. Signals at ``t`` fill at the **next** bar's open
   (``t+1``), never at the current bar's close. See
   :mod:`core.strategy`.

2. **Dishonest fills.** Every fill pays the modelled spread, commission,
   and per-unit slippage; overnight positions accrue swap. A backtest
   that ignores costs is a lie. See :mod:`core.backtest.costs`.

Every order routes through the same :class:`core.risk_manager.RiskManager`
the live system uses — there is exactly ONE risk gate, and the backtester
honours it. A drawdown breach trips the kill switch and halts the run.

Run with :class:`BacktestEngine`. Metrics, walk-forward windowing, and
report rendering live in this package.
"""

from core.backtest.costs import CostBreakdown, CostModel
from core.backtest.engine import (
    BacktestBroker,
    BacktestEngine,
)
from core.backtest.metrics import BacktestMetrics, compute_metrics
from core.backtest.types import (
    BacktestResult,
    EquityPoint,
    TradeRecord,
)
from core.backtest.walkforward import (
    DataSplit,
    OutOfSampleAccessError,
    WalkForwardConfig,
    walk_forward_windows,
)

__all__ = [
    "BacktestBroker",
    "BacktestEngine",
    "BacktestMetrics",
    "BacktestResult",
    "CostBreakdown",
    "CostModel",
    "DataSplit",
    "EquityPoint",
    "OutOfSampleAccessError",
    "TradeRecord",
    "WalkForwardConfig",
    "compute_metrics",
    "walk_forward_windows",
]
