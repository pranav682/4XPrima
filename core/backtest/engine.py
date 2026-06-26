"""Event-driven bar-by-bar backtest engine.

This is the ground-truth component the slow loop's improvement cycle
depends on. It owes you three things:

1. **No look-ahead.** Strategies see a structurally bounded
   :class:`core.strategy.PointInTimeView`. Signals at bar ``t`` fill at
   bar ``t+1``'s open, with cost. See :mod:`core.strategy` for the view
   semantics.

2. **Honest fills.** Every fill pays modelled spread, commission, and
   slippage. Overnight positions accrue swap. See
   :mod:`core.backtest.costs`.

3. **One risk gate.** Every order routes through the same
   :class:`core.risk_manager.RiskManager` the live system uses. The
   backtester does NOT have a parallel risk path. Tripping the kill
   switch (via drawdown or daily loss) halts the run cleanly.

Determinism: identical bars + strategy state + risk config + cost model +
starting balance ⇒ identical ``config_hash`` ⇒ identical
:class:`BacktestResult`. No randomness inside the engine.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import structlog

from core.backtest.costs import (
    CostBreakdown,
    CostModel,
    buy_fill_price,
    commission_for,
    sell_fill_price,
    swap_per_unit_per_day,
)
from core.backtest.metrics import compute_metrics
from core.backtest.types import (
    BacktestResult,
    EquityPoint,
    TradeRecord,
)
from core.models import (
    AccountState,
    Candle,
    Direction,
    OrderRequest,
    Position,
    RiskConfig,
)
from core.risk_manager import RiskManager
from core.strategy import PointInTimeView, Strategy

# ---------------------------------------------------------------------------
# BacktestBroker — bar-driven, deterministic
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _OpenLot:
    """Internal: one open position with its trade-id bookkeeping."""

    position: Position
    trade_id: str
    entry_commission: Decimal
    swap_accumulated: Decimal


class BacktestBroker:
    """In-process broker driven by bars, not live quotes.

    Mutable internal state (cash, open lots, accumulators) but every
    snapshot it hands out is a frozen :class:`AccountState`. Costs are
    attributed to a shared :class:`CostBreakdown`.

    Day rollover is a UTC-hour boundary (default 21:00 UTC, ≈ 17:00 NY).
    Wednesday rollover charges the weekend multiplier from
    :class:`CostModel`.
    """

    def __init__(
        self,
        *,
        starting_balance: Decimal,
        cost_model: CostModel,
        day_rollover_utc_hour: int = 21,
    ) -> None:
        if starting_balance <= 0:
            raise ValueError("starting_balance must be positive")
        if not 0 <= day_rollover_utc_hour <= 23:
            raise ValueError("day_rollover_utc_hour must be 0..23")
        self._cost_model = cost_model
        self._rollover_hour = day_rollover_utc_hour
        self._cash: Decimal = starting_balance
        self._open_lots: list[_OpenLot] = []
        self._realized_pnl_today: Decimal = Decimal("0")
        self._peak_equity: Decimal = starting_balance
        self._day_start_equity: Decimal = starting_balance
        self._current_marked_mid: dict[str, Decimal] = {}
        self._current_day: date | None = None
        self._cost_breakdown = CostBreakdown()
        # Track per-trade swap totals so TradeRecord ends up complete.
        self._trade_swap_acc: dict[str, Decimal] = {}

    # ---------------------------------------------------------------- props

    @property
    def cash(self) -> Decimal:
        return self._cash

    @property
    def open_lots(self) -> tuple[_OpenLot, ...]:
        return tuple(self._open_lots)

    @property
    def cost_breakdown(self) -> CostBreakdown:
        return self._cost_breakdown

    # ----------------------------------------------------- daily bookkeeping

    def maybe_roll_day(self, bar: Candle) -> None:
        """Reset the daily counters at the configured UTC hour."""
        day = self._bar_day(bar)
        if self._current_day is None:
            self._current_day = day
            return
        if day != self._current_day:
            # New trading day — capture starting equity and zero realized.
            equity = self._cash + self._unrealized_pnl()
            # day_start_equity must be > 0 (pydantic constraint on
            # AccountState). If equity is <= 0 we'd be wiped out anyway;
            # use a small positive sentinel so AccountState constructs
            # cleanly and the kill-switch checks fire.
            self._day_start_equity = max(equity, Decimal("0.01"))
            self._realized_pnl_today = Decimal("0")
            self._current_day = day

    def _bar_day(self, bar: Candle) -> date:
        # Anchor the trading day at the rollover hour. A bar timestamped
        # 22:00 UTC belongs to the NEXT trading day if rollover is at 21.
        ts = bar.time
        if ts.hour < self._rollover_hour:
            return ts.date()
        return (ts + timedelta(days=1)).date()

    # -------------------------------------------------------- mark to market

    def mark_to_market(self, bar: Candle) -> None:
        """Record the current mid for the pair of every open lot and
        update the rolling peak equity. Idempotent within a bar."""
        # We only need the mid for the bar's pair; bars are single-pair
        # in this stage's design. Defensive: handle multi-pair bars
        # transparently by always updating the seen pair.
        self._current_marked_mid[bar.pair] = bar.close
        equity = self._cash + self._unrealized_pnl()
        if equity > self._peak_equity:
            self._peak_equity = equity

    # ----------------------------------------------------------- snapshot

    def snapshot(self, *, as_of: datetime) -> AccountState:
        unrealized = self._unrealized_pnl()
        equity = self._cash + unrealized
        # peak_equity / day_start_equity must satisfy PositiveDecimal in
        # AccountState. They're always > 0 here by construction.
        return AccountState(
            balance=self._cash,
            equity=equity,
            open_positions=tuple(lot.position for lot in self._open_lots),
            realized_pnl_today=self._realized_pnl_today,
            unrealized_pnl=unrealized,
            peak_equity=self._peak_equity,
            day_start_equity=self._day_start_equity,
            as_of=as_of,
        )

    def _unrealized_pnl(self) -> Decimal:
        total = Decimal("0")
        for lot in self._open_lots:
            mid = self._current_marked_mid.get(lot.position.pair)
            if mid is None:
                continue
            total += self._mark_lot(lot, mid)
        return total

    def _mark_lot(self, lot: _OpenLot, mid: Decimal) -> Decimal:
        # Mark to the side we'd close at — conservative.
        if lot.position.direction is Direction.LONG:
            close_side = sell_fill_price(mid, self._cost_model)
            return (close_side - lot.position.entry_price) * lot.position.size
        close_side = buy_fill_price(mid, self._cost_model)
        return (lot.position.entry_price - close_side) * lot.position.size

    # --------------------------------------------------------- fills

    def open_at_next_open(
        self,
        order: OrderRequest,
        next_bar: Candle,
        *,
        trade_id: str,
    ) -> tuple[_OpenLot, Decimal]:
        """Open a position at ``next_bar.open`` with full cost modelling.

        Returns ``(_OpenLot, fill_price)``. Updates cash and the cost
        breakdown.
        """
        ref = next_bar.open
        if order.direction is Direction.LONG:
            fill_price = buy_fill_price(ref, self._cost_model)
        else:
            fill_price = sell_fill_price(ref, self._cost_model)

        commission = commission_for(order.size, self._cost_model)
        spread_paid = order.size * self._cost_model.half_spread
        slippage_paid = order.size * self._cost_model.slippage_per_unit

        self._cash -= commission
        self._cost_breakdown.spread_cost += spread_paid
        self._cost_breakdown.slippage += slippage_paid
        self._cost_breakdown.commission += commission

        position = Position(
            pair=order.pair,
            direction=order.direction,
            size=order.size,
            entry_price=fill_price,
            stop_price=order.stop_price,
        )
        lot = _OpenLot(
            position=position,
            trade_id=trade_id,
            entry_commission=commission,
            swap_accumulated=Decimal("0"),
        )
        self._open_lots.append(lot)
        self._trade_swap_acc[trade_id] = Decimal("0")
        return lot, fill_price

    def close_at_next_open(
        self,
        lot: _OpenLot,
        next_bar: Candle,
    ) -> tuple[Decimal, Decimal, Decimal]:
        """Close ``lot`` at ``next_bar.open`` with full cost modelling.

        Returns ``(exit_price, exit_commission, realized_pnl)``.
        """
        ref = next_bar.open
        if lot.position.direction is Direction.LONG:
            exit_price = sell_fill_price(ref, self._cost_model)
            pnl = (exit_price - lot.position.entry_price) * lot.position.size
        else:
            exit_price = buy_fill_price(ref, self._cost_model)
            pnl = (lot.position.entry_price - exit_price) * lot.position.size

        commission = commission_for(lot.position.size, self._cost_model)
        spread_paid = lot.position.size * self._cost_model.half_spread
        slippage_paid = lot.position.size * self._cost_model.slippage_per_unit

        self._cash += pnl - commission
        self._realized_pnl_today += pnl - commission
        self._cost_breakdown.spread_cost += spread_paid
        self._cost_breakdown.slippage += slippage_paid
        self._cost_breakdown.commission += commission

        # Remove the lot.
        self._open_lots = [lot_ for lot_ in self._open_lots if lot_ is not lot]
        return exit_price, commission, pnl

    # ----------------------------------------------------------- swap

    def apply_swap_if_rollover(self, current_bar: Candle, next_bar: Candle) -> Decimal:
        """If the gap from ``current_bar`` to ``next_bar`` crosses the
        rollover hour, charge daily swap on every open lot.

        Returns total swap applied (signed; negative = cost to trader).
        """
        if not self._open_lots:
            return Decimal("0")

        rollovers_crossed = self._rollovers_between(current_bar.time, next_bar.time)
        if rollovers_crossed == 0:
            return Decimal("0")

        # Wednesday rollover (the rollover whose date is a Wednesday in
        # UTC) carries the weekend multiplier.
        multiplier = self._weekend_multiplier_if_applicable(current_bar.time, next_bar.time)

        total_swap = Decimal("0")
        for lot in self._open_lots:
            per_unit = swap_per_unit_per_day(lot.position.direction, self._cost_model)
            charge = lot.position.size * per_unit * Decimal(rollovers_crossed) * Decimal(multiplier)
            self._cash += charge
            lot.swap_accumulated += charge
            self._trade_swap_acc[lot.trade_id] += charge
            total_swap += charge
        self._cost_breakdown.swap += total_swap
        return total_swap

    def _rollovers_between(self, start: datetime, end: datetime) -> int:
        """Count rollovers (at ``self._rollover_hour`` UTC) strictly in
        the half-open interval ``(start, end]``."""
        if end <= start:
            return 0
        # First candidate rollover after start.
        anchor = start.replace(
            hour=self._rollover_hour,
            minute=0,
            second=0,
            microsecond=0,
        )
        if anchor <= start:
            anchor = anchor + timedelta(days=1)
        count = 0
        while anchor <= end:
            count += 1
            anchor = anchor + timedelta(days=1)
        return count

    def _weekend_multiplier_if_applicable(self, start: datetime, end: datetime) -> int:
        """Return weekend multiplier if a Wednesday rollover falls in
        ``(start, end]``; otherwise 1. (Simple model: any Wed rollover
        in the interval pays the multiplier, even if non-Wed rollovers
        also crossed.)"""
        anchor = start.replace(hour=self._rollover_hour, minute=0, second=0, microsecond=0)
        if anchor <= start:
            anchor = anchor + timedelta(days=1)
        while anchor <= end:
            # Python weekday: Mon=0 .. Sun=6. Wednesday = 2.
            if anchor.weekday() == 2:
                return self._cost_model.weekend_swap_multiplier
            anchor = anchor + timedelta(days=1)
        return 1

    # ----------------------------------------------------- trade lookups

    def lot_for_trade_id(self, trade_id: str) -> _OpenLot | None:
        for lot in self._open_lots:
            if lot.trade_id == trade_id:
                return lot
        return None

    def consume_swap_for_trade(self, trade_id: str) -> Decimal:
        return self._trade_swap_acc.pop(trade_id, Decimal("0"))


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _LiveTrade:
    """In-progress trade record fields; converted to ``TradeRecord``
    once the trade closes (or the run ends with it still open)."""

    trade_id: str
    pair: str
    direction: Direction
    size: Decimal
    entry_time: datetime
    entry_price: Decimal
    entry_commission: Decimal


class BacktestEngine:
    """The one-and-only path that runs a strategy over historical bars.

    Construct, call :meth:`run`, get a :class:`BacktestResult`. No
    other entry point should issue strategy calls — the engine is the
    sole owner of the signal→risk-gate→fill flow.
    """

    def __init__(
        self,
        *,
        bars: list[Candle],
        strategy: Strategy,
        risk_config: RiskConfig,
        cost_model: CostModel,
        starting_balance: Decimal,
        day_rollover_utc_hour: int = 21,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        if len(bars) < 2:
            raise ValueError("backtest needs at least 2 bars")
        # Validate sort + uniform pair.
        pairs = {b.pair for b in bars}
        if len(pairs) != 1:
            raise ValueError(f"all bars must be the same pair, got {pairs}")
        self._pair = next(iter(pairs))
        ordered = sorted(bars, key=lambda b: b.time)
        if ordered != bars:
            # Don't silently re-order — surfacing this catches a class of
            # caller bugs that look like look-ahead.
            raise ValueError("bars must be passed in ascending time order")
        self._bars = list(bars)
        self._strategy = strategy
        self._risk_config = risk_config
        self._cost_model = cost_model
        self._starting_balance = starting_balance
        self._rollover_hour = day_rollover_utc_hour
        self._logger = (logger if logger is not None else _default_logger()).bind(
            component="backtest_engine"
        )

    # ---------------------------------------------------------------- run

    def run(self) -> BacktestResult:
        bars = self._bars
        n = len(bars)
        broker = BacktestBroker(
            starting_balance=self._starting_balance,
            cost_model=self._cost_model,
            day_rollover_utc_hour=self._rollover_hour,
        )
        risk_manager = RiskManager(self._risk_config)

        equity_curve: list[EquityPoint] = []
        live_trades: dict[str, _LiveTrade] = {}
        closed_trades: list[TradeRecord] = []
        n_proposed = 0
        n_accepted = 0
        n_rejected = 0
        # Deterministic trade-id counter. Trade ids must NOT use uuid4 — two
        # seeded runs have to produce byte-identical BacktestResults, trade_log
        # included. A monotonic per-run counter is unique within a run and
        # reproducible across runs. See `_trade_id`.
        trade_seq = 0
        halted = False
        halted_at: int | None = None
        halt_reason: str | None = None

        # Loop to n-1 so we always have a next_bar to fill at.
        for t in range(n - 1):
            current_bar = bars[t]
            next_bar = bars[t + 1]

            # 1. Daily rollover bookkeeping BEFORE marking, so day_start
            #    equity captures end-of-prior-day equity once a new day
            #    starts.
            broker.maybe_roll_day(current_bar)

            # 2. Mark to market at current bar's close.
            broker.mark_to_market(current_bar)
            account = broker.snapshot(as_of=current_bar.time)

            # 3. Defensive: catch kill-switch breaches that wouldn't be
            #    triggered until the next risk evaluate() — most relevantly
            #    drawdown via held-position unrealized PnL.
            if (
                account.drawdown_pct >= self._risk_config.max_drawdown_pct
                and not risk_manager.kill_switch_engaged
            ):
                risk_manager.trip(
                    (
                        f"drawdown {account.drawdown_pct} >= cap "
                        f"{self._risk_config.max_drawdown_pct}"
                    ),
                    tripped_by="drawdown",
                    now=current_bar.time,
                )
            if (
                account.daily_loss_pct >= self._risk_config.daily_loss_limit_pct
                and not risk_manager.kill_switch_engaged
            ):
                risk_manager.trip(
                    (
                        f"daily_loss {account.daily_loss_pct} >= cap "
                        f"{self._risk_config.daily_loss_limit_pct}"
                    ),
                    tripped_by="daily_loss",
                    now=current_bar.time,
                )

            # 4. Record equity sample for this bar.
            equity_curve.append(
                EquityPoint(
                    bar_index=t,
                    time=current_bar.time,
                    balance=broker.cash,
                    equity=account.equity,
                    drawdown_pct=account.drawdown_pct,
                    open_positions=len(broker.open_lots),
                )
            )

            # 5. If kill switch is engaged, flatten and halt.
            if risk_manager.kill_switch_engaged:
                halted = True
                halted_at = t
                halt_reason = (
                    f"kill switch tripped at bar {t} "
                    f"({risk_manager.kill_switch_state.tripped_by})"
                )
                self._flatten_all(
                    broker=broker,
                    next_bar=next_bar,
                    live_trades=live_trades,
                    closed_trades=closed_trades,
                )
                break

            # 6. Strategy decide step. The view is fresh per bar and
            #    physically contains only bars[0..t] — no look-ahead is
            #    possible.
            view = PointInTimeView(bars, end_index=t)
            try:
                signals = self._strategy.decide(view, account, as_of=current_bar.time)
            except Exception as e:
                halted = True
                halted_at = t
                halt_reason = f"strategy raised {type(e).__name__}: {e}"
                self._logger.error(
                    "strategy_raised",
                    bar_index=t,
                    error=repr(e),
                )
                self._flatten_all(
                    broker=broker,
                    next_bar=next_bar,
                    live_trades=live_trades,
                    closed_trades=closed_trades,
                )
                break

            n_proposed += len(signals)

            for sig in signals:
                if sig.pair != self._pair:
                    n_rejected += 1
                    continue
                decision = risk_manager.evaluate(sig, account)
                if not decision.accepted or decision.sized_order is None:
                    n_rejected += 1
                    if risk_manager.kill_switch_engaged:
                        # Drawdown/daily-loss tripped INSIDE evaluate().
                        halted = True
                        halted_at = t
                        halt_reason = f"kill switch tripped during evaluate at bar {t}"
                        break
                    continue

                sized_order = decision.sized_order
                n_accepted += 1

                # 7. Close any opposite-direction lot on the same pair.
                self._close_opposite(
                    broker=broker,
                    sized_order=sized_order,
                    next_bar=next_bar,
                    live_trades=live_trades,
                    closed_trades=closed_trades,
                )

                # 8. If a same-direction lot already exists, skip (no
                #    scale-in for the reference engine; smarter sizing is
                #    a future strategy-level concern).
                if any(
                    lot.position.pair == sized_order.pair
                    and lot.position.direction == sized_order.direction
                    for lot in broker.open_lots
                ):
                    continue

                # 9. Open new at next_bar.open.
                trade_id = _trade_id(trade_seq)
                trade_seq += 1
                lot, fill_price = broker.open_at_next_open(sized_order, next_bar, trade_id=trade_id)
                live_trades[trade_id] = _LiveTrade(
                    trade_id=trade_id,
                    pair=lot.position.pair,
                    direction=lot.position.direction,
                    size=lot.position.size,
                    entry_time=next_bar.time,
                    entry_price=fill_price,
                    entry_commission=lot.entry_commission,
                )

                # Re-snapshot for the next signal in this bar.
                account = broker.snapshot(as_of=current_bar.time)

            if halted:
                self._flatten_all(
                    broker=broker,
                    next_bar=next_bar,
                    live_trades=live_trades,
                    closed_trades=closed_trades,
                )
                break

            # 10. Swap rollover between current_bar and next_bar.
            broker.apply_swap_if_rollover(current_bar, next_bar)

        # End of loop.

        # Final equity sample. Mark to last fully-processed bar.
        last_processed_index = halted_at if halted_at is not None else n - 1
        last_bar = bars[last_processed_index]
        broker.mark_to_market(last_bar)
        final_account = broker.snapshot(as_of=last_bar.time)
        equity_curve.append(
            EquityPoint(
                bar_index=last_processed_index,
                time=last_bar.time,
                balance=broker.cash,
                equity=final_account.equity,
                drawdown_pct=final_account.drawdown_pct,
                open_positions=len(broker.open_lots),
            )
        )

        # Convert remaining open trades to records (with their open exit
        # left as None — they're still open).
        for trade_id, live in live_trades.items():
            swap = broker.consume_swap_for_trade(trade_id)
            closed_trades.append(
                TradeRecord(
                    trade_id=live.trade_id,
                    pair=live.pair,
                    direction=live.direction,
                    size=live.size,
                    entry_time=live.entry_time,
                    entry_price=live.entry_price,
                    entry_commission=live.entry_commission,
                    swap_paid=swap,
                )
            )

        metrics = compute_metrics(equity_curve, closed_trades)

        return BacktestResult(
            config_hash=self._config_hash(),
            pair=self._pair,
            bars_total=n,
            bars_processed=last_processed_index + 1,
            start_time=bars[0].time,
            end_time=last_bar.time,
            starting_balance=self._starting_balance,
            ending_balance=broker.cash,
            ending_equity=final_account.equity,
            peak_equity=final_account.peak_equity,
            max_drawdown_pct=max(
                (p.drawdown_pct for p in equity_curve),
                default=Decimal("0"),
            ),
            n_signals_proposed=n_proposed,
            n_signals_accepted=n_accepted,
            n_signals_rejected=n_rejected,
            halted_due_to_kill_switch=halted,
            halted_at_bar_index=halted_at,
            halt_reason=halt_reason,
            metrics=metrics,
            cost_breakdown=broker.cost_breakdown,
            equity_curve=tuple(equity_curve),
            trade_log=tuple(closed_trades),
        )

    # ------------------------------------------------------------ internals

    def _close_opposite(
        self,
        *,
        broker: BacktestBroker,
        sized_order: OrderRequest,
        next_bar: Candle,
        live_trades: dict[str, _LiveTrade],
        closed_trades: list[TradeRecord],
    ) -> None:
        opposing = [
            lot
            for lot in broker.open_lots
            if lot.position.pair == sized_order.pair
            and lot.position.direction != sized_order.direction
        ]
        for lot in opposing:
            exit_price, exit_commission, pnl = broker.close_at_next_open(lot, next_bar)
            live = live_trades.pop(lot.trade_id, None)
            swap = broker.consume_swap_for_trade(lot.trade_id)
            if live is None:
                continue
            closed_trades.append(
                TradeRecord(
                    trade_id=live.trade_id,
                    pair=live.pair,
                    direction=live.direction,
                    size=live.size,
                    entry_time=live.entry_time,
                    entry_price=live.entry_price,
                    entry_commission=live.entry_commission,
                    exit_time=next_bar.time,
                    exit_price=exit_price,
                    exit_commission=exit_commission,
                    swap_paid=swap,
                    realized_pnl=pnl,
                )
            )

    def _flatten_all(
        self,
        *,
        broker: BacktestBroker,
        next_bar: Candle,
        live_trades: dict[str, _LiveTrade],
        closed_trades: list[TradeRecord],
    ) -> None:
        for lot in list(broker.open_lots):
            exit_price, exit_commission, pnl = broker.close_at_next_open(lot, next_bar)
            live = live_trades.pop(lot.trade_id, None)
            swap = broker.consume_swap_for_trade(lot.trade_id)
            if live is None:
                continue
            closed_trades.append(
                TradeRecord(
                    trade_id=live.trade_id,
                    pair=live.pair,
                    direction=live.direction,
                    size=live.size,
                    entry_time=live.entry_time,
                    entry_price=live.entry_price,
                    entry_commission=live.entry_commission,
                    exit_time=next_bar.time,
                    exit_price=exit_price,
                    exit_commission=exit_commission,
                    swap_paid=swap,
                    realized_pnl=pnl,
                )
            )

    def _config_hash(self) -> str:
        payload: dict[str, Any] = {
            "strategy_name": self._strategy.name,
            "strategy_state_keys": sorted(
                k for k in vars(self._strategy).keys() if not k.startswith("_")
            ),
            "risk_config": self._risk_config.model_dump(mode="json"),
            "cost_model": {
                "half_spread": str(self._cost_model.half_spread),
                "commission_per_unit": str(self._cost_model.commission_per_unit),
                "slippage_per_unit": str(self._cost_model.slippage_per_unit),
                "swap_long_per_unit_per_day": str(self._cost_model.swap_long_per_unit_per_day),
                "swap_short_per_unit_per_day": str(self._cost_model.swap_short_per_unit_per_day),
                "weekend_swap_multiplier": self._cost_model.weekend_swap_multiplier,
            },
            "starting_balance": str(self._starting_balance),
            "rollover_hour": self._rollover_hour,
            "bars": {
                "n": len(self._bars),
                "pair": self._pair,
                "first_time": self._bars[0].time.isoformat(),
                "last_time": self._bars[-1].time.isoformat(),
                "first_close": str(self._bars[0].close),
                "last_close": str(self._bars[-1].close),
            },
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Trade ids
# ---------------------------------------------------------------------------


def _trade_id(seq: int) -> str:
    """Deterministic trade id from a per-run sequence number.

    Format ``t000000`` (zero-padded). Deliberately NOT ``uuid4`` — the engine
    must be reproducible: identical inputs ⇒ identical ``BacktestResult``,
    including every ``trade_id`` in ``trade_log``.
    """
    return f"t{seq:06d}"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _default_logger() -> structlog.stdlib.BoundLogger:
    if not structlog.is_configured():
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    return structlog.get_logger("core.backtest.engine")


# Tag unused import to keep ruff quiet for the timezone helper kept for
# future use.
_ = timezone


__all__ = [
    "BacktestBroker",
    "BacktestEngine",
]
