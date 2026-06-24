"""In-memory paper broker.

Implements :class:`core.broker.Broker` against an in-process state: cash
balance, open positions, realized P&L, peak equity. Fills happen at the current
quote from an injected :class:`core.market_data.PriceProvider`, with a
configurable broker spread markup and per-trade commission.

This is paper trading — no margin enforcement, no swap/rollover (that lives in
the backtester per ``skills/forex-cost-modeling``), no slippage beyond the
configured spread. The point at this stage is to:

1. Lock in the contract that orders only flow through the risk-gated
   :class:`core.execution.ExecutionEngine`.
2. Make sure the account-state arithmetic the risk manager depends on
   (equity, peak_equity, unrealized P&L) is exercised end-to-end.

Costs are deliberately non-zero by default in the test suites: a
perfect-fill paper broker would teach the wrong lessons about strategy edge.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated

import structlog
from pydantic import BaseModel, ConfigDict, Field

from core.market_data import PriceProvider
from core.models import (
    AccountState,
    Direction,
    Fill,
    NonNegativeDecimal,
    OrderRequest,
    Position,
    PositiveDecimal,
)


class PaperBrokerConfig(BaseModel):
    """Cost knobs for the paper broker. Frozen.

    - ``commission_per_unit``: charged on EACH fill (open and close), as
      ``size * commission_per_unit``. Use a small non-zero default in tests.
    - ``extra_half_spread``: added to the ask (buys) and subtracted from the
      bid (sells), modelling broker markup *on top of* whatever spread the
      injected :class:`PriceProvider` already returns.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    starting_balance: PositiveDecimal
    commission_per_unit: NonNegativeDecimal
    extra_half_spread: NonNegativeDecimal = Decimal("0")
    # day_start_equity is captured on construction unless overridden. In a real
    # deployment a daily rollover job would re-stamp it at the trading-day
    # boundary; tests can pass an explicit value to control the daily-loss math.
    day_start_equity: Annotated[Decimal | None, Field(default=None)] = None


class PaperBroker:
    """In-process broker implementing :class:`core.broker.Broker`.

    Mutable internal state lives here; the snapshots returned by
    :meth:`get_account_state` are FROZEN copies that the risk manager and the
    rest of the fast loop can pass around safely.
    """

    def __init__(
        self,
        config: PaperBrokerConfig,
        price_provider: PriceProvider,
        *,
        now_fn: Callable[[], datetime] | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        self._config = config
        self._price_provider = price_provider
        self._cash: Decimal = config.starting_balance
        self._open_positions: list[Position] = []
        self._realized_pnl_today: Decimal = Decimal("0")
        self._peak_equity: Decimal = config.starting_balance
        self._day_start_equity: Decimal = (
            config.day_start_equity
            if config.day_start_equity is not None
            else config.starting_balance
        )
        self._now_fn = now_fn or (lambda: datetime.now(UTC))
        self._logger = (
            logger if logger is not None else _default_logger()
        ).bind(component="paper_broker")

    # ------------------------------------------------------------ inspection

    @property
    def cash(self) -> Decimal:
        return self._cash

    @property
    def peak_equity(self) -> Decimal:
        return self._peak_equity

    # ----------------------------------------------------- Broker interface

    def get_account_state(self) -> AccountState:
        """Snapshot the account. Recomputes equity from current marks and
        bumps peak_equity if we've made a new high."""
        unrealized = self._unrealized_pnl_total()
        equity = self._cash + unrealized
        if equity > self._peak_equity:
            self._peak_equity = equity

        return AccountState(
            balance=self._cash,
            equity=equity,
            open_positions=tuple(self._open_positions),
            realized_pnl_today=self._realized_pnl_today,
            unrealized_pnl=unrealized,
            peak_equity=self._peak_equity,
            day_start_equity=self._day_start_equity,
            as_of=self._now_fn(),
        )

    def get_open_positions(self) -> list[Position]:
        return list(self._open_positions)

    def place_order(self, order: OrderRequest) -> Fill:
        """Open a new position at the current quote.

        Buys fill at ``ask + extra_half_spread``; sells at
        ``bid - extra_half_spread``. Commission = ``size * commission_per_unit``
        is debited from cash. The new position is appended to the open list.
        """
        quote = self._price_provider.get_quote(order.pair)
        self._validate_quote(quote)

        if order.direction is Direction.LONG:
            fill_price = quote.ask + self._config.extra_half_spread
        else:
            fill_price = quote.bid - self._config.extra_half_spread
        if fill_price <= 0:
            raise ValueError(
                f"non-positive effective fill price for {order.pair}: {fill_price}"
            )

        commission = order.size * self._config.commission_per_unit
        if commission > self._cash:
            raise ValueError(
                f"insufficient cash for commission: cash={self._cash} need={commission}"
            )

        self._cash -= commission
        position = Position(
            pair=order.pair,
            direction=order.direction,
            size=order.size,
            entry_price=fill_price,
            stop_price=order.stop_price,
        )
        self._open_positions.append(position)
        self._maybe_bump_peak()

        fill = Fill(
            pair=order.pair,
            direction=order.direction,
            size=order.size,
            fill_price=fill_price,
            commission=commission,
            timestamp=self._now_fn(),
        )
        self._logger.info(
            "paper_fill_open",
            pair=fill.pair,
            direction=fill.direction.value,
            size=str(fill.size),
            fill_price=str(fill.fill_price),
            commission=str(fill.commission),
            cash_after=str(self._cash),
        )
        return fill

    def close_position(self, position: Position) -> Fill:
        """Close ``position`` at the current quote; realize P&L into cash.

        Raises:
            ValueError: if ``position`` is not in the open list.
        """
        try:
            idx = self._open_positions.index(position)
        except ValueError as e:
            raise ValueError(f"position not open: {position!r}") from e

        quote = self._price_provider.get_quote(position.pair)
        self._validate_quote(quote)

        # The CLOSE side is opposite to the position side.
        if position.direction is Direction.LONG:
            fill_price = quote.bid - self._config.extra_half_spread
            close_direction = Direction.SHORT
            pnl = (fill_price - position.entry_price) * position.size
        else:
            fill_price = quote.ask + self._config.extra_half_spread
            close_direction = Direction.LONG
            pnl = (position.entry_price - fill_price) * position.size

        if fill_price <= 0:
            raise ValueError(
                f"non-positive effective close price for {position.pair}: {fill_price}"
            )

        commission = position.size * self._config.commission_per_unit
        # P&L gets booked even if it leaves cash negative — the broker reports
        # what happened; risk checks live one layer up.
        self._cash += pnl - commission
        self._realized_pnl_today += pnl - commission
        del self._open_positions[idx]
        self._maybe_bump_peak()

        fill = Fill(
            pair=position.pair,
            direction=close_direction,
            size=position.size,
            fill_price=fill_price,
            commission=commission,
            timestamp=self._now_fn(),
        )
        self._logger.info(
            "paper_fill_close",
            pair=fill.pair,
            direction=fill.direction.value,
            size=str(fill.size),
            fill_price=str(fill.fill_price),
            commission=str(fill.commission),
            realized_pnl=str(pnl - commission),
            cash_after=str(self._cash),
        )
        return fill

    # ----------------------------------------------------------- internals

    def _unrealized_pnl_total(self) -> Decimal:
        return sum(
            (self._mark_position(p) for p in self._open_positions),
            start=Decimal("0"),
        )

    def _mark_position(self, p: Position) -> Decimal:
        """Mark-to-market against the *close-side* of the spread — the
        conservative number, equal to what we'd realize if we closed now."""
        quote = self._price_provider.get_quote(p.pair)
        if p.direction is Direction.LONG:
            return (quote.bid - p.entry_price) * p.size
        return (p.entry_price - quote.ask) * p.size

    def _maybe_bump_peak(self) -> None:
        equity = self._cash + self._unrealized_pnl_total()
        if equity > self._peak_equity:
            self._peak_equity = equity

    @staticmethod
    def _validate_quote(quote: object) -> None:
        # Quotes coming from Quote(...) are already validated (positive prices,
        # ask >= bid). This guard catches a misbehaving provider that returns
        # something stale-but-corrupted at the boundary.
        from core.models import Quote

        if not isinstance(quote, Quote):  # pragma: no cover - defensive
            raise TypeError(f"price provider returned non-Quote: {type(quote)!r}")
        if quote.bid <= 0 or quote.ask <= 0:
            raise ValueError(f"non-positive quote: {quote}")


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
    return structlog.get_logger("core.paper_broker")
