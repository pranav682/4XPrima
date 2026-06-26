"""Strategy Protocol + point-in-time view + a reference strategy.

This module hosts the contract every backtested or live-runnable strategy
must satisfy, plus the structural defence against the engine's first
cardinal sin: **look-ahead bias**.

Look-ahead defence (structural, not advisory):
---------------------------------------------
At decision time ``t`` the strategy receives a :class:`PointInTimeView` that
**physically stores only bars 0..t** (the visible slice). Future bars are
not in the object — they cannot be peeked at by reaching into a private
attribute or by sneaky indexing, because the data simply is not there.

Any out-of-range request (positive or negative index past the visible
window) raises :class:`LookAheadError`. The engine constructs the view
fresh per bar; the strategy is structurally incapable of seeing the next
bar's open / high / low / close until the engine advances.

Signal timing convention:
------------------------
- The strategy decides at end-of-bar ``t``, using data through bar ``t``
  inclusive.
- Orders the strategy returns are filled by the engine at bar ``t+1``'s
  open, with realistic cost (spread + commission + slippage).
- This is the standard "next-bar-open" convention. The engine never lets
  a strategy fill at the same bar's close.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from core.models import (
    AccountState,
    Candle,
    Direction,
    OrderRequest,
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LookAheadError(IndexError):
    """Raised when a strategy attempts to access bar data it would not
    have had in real time. Subclass of :class:`IndexError` so existing
    try/except IndexError branches catch it naturally."""


# ---------------------------------------------------------------------------
# PointInTimeView — bounded view onto a candle history
# ---------------------------------------------------------------------------


class PointInTimeView:
    """A read-only window onto the bars visible at decision time.

    The view stores ONLY the visible slice ``bars[0..end_index]``. Future
    bars are not part of this object — they were never copied in. There is
    no ``._all_bars`` private attribute to reach into.

    Negative indexing works against the visible window (``view[-1]`` =
    latest visible bar). Out-of-range access (``view[len(view)]`` or
    further) raises :class:`LookAheadError`.

    The view is **immutable** (``__slots__`` + tuple storage); a strategy
    cannot grow it to see more.
    """

    __slots__ = ("_visible",)

    def __init__(self, bars: Sequence[Candle], end_index: int) -> None:
        if end_index < 0:
            raise ValueError("end_index must be ≥ 0")
        if end_index >= len(bars):
            raise ValueError(
                f"end_index {end_index} ≥ len(bars) {len(bars)} — "
                "would expose data that doesn't exist"
            )
        # Defensive tuple copy — strategy cannot mutate to see further.
        self._visible: tuple[Candle, ...] = tuple(bars[: end_index + 1])

    # ----------------------------------------------------- size / iter

    def __len__(self) -> int:
        return len(self._visible)

    def __iter__(self):
        return iter(self._visible)

    # ----------------------------------------------------------- access

    def __getitem__(self, i: int | slice) -> Candle | tuple[Candle, ...]:
        if isinstance(i, slice):
            # Slicing returns at most the visible window.
            return self._visible[i]
        n = len(self._visible)
        actual = i + n if i < 0 else i
        if actual < 0 or actual >= n:
            raise LookAheadError(f"index {i} out of point-in-time bounds [0, {n - 1}]")
        return self._visible[actual]

    @property
    def latest(self) -> Candle:
        """The most recent visible bar (the current decision bar)."""
        return self._visible[-1]

    def lookback(self, n: int) -> tuple[Candle, ...]:
        """Return the most recent ``n`` visible bars, oldest first.

        Caps at the visible window size; never silently extends past it.
        """
        if n <= 0:
            return ()
        return self._visible[-n:]

    def closes(self, n: int | None = None) -> tuple[Decimal, ...]:
        """Close prices for the visible window (or its last ``n`` bars)."""
        window = self._visible if n is None else self.lookback(n)
        return tuple(b.close for b in window)


# ---------------------------------------------------------------------------
# Strategy Protocol
# ---------------------------------------------------------------------------


class Strategy(Protocol):
    """The interface every strategy implements.

    Strategies are **stateful** (an indicator accumulates state across
    bars). A fresh instance must be used for each backtest run; the engine
    asserts nothing about this — discipline is on the caller.

    The strategy does NOT call the risk manager or broker. It only emits
    proposed ``OrderRequest`` objects; the engine evaluates and fills.
    """

    name: str

    def decide(
        self,
        bars: PointInTimeView,
        account: AccountState,
        *,
        as_of: datetime,
    ) -> list[OrderRequest]:
        """Emit zero or more orders to be filled at the NEXT bar's open.

        Args:
            bars: bounded view; only the visible window can be accessed.
            account: the broker's account snapshot at the current bar's
                close. Read-only.
            as_of: the current bar's timestamp (UTC).

        Returns:
            A (possibly empty) list of :class:`OrderRequest` objects.
            Engine policy:
              - Same-direction existing position on the same pair → ignored.
              - Opposite-direction existing position → closed then new opened.
        """
        ...


# ---------------------------------------------------------------------------
# REFERENCE strategy — exercises the engine, NOT a researched edge
# ---------------------------------------------------------------------------


class MovingAverageCrossover:
    """REFERENCE strategy used only to exercise the backtest engine.

    Long signal on the fast MA crossing **above** the slow MA; short signal
    on the reverse cross. Fixed size, fixed-distance stop, single pair.

    **This is not a researched edge.** It exists so the engine has a real
    strategy to drive in tests and the dev CLI. Do not use it as a
    starting point for live trading; the strategy lab (Stage 3 part 2)
    will produce candidate strategies the proper way.
    """

    name: str = "ma_crossover_reference"

    def __init__(
        self,
        *,
        pair: str,
        fast_period: int,
        slow_period: int,
        size: Decimal,
        stop_distance: Decimal,
    ) -> None:
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be < slow_period")
        if size <= 0 or stop_distance <= 0:
            raise ValueError("size and stop_distance must be positive")
        self._pair = pair.upper()
        self._fast = fast_period
        self._slow = slow_period
        self._size = size
        self._stop_distance = stop_distance
        # Cross-detection state: the previous (fast - slow) value.
        self._prev_diff: Decimal | None = None

    @property
    def pair(self) -> str:
        return self._pair

    def decide(
        self,
        bars: PointInTimeView,
        account: AccountState,
        *,
        as_of: datetime,
    ) -> list[OrderRequest]:
        if len(bars) < self._slow:
            return []
        closes = bars.closes(self._slow)
        fast_ma = sum(closes[-self._fast :], Decimal("0")) / Decimal(self._fast)
        slow_ma = sum(closes, Decimal("0")) / Decimal(self._slow)
        diff = fast_ma - slow_ma

        signals: list[OrderRequest] = []
        if self._prev_diff is not None:
            current_close = bars.latest.close
            if self._prev_diff <= 0 and diff > 0:
                # Cross up → long signal.
                signals.append(
                    OrderRequest(
                        pair=self._pair,
                        direction=Direction.LONG,
                        size=self._size,
                        entry_price=current_close,
                        stop_price=current_close - self._stop_distance,
                    )
                )
            elif self._prev_diff >= 0 and diff < 0:
                # Cross down → short signal.
                signals.append(
                    OrderRequest(
                        pair=self._pair,
                        direction=Direction.SHORT,
                        size=self._size,
                        entry_price=current_close,
                        stop_price=current_close + self._stop_distance,
                    )
                )
        self._prev_diff = diff
        return signals


__all__ = [
    "LookAheadError",
    "MovingAverageCrossover",
    "PointInTimeView",
    "Strategy",
]
