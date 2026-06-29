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

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol

from core.models import (
    AccountState,
    Candle,
    Direction,
    OrderRequest,
    ParamRange,
    StrategyArchetype,
    StrategyCandidate,
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

    def __iter__(self) -> Iterator[Candle]:
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

    def params(self) -> dict[str, Any]:
        """The behaviour-defining parameters that constitute this strategy's
        identity — JSON-serialisable, ordered, and **deterministic**.

        This is what the backtester hashes into ``config_hash`` and what the
        optimization agent / champion-challenger registry treat as a
        strategy's identity. Two instances with the same ``params()`` are the
        same strategy; two with different ``params()`` are not.

        It MUST contain only the parameters that change behaviour (periods,
        sizing, pair, …) and MUST EXCLUDE mutable runtime state (indicator
        accumulators, the previous-cross marker, etc.) — otherwise identity
        would drift mid-run and two equivalent strategies would hash apart.
        Decimal values are emitted as strings so the dict is JSON-safe.
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

    def params(self) -> dict[str, Any]:
        """Identity = the constructor parameters only.

        Deliberately excludes ``_prev_diff`` (the cross-detection accumulator),
        which is transient runtime state, not part of what the strategy *is*.
        Decimals are stringified so the dict is JSON-serialisable.
        """
        return {
            "pair": self._pair,
            "fast_period": self._fast,
            "slow_period": self._slow,
            "size": str(self._size),
            "stop_distance": str(self._stop_distance),
        }

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


# ---------------------------------------------------------------------------
# Strategy archetype registry — the FIXED catalog the strategy_lab_agent may
# propose from. Each archetype maps to a concrete, constructible Strategy.
# ---------------------------------------------------------------------------


def pip_size(pair: str) -> Decimal:
    """Price increment of one pip for ``pair``.

    JPY-quoted pairs (USDJPY, EURJPY, …) use ``0.01``; everything else uses
    ``0.0001``. The quote currency is the last three letters. This is the
    convention that makes ``stop_distance`` limits **price-relative** instead of
    absolute: a 50-pip stop is 0.005 on EURUSD but 0.5 on USDJPY, yet both are
    "50 pips". Exotic / metals conventions are out of scope for now.
    """
    return Decimal("0.01") if pair.upper().endswith("JPY") else Decimal("0.0001")


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """Hard limits for one archetype parameter — the outer fence the agent's
    proposed ``parameter_ranges`` must sit inside.

    ``unit`` is documentation only (rendered into the agent's catalog so it
    proposes in the right units). Limits for a pip-denominated parameter (e.g.
    ``stop_distance_pips``) are themselves in PIPS, so they are pair-independent
    — the per-pair conversion to price units happens in the archetype builder.
    """

    name: str
    minimum: Decimal
    maximum: Decimal
    integer: bool
    unit: str = "unitless"


@dataclass(frozen=True, slots=True)
class ArchetypeSpec:
    """One entry in the registry: how to validate + build an archetype.

    ``builder`` takes ``(instrument, params)`` and returns a real
    :class:`Strategy`. It is the SOLE bridge from a proposed
    :class:`core.models.StrategyCandidate` to something the engine can run.
    """

    archetype: StrategyArchetype
    description: str
    param_specs: tuple[ParamSpec, ...]
    builder: Callable[[str, dict[str, Decimal]], Strategy]

    def param_names(self) -> frozenset[str]:
        return frozenset(ps.name for ps in self.param_specs)


def _build_ma_crossover(instrument: str, params: dict[str, Decimal]) -> Strategy:
    # `stop_distance_pips` is in PIPS; convert to price units per the pair's pip
    # size before constructing the (price-unit) strategy. This is the single
    # place the pip→price conversion lives.
    stop_price_units = params["stop_distance_pips"] * pip_size(instrument)
    return MovingAverageCrossover(
        pair=instrument,
        fast_period=int(params["fast_period"]),
        slow_period=int(params["slow_period"]),
        size=params["size"],
        stop_distance=stop_price_units,
    )


STRATEGY_REGISTRY: dict[StrategyArchetype, ArchetypeSpec] = {
    StrategyArchetype.MA_CROSSOVER: ArchetypeSpec(
        archetype=StrategyArchetype.MA_CROSSOVER,
        description=(
            "Moving-average crossover. Long when the fast MA crosses above the "
            "slow MA, short on the reverse cross. Single pair, fixed size, "
            "stop in PIPS. Requires fast_period < slow_period."
        ),
        param_specs=(
            ParamSpec("fast_period", Decimal("2"), Decimal("200"), integer=True, unit="bars"),
            ParamSpec("slow_period", Decimal("3"), Decimal("400"), integer=True, unit="bars"),
            # size is a COUNT of base-currency units — pair-independent, not a
            # price — so its limits don't suffer the absolute-vs-relative trap.
            ParamSpec(
                "size",
                Decimal("1"),
                Decimal("10000000"),
                integer=False,
                unit="base-currency units",
            ),
            # Pip-denominated → pair-independent limits. 5..500 pips covers every
            # realistic stop (a 30-pip EURUSD scalp through a 300-pip swing) on
            # both 0.0001-pip and 0.01-pip (JPY) pairs.
            ParamSpec(
                "stop_distance_pips",
                Decimal("5"),
                Decimal("500"),
                integer=False,
                unit="pips",
            ),
        ),
        builder=_build_ma_crossover,
    ),
}


def archetype_catalog() -> str:
    """Deterministic human-readable catalog of the registry, for the agent's
    stable system prefix. Byte-stable across processes (sorted), so it doesn't
    break OpenAI prompt caching.

    Each parameter renders with its UNIT so the proposer reasons in the right
    units. In particular ``stop_distance_pips`` is in PIPS, NOT price units —
    the limits are pair-independent and the engine converts pips to price using
    1 pip = 0.0001 (or 0.01 for JPY-quoted pairs)."""
    lines: list[str] = [
        "Units note: parameter limits are stated in each parameter's own unit. "
        "stop_distance_pips is in PIPS (1 pip = 0.0001, or 0.01 for JPY-quoted "
        "pairs like USDJPY), so the SAME pip range fits every pair; do not "
        "propose stops in raw price units. size is a count of base-currency "
        "units; periods are in bars.",
        "",
    ]
    for archetype in sorted(STRATEGY_REGISTRY, key=lambda a: a.value):
        spec = STRATEGY_REGISTRY[archetype]
        lines.append(f"- {archetype.value}: {spec.description}")
        for ps in spec.param_specs:
            kind = "integer" if ps.integer else "decimal"
            lines.append(f"    * {ps.name} ({kind}, {ps.unit}) in [{ps.minimum}, {ps.maximum}]")
    return "\n".join(lines)


def validate_candidate(candidate: StrategyCandidate) -> tuple[str, ...]:
    """Deterministic structural validation of a proposed candidate.

    Returns a tuple of error strings (empty == valid). Checks archetype
    membership, that the parameter / range key sets exactly match the
    archetype, that each declared range sits inside the archetype's hard
    limits with ``low < high``, integer-ness, and that each concrete value
    sits inside both the archetype limits and its own declared range. Does
    NOT construct — see :func:`build_strategy` for the constructibility check.
    """
    spec = STRATEGY_REGISTRY.get(candidate.archetype)
    if spec is None:
        return (f"unknown archetype {candidate.archetype!r}",)

    errors: list[str] = []
    params = candidate.params_as_dict()
    ranges: dict[str, ParamRange] = candidate.ranges_as_dict()
    expected = spec.param_names()
    if frozenset(params) != expected:
        errors.append(f"parameters {sorted(params)} != required {sorted(expected)}")
    if frozenset(ranges) != expected:
        errors.append(f"parameter_ranges {sorted(ranges)} != required {sorted(expected)}")

    for ps in spec.param_specs:
        rng = ranges.get(ps.name)
        if rng is not None:
            if rng.low >= rng.high:
                errors.append(f"{ps.name} range low {rng.low} >= high {rng.high}")
            if rng.low < ps.minimum or rng.high > ps.maximum:
                errors.append(
                    f"{ps.name} range [{rng.low}, {rng.high}] outside archetype "
                    f"limits [{ps.minimum}, {ps.maximum}]"
                )
            if ps.integer and (
                rng.low != rng.low.to_integral_value() or rng.high != rng.high.to_integral_value()
            ):
                errors.append(f"{ps.name} is integer; range bounds must be whole")
        val = params.get(ps.name)
        if val is not None:
            if ps.integer and val != val.to_integral_value():
                errors.append(f"{ps.name} must be an integer, got {val}")
            if val < ps.minimum or val > ps.maximum:
                errors.append(
                    f"{ps.name} value {val} outside archetype limits "
                    f"[{ps.minimum}, {ps.maximum}]"
                )
            if rng is not None and not (rng.low <= val <= rng.high):
                errors.append(
                    f"{ps.name} value {val} outside its declared range " f"[{rng.low}, {rng.high}]"
                )
    return tuple(errors)


def build_strategy(candidate: StrategyCandidate) -> Strategy:
    """Construct the real :class:`Strategy` for a candidate.

    Raises ``ValueError`` / ``KeyError`` if the archetype is unknown, a
    parameter is missing, or the archetype's own constructor rejects the
    values (e.g. ``fast_period >= slow_period``). Callers treat any exception
    as "not constructible" — that's the hard gate before a candidate ships.
    """
    spec = STRATEGY_REGISTRY.get(candidate.archetype)
    if spec is None:
        raise ValueError(f"unknown archetype {candidate.archetype!r}")
    return spec.builder(candidate.instrument, candidate.params_as_dict())


__all__ = [
    "STRATEGY_REGISTRY",
    "ArchetypeSpec",
    "LookAheadError",
    "MovingAverageCrossover",
    "ParamSpec",
    "PointInTimeView",
    "Strategy",
    "archetype_catalog",
    "build_strategy",
    "pip_size",
    "validate_candidate",
]
