"""Frozen domain models shared by the fast loop.

Conventions:

- All monetary values are :class:`decimal.Decimal`. We never round-trip account
  balances or P&L through float; the fast loop's correctness lives or dies here.
- All datetimes are timezone-aware UTC. Naive datetimes are rejected.
- Models are pydantic v2 ``BaseModel`` with ``frozen=True`` — direct attribute
  mutation raises ``pydantic.ValidationError``. ``RiskConfig`` in particular is
  required by the architecture invariants to be immutable at runtime; the slow
  loop can only *propose* a new config for human approval, never mutate the
  live instance.
- Sizing convention (paper-trading stage): ``size`` is in units of the *base*
  currency; ``entry_price`` / ``stop_price`` are quote-per-base. Risk-at-stop
  in quote currency is ``size * abs(entry_price - stop_price)``. Notional
  exposure in quote currency is ``size * entry_price``. We assume the account
  is denominated in the quote currency for now — multi-currency P&L
  conversion is out of scope for Stage 1.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Direction(StrEnum):
    """Long or short. Forex is symmetric; we don't carry a `flat`."""

    LONG = "long"
    SHORT = "short"


class DecisionKind(StrEnum):
    """Outcome of one ``RiskManager.evaluate`` call."""

    APPROVE = "approve"
    RESIZE = "resize"   # approved, but size was clipped down by a cap
    REJECT = "reject"


class RejectionReason(StrEnum):
    """Why a ``RiskDecision`` was rejected (or resized).

    The set is closed; ``RiskManager`` MUST NOT invent new reasons at runtime.
    Adding a value is a deliberate change reviewed against the risk spec.
    """

    KILL_SWITCH = "kill_switch"
    PER_TRADE_CAP = "per_trade_cap"
    MAX_CONCURRENT_POSITIONS = "max_concurrent_positions"
    PORTFOLIO_RISK_CAP = "portfolio_risk_cap"
    PER_PAIR_EXPOSURE_CAP = "per_pair_exposure_cap"
    CORRELATED_EXPOSURE_CAP = "correlated_exposure_cap"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_CAP = "drawdown_cap"
    INVALID_INPUT = "invalid_input"
    STOP_DISTANCE_NONPOSITIVE = "stop_distance_nonpositive"
    NONPOSITIVE_EQUITY = "nonpositive_equity"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_utc(value: datetime) -> datetime:
    """Reject naive datetimes outright — UTC is the only acceptable timezone."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware (UTC)")
    if value.utcoffset() != UTC.utcoffset(None):  # exact UTC, not just any tz
        # Accept any tz that resolves to a zero offset — keep it permissive.
        if value.utcoffset().total_seconds() != 0:  # type: ignore[union-attr]
            raise ValueError("datetime must be UTC")
    return value


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


PositiveDecimal = Annotated[Decimal, Field(gt=Decimal("0"))]
NonNegativeDecimal = Annotated[Decimal, Field(ge=Decimal("0"))]
LossPercentDecimal = Annotated[Decimal, Field(gt=Decimal("0"), le=Decimal("1"))]
"""A loss-side fractional percent in (0, 1] — used for caps that measure a
loss against equity (per-trade risk, portfolio risk, daily loss, drawdown).
Exposure caps use ``PositiveDecimal`` since leveraged notional can legitimately
exceed 100% of equity."""


class Position(BaseModel):
    """An open position. Immutable; lives only as a snapshot inside AccountState."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair: str
    direction: Direction
    size: PositiveDecimal
    entry_price: PositiveDecimal
    stop_price: PositiveDecimal

    @field_validator("pair")
    @classmethod
    def _normalize_pair(cls, v: str) -> str:
        if not v or not v.isalpha():
            raise ValueError("pair must be alphabetic, e.g. 'EURUSD'")
        return v.upper()

    @property
    def stop_distance(self) -> Decimal:
        """Absolute price distance to the stop, in quote currency per unit base."""
        return abs(self.entry_price - self.stop_price)

    @property
    def risk_at_stop(self) -> Decimal:
        """Loss in quote currency if the stop fills exactly at ``stop_price``."""
        return self.size * self.stop_distance

    @property
    def notional(self) -> Decimal:
        """Notional exposure in quote currency at entry."""
        return self.size * self.entry_price


class OrderRequest(BaseModel):
    """A request submitted to :meth:`RiskManager.evaluate`.

    ``size`` is the requested size; the risk manager may resize it down (and
    return ``DecisionKind.RESIZE``) but never resize it up.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair: str
    direction: Direction
    size: PositiveDecimal
    entry_price: PositiveDecimal
    stop_price: PositiveDecimal

    @field_validator("pair")
    @classmethod
    def _normalize_pair(cls, v: str) -> str:
        if not v or not v.isalpha():
            raise ValueError("pair must be alphabetic, e.g. 'EURUSD'")
        return v.upper()

    @property
    def stop_distance(self) -> Decimal:
        return abs(self.entry_price - self.stop_price)

    @property
    def risk_at_stop(self) -> Decimal:
        return self.size * self.stop_distance

    @property
    def notional(self) -> Decimal:
        return self.size * self.entry_price

    def with_size(self, new_size: Decimal) -> OrderRequest:
        """Return a copy with ``size`` replaced (used for resize decisions).

        Frozen models support ``model_copy(update=...)`` — this is the safe
        idiom for producing a downsized order without mutating the original.
        """
        return self.model_copy(update={"size": new_size})


class AccountState(BaseModel):
    """Snapshot of the account at decision time. Immutable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    balance: Decimal                 # cash balance, may be negative if margin-called
    equity: Decimal                  # balance + unrealized_pnl; the working number
    open_positions: tuple[Position, ...] = ()
    realized_pnl_today: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    peak_equity: PositiveDecimal      # rolling all-time peak, for drawdown
    day_start_equity: PositiveDecimal  # for the daily loss limit
    as_of: datetime

    @field_validator("as_of")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return _require_utc(v)

    # No invariant tying equity to balance + unrealized_pnl is enforced — the
    # broker is the source of truth and tiny rounding can desync them.

    @property
    def drawdown_pct(self) -> Decimal:
        """Current drawdown from peak as a fraction in [0, 1].

        Returns 0 when at or above peak. Saturates at 1 if equity is non-positive,
        so callers comparing against a fractional cap behave correctly even at
        the edge.
        """
        if self.peak_equity <= 0:
            return Decimal("0")
        if self.equity <= 0:
            return Decimal("1")
        if self.equity >= self.peak_equity:
            return Decimal("0")
        return (self.peak_equity - self.equity) / self.peak_equity

    @property
    def daily_loss_pct(self) -> Decimal:
        """Daily loss as a fraction of ``day_start_equity`` in [0, 1].

        Same edge semantics as :meth:`drawdown_pct`.
        """
        if self.day_start_equity <= 0:
            return Decimal("0")
        if self.equity >= self.day_start_equity:
            return Decimal("0")
        if self.equity <= 0:
            return Decimal("1")
        return (self.day_start_equity - self.equity) / self.day_start_equity


class Granularity(StrEnum):
    """Bar granularities accepted by OANDA's candles endpoint.

    Names mirror OANDA's `CandlestickGranularity` enum verbatim — keeping
    them identical removes a translation step when we send the query string.
    """

    S5 = "S5"
    S10 = "S10"
    S15 = "S15"
    S30 = "S30"
    M1 = "M1"
    M2 = "M2"
    M4 = "M4"
    M5 = "M5"
    M10 = "M10"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H2 = "H2"
    H3 = "H3"
    H4 = "H4"
    H6 = "H6"
    H8 = "H8"
    H12 = "H12"
    D = "D"
    W = "W"
    M = "M"


class Candle(BaseModel):
    """One OHLCV bar at a specific granularity. Frozen.

    ``complete=False`` for the most recent bar when it's still forming. We
    keep both complete and incomplete in the same model so downstream
    consumers can decide whether to drop the forming bar themselves.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair: str
    granularity: Granularity
    time: datetime
    open: PositiveDecimal
    high: PositiveDecimal
    low: PositiveDecimal
    close: PositiveDecimal
    volume: Annotated[int, Field(ge=0)]
    complete: bool

    @field_validator("pair")
    @classmethod
    def _normalize_pair(cls, v: str) -> str:
        if not v or not v.isalpha():
            raise ValueError("pair must be alphabetic, e.g. 'EURUSD'")
        return v.upper()

    @field_validator("time")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return _require_utc(v)

    def model_post_init(self, __context: object) -> None:
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise ValueError(
                f"OHLC inconsistent: open={self.open} high={self.high} "
                f"low={self.low} close={self.close}"
            )


class Quote(BaseModel):
    """A bid/ask snapshot for a pair at a moment in time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair: str
    bid: PositiveDecimal
    ask: PositiveDecimal
    timestamp: datetime

    @field_validator("pair")
    @classmethod
    def _normalize_pair(cls, v: str) -> str:
        if not v or not v.isalpha():
            raise ValueError("pair must be alphabetic, e.g. 'EURUSD'")
        return v.upper()

    @field_validator("timestamp")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return _require_utc(v)

    def model_post_init(self, __context: object) -> None:
        # pydantic v2 hook — runs after all field validators. We can't enforce
        # ask >= bid via Field alone because the constraint crosses two fields.
        if self.ask < self.bid:
            raise ValueError(f"inverted quote: ask {self.ask} < bid {self.bid}")

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid


class Fill(BaseModel):
    """One executed trade. ``direction`` is the side of the *fill itself*:
    LONG = buy (opens a long, or closes a short); SHORT = sell (opens a short,
    or closes a long)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair: str
    direction: Direction
    size: PositiveDecimal
    fill_price: PositiveDecimal
    commission: NonNegativeDecimal
    timestamp: datetime

    @field_validator("pair")
    @classmethod
    def _normalize_pair(cls, v: str) -> str:
        if not v or not v.isalpha():
            raise ValueError("pair must be alphabetic, e.g. 'EURUSD'")
        return v.upper()

    @field_validator("timestamp")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return _require_utc(v)

    @property
    def notional(self) -> Decimal:
        return self.size * self.fill_price


class RiskConfig(BaseModel):
    """All risk caps. Frozen at runtime.

    The slow loop cannot mutate a live instance; a config change is a *new*
    object written by a human-approved deploy step and re-loaded by the fast
    loop on its own schedule. See ``CLAUDE.md`` invariant #2.
    """

    # frozen=True → ``ValidationError`` on any ``__setattr__``.
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "1.0"
    account_currency: str = "USD"

    # Per-trade and aggregate risk (loss-side; bounded ≤ 100% of equity)
    max_risk_per_trade_pct: LossPercentDecimal
    max_portfolio_risk_pct: LossPercentDecimal

    # Concurrency
    max_concurrent_positions: Annotated[int, Field(ge=1)]

    # Exposure (notional) caps — leveraged forex can exceed 100% of equity,
    # so these are simply positive Decimals.
    max_exposure_per_pair_pct: PositiveDecimal
    max_correlated_exposure_pct: PositiveDecimal

    # Correlation grouping. A pair may appear in multiple groups (e.g. EURUSD
    # belongs to a "USD_QUOTE" group AND a "EUR_BASE" group). Each group caps
    # the sum of notional exposure for pairs inside it.
    correlation_groups: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    # Loss limits (loss-side; bounded ≤ 100% of equity)
    daily_loss_limit_pct: LossPercentDecimal
    max_drawdown_pct: LossPercentDecimal

    @field_validator("correlation_groups")
    @classmethod
    def _normalize_groups(
        cls, v: dict[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        # Normalise pair codes to upper-case for predictable membership tests.
        return {gname: tuple(p.upper() for p in pairs) for gname, pairs in v.items()}

    def groups_containing(self, pair: str) -> tuple[str, ...]:
        """Names of correlation groups that include ``pair``."""
        pair = pair.upper()
        return tuple(g for g, members in self.correlation_groups.items() if pair in members)


class RiskDecision(BaseModel):
    """The output of :meth:`RiskManager.evaluate`.

    On ``DecisionKind.APPROVE`` or ``DecisionKind.RESIZE``, ``sized_order`` is
    set. On ``REJECT``, ``sized_order`` is ``None``.
    ``limiting_rule`` is the first rejection reason that caused the decision,
    for easy log filtering; ``rejected_by`` is the full list (a single order
    can violate several caps at once).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: str
    kind: DecisionKind
    sized_order: OrderRequest | None
    rejected_by: tuple[RejectionReason, ...] = ()
    reason: str                              # human-readable
    limiting_rule: RejectionReason | None = None
    config_hash: str
    as_of: datetime

    @field_validator("as_of")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return _require_utc(v)

    @property
    def accepted(self) -> bool:
        """True iff the decision admits a (possibly resized) order."""
        return self.kind in (DecisionKind.APPROVE, DecisionKind.RESIZE)
