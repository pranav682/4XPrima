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

import re
from datetime import UTC, datetime
from datetime import date as _date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

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
    RESIZE = "resize"  # approved, but size was clipped down by a cap
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
NonNegativeInt = Annotated[int, Field(ge=0)]
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

    balance: Decimal  # cash balance, may be negative if margin-called
    equity: Decimal  # balance + unrealized_pnl; the working number
    open_positions: tuple[Position, ...] = ()
    realized_pnl_today: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    peak_equity: PositiveDecimal  # rolling all-time peak, for drawdown
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


class VolState(StrEnum):
    """Volatility regime for a pair."""

    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    UNKNOWN = "unknown"


class RiskState(StrEnum):
    """Cross-market risk appetite tag."""

    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class TrendState(StrEnum):
    """Directional / range tag for a pair."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    RANGE_TIGHT = "range_tight"
    RANGE_WIDE = "range_wide"
    EVENT_DRIVEN = "event_driven"
    UNKNOWN = "unknown"


class SentimentLabel(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class FlagSeverity(StrEnum):
    INFO = "info"
    WARN = "warn"
    ALERT = "alert"


class ImpactLevel(StrEnum):
    """Impact tag attached to a scheduled economic release.

    `UNKNOWN` is a safe fallback for sources that occasionally emit values
    outside the documented set — we never silently drop a release just because
    its impact label is unfamiliar.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    HOLIDAY = "holiday"
    UNKNOWN = "unknown"


class EconomicEvent(BaseModel):
    """One scheduled (or recently released) economic data point.

    ``actual`` / ``forecast`` / ``previous`` are the parsed numeric values
    (with suffix handling for K/M/B/T, %, $, commas). The raw source strings
    are preserved alongside so we never lose the original. ``surprise`` is a
    computed field — ``actual - forecast`` when both are numeric, ``None``
    otherwise — so downstream consumers don't reinvent the calculation per
    agent / report.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    when: datetime  # UTC, the release time
    currency: str  # ISO-4217: "USD", "EUR", "GBP", ...
    name: str
    impact: ImpactLevel

    raw_actual: str | None = None
    raw_forecast: str | None = None
    raw_previous: str | None = None

    actual: Decimal | None = None
    forecast: Decimal | None = None
    previous: Decimal | None = None

    @field_validator("when")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return _require_utc(v)

    @field_validator("currency")
    @classmethod
    def _validate_ccy(cls, v: str) -> str:
        v = v.upper()
        if not v.isalpha() or len(v) != 3:
            raise ValueError(f"currency must be ISO-4217 (3 letters), got {v!r}")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def surprise(self) -> Decimal | None:
        """``actual - forecast`` when both are numeric; ``None`` otherwise.

        Computed deterministically so every agent / report sees the same
        number. If either input is missing or unparseable (e.g. "Tentative",
        empty string), the answer is ``None``, never a guess.
        """
        if self.actual is None or self.forecast is None:
            return None
        return self.actual - self.forecast


class MacroSeriesPoint(BaseModel):
    """One observation in a FRED-style macro time series.

    ``value`` is ``None`` for explicitly-missing observations (FRED encodes
    these as ``"."``). Callers decide whether to skip or interpolate; we
    refuse to silently coerce missing data to 0.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: str
    date: _date
    value: Decimal | None = None


class NewsEvent(BaseModel):
    """One news article reference, deduplicated by URL upstream.

    ``tone``, ``themes``, and ``entities`` are optional — many sources
    (GDELT's basic article list, for instance) don't populate them per
    article. Treat empty tuples and ``None`` as "not provided by this
    source", not as "definitely zero / definitely empty".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime  # UTC
    title: str
    source: str  # domain or publication
    url: str
    tone: Decimal | None = None  # GDELT scale: roughly -100..+100
    themes: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()

    @field_validator("timestamp")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return _require_utc(v)


_TRADE_CALL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "go long", "going short", "let's go long"
    re.compile(r"\b(?:go|going|will go|to go)\s+(?:long|short)\b", re.IGNORECASE),
    re.compile(r"\benter(?:ing)?\s+(?:a\s+)?(?:long|short)\b", re.IGNORECASE),
    re.compile(r"\b(?:buy|sell)\s+(?:the|now|at|near|on|signal)\b", re.IGNORECASE),
    re.compile(r"\b(?:should|recommend|advise)\s+(?:buy|sell|long|short)\b", re.IGNORECASE),
    re.compile(r"\b(?:long|short|bullish|bearish)\s+bias\b", re.IGNORECASE),
    re.compile(r"\btake\s+profit\b|\bstop\s+loss\b", re.IGNORECASE),
    re.compile(r"\btarget\s+(?:price|level)\b", re.IGNORECASE),
    re.compile(r"\b(?:bullish|bearish)\s+setup\b", re.IGNORECASE),
)


def _scrub_trade_calls(field_name: str, text: str) -> str:
    """Reject anything that looks like a trade recommendation in a free-text
    field. Used by :class:`MarketContextReport` text-bearing fields.

    The regex set is intentionally focused on imperative trade language —
    "USD weaker on CPI surprise" is descriptive and fine; "buy USD now" is
    not. False positives are preferred over false negatives here: the
    context agent's contract forbids trade calls in its output entirely.
    """
    for pattern in _TRADE_CALL_PATTERNS:
        m = pattern.search(text)
        if m:
            raise ValueError(
                f"trade-recommendation language in {field_name}: matched {m.group()!r}"
            )
    return text


class RegimeAssessment(BaseModel):
    """Regime tags for one currency pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair: str
    risk_state: RiskState
    trend_state: TrendState
    vol_state: VolState
    confidence: Annotated[Decimal, Field(ge=Decimal("0"), le=Decimal("1"))]
    rationale: str = ""

    @field_validator("pair")
    @classmethod
    def _normalize_pair(cls, v: str) -> str:
        if not v or not v.isalpha():
            raise ValueError("pair must be alphabetic, e.g. 'EURUSD'")
        return v.upper()

    @field_validator("rationale")
    @classmethod
    def _no_trade_calls(cls, v: str) -> str:
        return _scrub_trade_calls("rationale", v)


class ScheduledEventSummary(BaseModel):
    """One scheduled release the agent flagged as material."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    when: datetime
    currency: str
    name: str
    impact: ImpactLevel

    @field_validator("when")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return _require_utc(v)

    @field_validator("currency")
    @classmethod
    def _ccy(cls, v: str) -> str:
        v = v.upper()
        if not v.isalpha() or len(v) != 3:
            raise ValueError(f"currency must be ISO-4217, got {v!r}")
        return v


class NotableSurprise(BaseModel):
    """A past release whose actual diverged meaningfully from forecast."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    when: datetime
    currency: str
    name: str
    actual: Decimal
    forecast: Decimal
    surprise: Decimal
    significance: str = ""

    @field_validator("when")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return _require_utc(v)

    @field_validator("currency")
    @classmethod
    def _ccy(cls, v: str) -> str:
        v = v.upper()
        if not v.isalpha() or len(v) != 3:
            raise ValueError(f"currency must be ISO-4217, got {v!r}")
        return v

    @field_validator("significance")
    @classmethod
    def _no_trade_calls(cls, v: str) -> str:
        return _scrub_trade_calls("significance", v)


class SentimentRead(BaseModel):
    """Sentiment the agent derived FROM HEADLINES.

    Scoped to a currency rather than a pair. GDELT's dictionary tone is not
    ingested — the agent reads headline text itself; see the
    market_context_agent spec.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    currency: str
    label: SentimentLabel
    score: Annotated[Decimal, Field(ge=Decimal("-1"), le=Decimal("1"))]
    rationale: str = ""

    @field_validator("currency")
    @classmethod
    def _ccy(cls, v: str) -> str:
        v = v.upper()
        if not v.isalpha() or len(v) != 3:
            raise ValueError(f"currency must be ISO-4217, got {v!r}")
        return v

    @field_validator("rationale")
    @classmethod
    def _no_trade_calls(cls, v: str) -> str:
        return _scrub_trade_calls("rationale", v)


class RiskFlagOut(BaseModel):
    """One named risk flag in the report (e.g. ``FOMC_T+18h``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    severity: FlagSeverity
    description: str = ""

    @field_validator("description")
    @classmethod
    def _no_trade_calls(cls, v: str) -> str:
        return _scrub_trade_calls("description", v)


class MarketContextReport(BaseModel):
    """The structured output of :class:`market_context_agent`.

    This model is the agent's contract; downstream consumers (strategy
    lab, optimization, critic, reporting) only see this — never raw feeds.

    **Invariant.** No trade recommendations, anywhere. Free-text fields
    are validated against a focused regex of imperative trade language
    ("go long", "buy now", "long bias", "stop loss", "target price", …).
    The agent prompt also forbids them; this model is the belt-and-braces
    machine check.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    as_of: datetime
    schema_version: str = "1.0"
    regimes: tuple[RegimeAssessment, ...] = ()
    key_scheduled_events: tuple[ScheduledEventSummary, ...] = ()
    notable_surprises: tuple[NotableSurprise, ...] = ()
    sentiment: tuple[SentimentRead, ...] = ()
    risk_flags: tuple[RiskFlagOut, ...] = ()
    notes: str = ""
    confidence: Annotated[Decimal, Field(ge=Decimal("0"), le=Decimal("1"))] = Decimal("0.5")

    @field_validator("as_of")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return _require_utc(v)

    @field_validator("notes")
    @classmethod
    def _no_trade_calls(cls, v: str) -> str:
        if len(v) > 1000:
            raise ValueError("notes must be ≤ 1000 chars")
        return _scrub_trade_calls("notes", v)


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
    def _normalize_groups(cls, v: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
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
    reason: str  # human-readable
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


class PairProfile(BaseModel):
    """Structural characterisation of a single pair over a lookback window.

    **This profile contains NO measure of historical return, expected profit,
    or any performance score — by design.** Selecting a trading universe by
    past return is selection bias; profitability is decided later by the
    backtester + out-of-sample + critic, never here. Every field is structural:
    data completeness, dispersion/range, cost-to-move, and behaviour
    descriptors.

    The behaviour descriptors (``autocorrelations``, ``variance_ratio``,
    ``behavior_descriptor``) are **descriptive and regime-dependent, NOT
    predictive** — they characterise how the series moved in-sample and must
    not be read as a forecast of future direction or profit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair: str
    granularity: Granularity

    # --- Data coverage / completeness ---
    candle_count: NonNegativeInt
    expected_count: NonNegativeInt
    coverage_ratio: Decimal  # candle_count / expected_count, clamped to [0, 1]
    gap_count: NonNegativeInt  # inter-candle intervals beyond normal cadence
    largest_gap_bars: NonNegativeInt  # biggest single gap, in missing-bar units
    window_start: datetime | None
    window_end: datetime | None

    # --- Dispersion / range (magnitude of movement, NOT direction or profit) ---
    last_close: Decimal
    atr: Decimal  # average true range, price units
    atr_pct: Decimal  # atr / last_close (fraction)
    realized_vol_annualized: float  # annualised stdev of per-bar simple returns

    # --- Cost-to-move ---
    spread: Decimal | None  # current bid/ask spread, price units (None if no quote)
    spread_to_atr: Decimal | None  # spread / atr; larger = costlier to move

    # --- Behaviour descriptors: DESCRIPTIVE & REGIME-DEPENDENT, NOT predictive ---
    autocorrelations: tuple[tuple[int, float], ...]  # (lag, coefficient) pairs
    variance_ratio: float
    variance_ratio_horizon: int
    behavior_descriptor: str

    @field_validator("pair")
    @classmethod
    def _normalize_pair(cls, v: str) -> str:
        if not v or not v.isalpha():
            raise ValueError("pair must be alphabetic, e.g. 'EURUSD'")
        return v.upper()

    @field_validator("window_start", "window_end")
    @classmethod
    def _utc_only(cls, v: datetime | None) -> datetime | None:
        return None if v is None else _require_utc(v)


# ---------------------------------------------------------------------------
# Strategy proposals (strategy_lab_agent output)
# ---------------------------------------------------------------------------


class StrategyArchetype(StrEnum):
    """The FIXED, code-defined set of strategy types the backtester can run.

    The strategy_lab_agent may ONLY propose an archetype from this enum; it
    cannot invent a strategy type the engine can't construct. Each archetype
    maps to a concrete :class:`core.strategy.Strategy` builder in
    ``core.strategy.STRATEGY_REGISTRY``. Add a value here only when the engine
    actually supports it.
    """

    MA_CROSSOVER = "ma_crossover"


_EXECUTION_INTENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bgo(?:ing)?\s+live\b", re.IGNORECASE),
    re.compile(r"\blive[\s-]?trad(?:e|es|ing)\b", re.IGNORECASE),
    re.compile(r"\bdeploy(?:ing|ment|ed)?\b", re.IGNORECASE),
    re.compile(r"\b(?:place|send|route|submit|fire)\s+(?:the\s+|a\s+)?orders?\b", re.IGNORECASE),
    re.compile(r"\bexecute\s+(?:the\s+|a\s+)?(?:trade|order|position)s?\b", re.IGNORECASE),
    re.compile(
        r"\b(?:enable|activate|turn\s+on)\s+(?:live|real[\s-]?money|trading)\b", re.IGNORECASE
    ),
    re.compile(r"\breal[\s-]?money\b", re.IGNORECASE),
)


def _scrub_execution_intent(field_name: str, text: str) -> str:
    """Reject any language that signals intent to EXECUTE / DEPLOY / trade live.

    The strategy lab proposes specs for the backtester to judge — it must never
    signal that anything should be traded live. Belt-and-braces machine check
    mirroring :func:`_scrub_trade_calls`; the agent prompt also forbids it.
    """
    for pattern in _EXECUTION_INTENT_PATTERNS:
        m = pattern.search(text)
        if m:
            raise ValueError(
                f"execution/deployment language in {field_name}: matched {m.group()!r}"
            )
    return text


class StrategyParam(BaseModel):
    """One concrete parameter value (e.g. ``fast_period = 10``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: Decimal


class ParamRange(BaseModel):
    """A bounded range one parameter's optimizer may explore later.

    The optimization_agent's sandbox: ``[low, high]`` inclusive. Validated as
    sane here (``low < high``); archetype-specific limits are enforced against
    ``core.strategy.STRATEGY_REGISTRY``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    low: Decimal
    high: Decimal

    @model_validator(mode="after")
    def _low_below_high(self) -> ParamRange:
        if self.low >= self.high:
            raise ValueError(f"param {self.name!r} range low {self.low} >= high {self.high}")
        return self


class StrategyCandidate(BaseModel):
    """A single proposed strategy SPEC the deterministic backtester can run.

    **The agent proposes; it does not run, optimize, execute, or approve.**
    The archetype is restricted to :class:`StrategyArchetype`; the instrument
    to the allowed universe; and every candidate must be constructible into a
    real :class:`core.strategy.Strategy` before the output is accepted. There
    is NO execution / deployment / live-trading field anywhere, and the
    free-text ``rationale`` is scrubbed for execution intent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    run_id: str
    archetype: StrategyArchetype
    instrument: str
    timeframe: Granularity
    parameters: tuple[StrategyParam, ...]
    parameter_ranges: tuple[ParamRange, ...]
    rationale: str  # ties the proposal to the MarketContextReport

    @field_validator("instrument")
    @classmethod
    def _normalize_instrument(cls, v: str) -> str:
        if not v or not v.isalpha():
            raise ValueError("instrument must be alphabetic, e.g. 'EURUSD'")
        return v.upper()

    @field_validator("rationale")
    @classmethod
    def _no_execution_intent(cls, v: str) -> str:
        if len(v) > 500:
            raise ValueError("rationale must be ≤ 500 chars")
        return _scrub_execution_intent("rationale", v)

    def params_as_dict(self) -> dict[str, Decimal]:
        return {p.name: p.value for p in self.parameters}

    def ranges_as_dict(self) -> dict[str, ParamRange]:
        return {r.name: r for r in self.parameter_ranges}


class StrategyProposal(BaseModel):
    """The strategy_lab_agent's structured output: a small set of candidates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    as_of: datetime
    schema_version: str = "1.0"
    candidates: tuple[StrategyCandidate, ...] = ()

    @field_validator("as_of")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return _require_utc(v)
