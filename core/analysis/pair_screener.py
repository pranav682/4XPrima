"""Deterministic structural pair screener — pick a *starting* trading universe.

Pure fast-loop Python: **NO LLM, NO strategy logic, NO trading, NO agents.** It
reads historical candles (via :class:`core.market_data.CandleProvider`) and an
optional current quote (via :class:`core.market_data.PriceProvider`) and emits a
structural profile per pair plus a diversified shortlist.

CRITICAL — this is **STRUCTURAL CHARACTERISATION to narrow the universe, NOT a
profitability ranking.** It does not compute, rank, or select pairs by
historical return, momentum, or any "expected profit" score. Selecting a
universe by past return is selection bias. Profitability is decided later, and
only there, by the backtester + out-of-sample split + critic. The shortlist is
chosen purely on structural grounds:

1. **sufficient data** (coverage / completeness),
2. **low cost-to-move** (spread relative to ATR), and
3. **low mutual correlation** (a greedily de-correlated, genuinely diverse set).

The behaviour descriptors (return autocorrelations, variance ratio) are
**descriptive and regime-dependent, NOT predictive** — they describe how a
series moved in-sample, never a forecast.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from core.market_data import (
    CandleProvider,
    MarketDataError,
    PriceProvider,
    to_canonical_pair,
    to_oanda_instrument,
)
from core.models import Candle, Granularity, PairProfile

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MAJORS: tuple[str, ...] = (
    "EURUSD",
    "USDJPY",
    "GBPUSD",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
)
DEFAULT_AUTOCORR_LAGS: tuple[int, ...] = (1, 2, 3, 5)

# Nominal seconds per bar, used for gap detection + return annualisation. The
# month/week figures are nominal; this is a screening tool, not a calendar.
_GRANULARITY_SECONDS: dict[Granularity, float] = {
    Granularity.S5: 5.0,
    Granularity.S10: 10.0,
    Granularity.S15: 15.0,
    Granularity.S30: 30.0,
    Granularity.M1: 60.0,
    Granularity.M2: 120.0,
    Granularity.M4: 240.0,
    Granularity.M5: 300.0,
    Granularity.M10: 600.0,
    Granularity.M15: 900.0,
    Granularity.M30: 1800.0,
    Granularity.H1: 3600.0,
    Granularity.H2: 7200.0,
    Granularity.H3: 10800.0,
    Granularity.H4: 14400.0,
    Granularity.H6: 21600.0,
    Granularity.H8: 28800.0,
    Granularity.H12: 43200.0,
    Granularity.D: 86400.0,
    Granularity.W: 604800.0,
    Granularity.M: 2592000.0,  # nominal 30 days
}

# A 252-trading-day year, in seconds (matches core/backtest/metrics.py).
_SECONDS_PER_TRADING_YEAR: float = 252.0 * 24.0 * 3600.0
# A forex weekend close is ~2.5 days; intervals up to this from a Fri/Sat bar
# are normal market closure, not missing data.
_WEEKEND_MAX_SECONDS: float = 3.2 * 86400.0


# ---------------------------------------------------------------------------
# Pure metric functions (structural only — none of these is a return ranking)
# ---------------------------------------------------------------------------


def granularity_seconds(granularity: Granularity) -> float:
    """Nominal seconds per bar for ``granularity``."""
    return _GRANULARITY_SECONDS[granularity]


def periods_per_year(granularity: Granularity) -> float:
    """Bars per (252-day) trading year for ``granularity``."""
    return _SECONDS_PER_TRADING_YEAR / granularity_seconds(granularity)


def simple_returns(closes: Sequence[Decimal]) -> list[float]:
    """Per-bar simple returns ``(c[i] - c[i-1]) / c[i-1]`` as floats.

    Returns dispersion input only — direction/magnitude here are never ranked
    or summed into a performance score.
    """
    out: list[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if prev > 0:
            out.append(float((closes[i] - prev) / prev))
    return out


def realized_volatility(returns: Sequence[float], bars_per_year: float) -> float:
    """Annualised standard deviation of per-bar returns (population stdev)."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / n
    return math.sqrt(variance) * math.sqrt(bars_per_year)


def average_true_range(candles: Sequence[Candle]) -> Decimal:
    """Mean True Range over the window, in price units.

    ``TR = max(high-low, |high-prev_close|, |low-prev_close|)`` with the first
    bar's TR taken as its high-low range. A dispersion measure, not a return.
    """
    if not candles:
        return Decimal("0")
    trs: list[Decimal] = []
    prev_close: Decimal | None = None
    for c in candles:
        if prev_close is None:
            tr = c.high - c.low
        else:
            tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        trs.append(tr)
        prev_close = c.close
    return sum(trs, Decimal("0")) / Decimal(len(trs))


def autocorrelation(returns: Sequence[float], lag: int) -> float:
    """Biased sample autocorrelation of ``returns`` at ``lag`` (Pearson).

    Returns 0.0 when the series is too short or has zero variance. Descriptive
    only — a regime-dependent structure measure, never a forecast.
    """
    n = len(returns)
    if lag <= 0 or n <= lag + 1:
        return 0.0
    mean = sum(returns) / n
    denom = sum((r - mean) ** 2 for r in returns)
    if denom == 0.0:
        return 0.0
    num = sum((returns[i] - mean) * (returns[i + lag] - mean) for i in range(n - lag))
    return num / denom


def variance_ratio(returns: Sequence[float], horizon: int) -> float:
    """Lo-MacKinlay variance ratio VR(q) using overlapping q-bar returns.

    ``VR ≈ 1`` random-walk-like, ``> 1`` persistent/trending, ``< 1``
    mean-reverting. **Descriptive and regime-dependent, NOT predictive.**
    Returns 1.0 (the neutral value) when the series is too short or degenerate.
    """
    n = len(returns)
    if horizon < 2 or n < horizon + 1:
        return 1.0
    mean = sum(returns) / n
    var_1 = sum((r - mean) ** 2 for r in returns) / n
    if var_1 == 0.0:
        return 1.0
    q_sums = [sum(returns[i : i + horizon]) for i in range(0, n - horizon + 1)]
    mean_q = horizon * mean
    var_q = sum((y - mean_q) ** 2 for y in q_sums) / len(q_sums)
    return var_q / (horizon * var_1)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation of two equal-length series; 0.0 if undefined."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0.0 or syy == 0.0:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return sxy / math.sqrt(sxx * syy)


def _is_weekend_gap(start: datetime, delta_seconds: float) -> bool:
    """A Fri/Sat-anchored interval no longer than ~3.2 days is the normal
    forex weekend close, not missing data."""
    return start.weekday() >= 4 and delta_seconds <= _WEEKEND_MAX_SECONDS


def detect_gaps(candles: Sequence[Candle], granularity: Granularity) -> tuple[int, int]:
    """Count cadence gaps in a candle series → ``(gap_count, largest_gap_bars)``.

    A gap is an inter-candle interval longer than ``1.5x`` the nominal bar
    interval, excluding regular weekend closes. ``largest_gap_bars`` is the
    biggest single gap expressed in missing-bar units. Tuned for the daily/
    weekly cadence this tool screens by default.
    """
    if len(candles) < 2:
        return (0, 0)
    nominal = granularity_seconds(granularity)
    threshold = 1.5 * nominal
    gap_count = 0
    largest_gap_bars = 0
    for i in range(1, len(candles)):
        delta = (candles[i].time - candles[i - 1].time).total_seconds()
        if delta <= threshold:
            continue
        if _is_weekend_gap(candles[i - 1].time, delta):
            continue
        gap_count += 1
        missing = max(0, round(delta / nominal) - 1)
        largest_gap_bars = max(largest_gap_bars, missing)
    return (gap_count, largest_gap_bars)


def behavior_descriptor(variance_ratio_value: float, lag1_autocorr: float) -> str:
    """Short descriptive label from VR + lag-1 autocorrelation.

    **Descriptive and regime-dependent, NOT predictive.**
    """
    if variance_ratio_value > 1.1 or lag1_autocorr > 0.1:
        return "persistent/trending (descriptive, regime-dependent)"
    if variance_ratio_value < 0.9 or lag1_autocorr < -0.1:
        return "mean-reverting (descriptive, regime-dependent)"
    return "random-walk-like (descriptive, regime-dependent)"


# ---------------------------------------------------------------------------
# Profile construction
# ---------------------------------------------------------------------------


def _quantize(value: Decimal, places: str = "0.00000001") -> Decimal:
    return value.quantize(Decimal(places))


def build_profile(
    pair: str,
    candles: Sequence[Candle],
    *,
    granularity: Granularity,
    expected_count: int,
    spread: Decimal | None,
    autocorr_lags: Sequence[int] = DEFAULT_AUTOCORR_LAGS,
    variance_ratio_horizon: int = 5,
) -> PairProfile:
    """Build the structural :class:`PairProfile` for one pair. Pure function."""
    ordered = sorted(candles, key=lambda c: c.time)
    count = len(ordered)
    closes = [c.close for c in ordered]
    last_close = closes[-1] if closes else Decimal("0")

    if expected_count > 0:
        ratio = Decimal(count) / Decimal(expected_count)
        coverage = _quantize(min(Decimal("1"), ratio), "0.0001")
    else:
        coverage = Decimal("0")

    gap_count, largest_gap = detect_gaps(ordered, granularity)

    atr = _quantize(average_true_range(ordered))
    atr_pct = _quantize(atr / last_close) if last_close > 0 else Decimal("0")

    returns = simple_returns(closes)
    bars_per_year = periods_per_year(granularity)
    rvol = realized_volatility(returns, bars_per_year)

    autocorrs = tuple((lag, autocorrelation(returns, lag)) for lag in autocorr_lags)
    vr = variance_ratio(returns, variance_ratio_horizon)
    lag1 = autocorrs[0][1] if autocorrs else 0.0
    descriptor = behavior_descriptor(vr, lag1) if len(returns) >= 2 else "insufficient data"

    if spread is not None and atr > 0:
        spread_to_atr: Decimal | None = _quantize(spread / atr)
    else:
        spread_to_atr = None

    return PairProfile(
        pair=pair,
        granularity=granularity,
        candle_count=count,
        expected_count=expected_count,
        coverage_ratio=coverage,
        gap_count=gap_count,
        largest_gap_bars=largest_gap,
        window_start=ordered[0].time if ordered else None,
        window_end=ordered[-1].time if ordered else None,
        last_close=last_close,
        atr=atr,
        atr_pct=atr_pct,
        realized_vol_annualized=rvol,
        spread=spread,
        spread_to_atr=spread_to_atr,
        autocorrelations=autocorrs,
        variance_ratio=vr,
        variance_ratio_horizon=variance_ratio_horizon,
        behavior_descriptor=descriptor,
    )


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------


class ScreenConfig(BaseModel):
    """Inputs for one screening run. Frozen; all thresholds are structural."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pairs: tuple[str, ...] = DEFAULT_MAJORS
    granularity: Granularity = Granularity.D
    lookback_count: int = Field(default=500, gt=0)  # ~2 trading years of dailies
    shortlist_size: int = Field(default=5, gt=0)
    autocorr_lags: tuple[int, ...] = DEFAULT_AUTOCORR_LAGS
    variance_ratio_horizon: int = Field(default=5, ge=2)

    # --- Structural eligibility thresholds (NOT return thresholds) ---
    min_coverage_ratio: Decimal = Field(default=Decimal("0.90"), ge=0, le=1)
    min_candles: int = Field(default=100, ge=2)
    max_correlation: float = Field(default=0.80, gt=0, le=1)
    # Cost-to-move cap: spread may be at most this fraction of ATR. None = no cap.
    max_spread_to_atr: Decimal | None = Field(default=Decimal("0.25"), gt=0)


class CorrelationMatrix(BaseModel):
    """Symmetric matrix of return correlations across pairs with data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pairs: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]

    def get(self, a: str, b: str) -> float:
        """Correlation between ``a`` and ``b``; 0.0 if either is absent."""
        try:
            ia = self.pairs.index(a)
            ib = self.pairs.index(b)
        except ValueError:
            return 0.0
        return self.matrix[ia][ib]


class ShortlistEntry(BaseModel):
    """One selected pair and the STRUCTURAL reason it was chosen."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair: str
    selection_rank: int  # 1-based order of selection (NOT a profitability rank)
    cost_to_move: Decimal | None  # spread / ATR at selection time
    max_correlation_with_selected: float
    reason: str


class ExclusionEntry(BaseModel):
    """One pair left out of the shortlist, with the structural reason."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair: str
    reason: str


class ScreeningReport(BaseModel):
    """The full output of one screen: profiles + correlation + shortlist."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of: datetime
    granularity: Granularity
    lookback_count: int
    candidate_pairs: tuple[str, ...]
    profiles: tuple[PairProfile, ...]
    correlation: CorrelationMatrix
    shortlist: tuple[ShortlistEntry, ...]
    excluded: tuple[ExclusionEntry, ...]

    def render(self) -> str:
        """A readable text report: profile table + correlation matrix + picks."""
        return _render_report(self)


# ---------------------------------------------------------------------------
# Screener
# ---------------------------------------------------------------------------


def _canonical(pair: str) -> str:
    return to_canonical_pair(to_oanda_instrument(pair))


class PairScreener:
    """Run a structural screen over candidate pairs.

    Construct with a :class:`CandleProvider` (required) and an optional
    :class:`PriceProvider` (for the live spread used in cost-to-move). Both are
    Protocols, so tests inject in-memory stubs — no network.
    """

    def __init__(
        self,
        candle_provider: CandleProvider,
        *,
        price_provider: PriceProvider | None = None,
    ) -> None:
        self._candles = candle_provider
        self._prices = price_provider

    # -------------------------------------------------------------- public API

    def screen(self, config: ScreenConfig, *, as_of: datetime) -> ScreeningReport:
        profiles: list[PairProfile] = []
        returns_by_pair: dict[str, dict[datetime, float]] = {}
        excluded: list[ExclusionEntry] = []
        canon_pairs: list[str] = []

        for raw_pair in config.pairs:
            try:
                canon = _canonical(raw_pair)
            except ValueError:
                excluded.append(ExclusionEntry(pair=raw_pair.upper(), reason="invalid pair format"))
                continue
            canon_pairs.append(canon)

            candles = self._fetch_candles(raw_pair, config)
            spread = self._fetch_spread(raw_pair)
            profile = build_profile(
                canon,
                candles,
                granularity=config.granularity,
                expected_count=config.lookback_count,
                spread=spread,
                autocorr_lags=config.autocorr_lags,
                variance_ratio_horizon=config.variance_ratio_horizon,
            )
            profiles.append(profile)
            if len(candles) >= 2:
                returns_by_pair[canon] = _returns_by_time(candles)

        correlation = _build_correlation_matrix(returns_by_pair)
        shortlist, corr_excluded = _select_shortlist(profiles, correlation, config)
        excluded.extend(corr_excluded)

        return ScreeningReport(
            as_of=as_of,
            granularity=config.granularity,
            lookback_count=config.lookback_count,
            candidate_pairs=tuple(canon_pairs),
            profiles=tuple(profiles),
            correlation=correlation,
            shortlist=tuple(shortlist),
            excluded=tuple(excluded),
        )

    # ----------------------------------------------------------- data fetch

    def _fetch_candles(self, pair: str, config: ScreenConfig) -> list[Candle]:
        try:
            return self._candles.get_candles(
                pair, granularity=config.granularity, count=config.lookback_count
            )
        except (MarketDataError, KeyError, ValueError):
            # Missing/unavailable history is screened out, not fatal.
            return []

    def _fetch_spread(self, pair: str) -> Decimal | None:
        if self._prices is None:
            return None
        try:
            quote = self._prices.get_quote(pair)
        except (MarketDataError, KeyError, ValueError):
            return None
        return quote.spread


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def _returns_by_time(candles: Sequence[Candle]) -> dict[datetime, float]:
    """Map each candle time to the simple return from the previous bar."""
    ordered = sorted(candles, key=lambda c: c.time)
    out: dict[datetime, float] = {}
    for i in range(1, len(ordered)):
        prev = ordered[i - 1].close
        if prev > 0:
            out[ordered[i].time] = float((ordered[i].close - prev) / prev)
    return out


def _build_correlation_matrix(
    returns_by_pair: dict[str, dict[datetime, float]],
) -> CorrelationMatrix:
    pairs = tuple(sorted(returns_by_pair))
    n = len(pairs)
    matrix: list[tuple[float, ...]] = []
    for i in range(n):
        row: list[float] = []
        for j in range(n):
            if i == j:
                row.append(1.0)
                continue
            a = returns_by_pair[pairs[i]]
            b = returns_by_pair[pairs[j]]
            common = sorted(a.keys() & b.keys())
            xs = [a[t] for t in common]
            ys = [b[t] for t in common]
            row.append(round(pearson(xs, ys), 6))
        matrix.append(tuple(row))
    return CorrelationMatrix(pairs=pairs, matrix=tuple(matrix))


# ---------------------------------------------------------------------------
# Shortlist selection — STRUCTURAL ONLY (data + cost-to-move + diversity)
# ---------------------------------------------------------------------------


def _cost_sort_key(profile: PairProfile) -> tuple[int, Decimal]:
    # Measured cost-to-move sorts first (ascending = cheaper); pairs with no
    # spread measurement sort last. This is a COST order, never a return order.
    if profile.spread_to_atr is None:
        return (1, Decimal("0"))
    return (0, profile.spread_to_atr)


def _ineligibility_reason(profile: PairProfile, config: ScreenConfig) -> str | None:
    if profile.candle_count < config.min_candles:
        return f"insufficient data ({profile.candle_count} < " f"{config.min_candles} candles)"
    if profile.coverage_ratio < config.min_coverage_ratio:
        return f"coverage {profile.coverage_ratio} < {config.min_coverage_ratio}"
    if (
        config.max_spread_to_atr is not None
        and profile.spread_to_atr is not None
        and profile.spread_to_atr > config.max_spread_to_atr
    ):
        return f"cost-to-move spread/ATR {profile.spread_to_atr} > " f"{config.max_spread_to_atr}"
    return None


def _select_shortlist(
    profiles: Sequence[PairProfile],
    correlation: CorrelationMatrix,
    config: ScreenConfig,
) -> tuple[list[ShortlistEntry], list[ExclusionEntry]]:
    """Greedy, structural selection: cheapest-to-move first, dropping any pair
    too correlated with one already chosen, until ``shortlist_size`` is met."""
    excluded: list[ExclusionEntry] = []
    eligible: list[PairProfile] = []
    for profile in profiles:
        reason = _ineligibility_reason(profile, config)
        if reason is None:
            eligible.append(profile)
        else:
            excluded.append(ExclusionEntry(pair=profile.pair, reason=reason))

    # Consider cheapest-to-move first. Correlation then enforces diversity.
    eligible.sort(key=_cost_sort_key)

    selected: list[ShortlistEntry] = []
    chosen_pairs: list[str] = []
    for profile in eligible:
        if len(selected) >= config.shortlist_size:
            excluded.append(
                ExclusionEntry(
                    pair=profile.pair,
                    reason=f"shortlist full ({config.shortlist_size} selected)",
                )
            )
            continue

        max_corr = 0.0
        most_corr_pair = ""
        for other in chosen_pairs:
            c = abs(correlation.get(profile.pair, other))
            if c > max_corr:
                max_corr = c
                most_corr_pair = other

        if chosen_pairs and max_corr > config.max_correlation:
            excluded.append(
                ExclusionEntry(
                    pair=profile.pair,
                    reason=(
                        f"|correlation| {max_corr:.2f} with {most_corr_pair} "
                        f"exceeds {config.max_correlation:.2f}"
                    ),
                )
            )
            continue

        rank = len(selected) + 1
        cost = profile.spread_to_atr
        cost_str = "n/a" if cost is None else f"{cost}"
        if not chosen_pairs:
            reason = f"lowest cost-to-move among eligible (spread/ATR={cost_str})"
        else:
            reason = (
                f"cost-to-move spread/ATR={cost_str}; |corr| {max_corr:.2f} with "
                f"selected ≤ {config.max_correlation:.2f}"
            )
        selected.append(
            ShortlistEntry(
                pair=profile.pair,
                selection_rank=rank,
                cost_to_move=cost,
                max_correlation_with_selected=round(max_corr, 6),
                reason=reason,
            )
        )
        chosen_pairs.append(profile.pair)

    return selected, excluded


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_report(report: ScreeningReport) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("PAIR SCREEN — structural characterisation (NOT a return ranking)")
    lines.append("=" * 78)
    lines.append(f"as_of:      {report.as_of.isoformat()}")
    lines.append(f"granularity:{report.granularity.value}   lookback={report.lookback_count} bars")
    lines.append(f"candidates: {', '.join(report.candidate_pairs)}")
    lines.append("")

    lines.append("PROFILES")
    header = (
        f"  {'pair':8} {'cov':>5} {'candles':>7} {'gaps':>4} "
        f"{'ATR%':>7} {'annVol':>7} {'spr/ATR':>8} {'VR':>5}  behaviour"
    )
    lines.append(header)
    for p in sorted(report.profiles, key=lambda x: x.pair):
        spr = "n/a" if p.spread_to_atr is None else f"{p.spread_to_atr:.4f}"
        lines.append(
            f"  {p.pair:8} {p.coverage_ratio:>5} {p.candle_count:>7} "
            f"{p.gap_count:>4} {float(p.atr_pct):>7.4f} "
            f"{p.realized_vol_annualized:>7.3f} {spr:>8} "
            f"{p.variance_ratio:>5.2f}  {p.behavior_descriptor}"
        )

    lines.append("")
    lines.append("RETURN CORRELATION MATRIX")
    if report.correlation.pairs:
        head = "          " + " ".join(f"{p:>8}" for p in report.correlation.pairs)
        lines.append(head)
        for i, pair in enumerate(report.correlation.pairs):
            row = " ".join(f"{v:>8.2f}" for v in report.correlation.matrix[i])
            lines.append(f"  {pair:8} {row}")
    else:
        lines.append("  (no pairs with sufficient data)")

    lines.append("")
    lines.append("SHORTLIST (structural: data + cost-to-move + low correlation)")
    if report.shortlist:
        for e in report.shortlist:
            lines.append(f"  {e.selection_rank}. {e.pair:8} — {e.reason}")
    else:
        lines.append("  (none eligible)")

    if report.excluded:
        lines.append("")
        lines.append("EXCLUDED")
        for x in report.excluded:
            lines.append(f"  {x.pair:8} — {x.reason}")

    lines.append("=" * 78)
    return "\n".join(lines)


__all__ = [
    "DEFAULT_AUTOCORR_LAGS",
    "DEFAULT_MAJORS",
    "CorrelationMatrix",
    "ExclusionEntry",
    "PairScreener",
    "ScreenConfig",
    "ScreeningReport",
    "ShortlistEntry",
    "autocorrelation",
    "average_true_range",
    "behavior_descriptor",
    "build_profile",
    "detect_gaps",
    "pearson",
    "periods_per_year",
    "realized_volatility",
    "simple_returns",
    "variance_ratio",
]
