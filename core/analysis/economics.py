"""Deterministic net-of-cost economics + historical decay read — NO LLM, NO engine.

Given the ALREADY-PERSISTED :class:`core.models.BacktestEvidence` for a
candidate's in-sample (and optional out-of-sample) windows, this derives the
economic-health figures a trader actually needs. It reads stored verbatim values
and does only arithmetic on them — it NEVER imports or calls the backtest engine
to recompute. (This module must not import ``core.backtest``.)

The principles it encodes:

- **Expectancy, not win rate, is the measure.** Win rate is only meaningful next
  to average win / average loss; on its own it is misleading, so the helper
  always returns them together and the UI never shows win rate alone.
- **Net-of-cost is the only number that counts.** The engine stores each trade's
  ``realized_pnl`` GROSS of commission/swap (costs live in ``cost_total``), so
  ``avg_trade_pnl`` is the *gross* per-trade edge; the **net** expectancy is
  ``avg_trade_pnl - cost_total / trade_count``. Net leads; gross is subordinate.
- **Edge must dwarf cost.** ``cost_to_edge`` = total cost / gross P&L; a strategy
  whose costs eat most of its gross edge is flagged, not celebrated.

The IS→OOS comparison here is **historical** decay (backtest in-sample vs the
sealed out-of-sample slice). It is NOT live/forward-test decay — that needs a
running champion (Stage 4, not built).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from core.models import BacktestEvidence, EvidenceSegment, JsonDecimal

# ---------------------------------------------------------------------------
# Thresholds (deterministic flag policy — documented; lives in config)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EconomicsThresholds:
    """Explicit, deterministic thresholds for the retire/concern flags. A flag is
    the system working, not an alarm — it says "this is no longer (or never was)
    economic", honestly."""

    # Broker's share of gross P&L. Above the concern ceiling = costs eat the edge.
    cost_to_edge_concern: Decimal = Decimal("0.50")  # broker takes ≥ 50%
    cost_to_edge_retire: Decimal = Decimal("1.00")  # costs ≥ gross edge (net ≤ 0)
    # Out-of-sample expectancy as a fraction of in-sample (historical decay).
    oos_fraction_concern: Decimal = Decimal("0.60")
    oos_fraction_retire: Decimal = Decimal("0.25")
    # Statistical-power floor for a trustworthy out-of-sample read.
    min_trade_count: int = 30


DEFAULT_THRESHOLDS = EconomicsThresholds()


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class EconomicFlag(StrEnum):
    OK = "ok"
    CONCERN = "concern"
    RETIRE = "retire"


class EconomicConcern(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    level: EconomicFlag  # concern | retire
    reason: str


class WindowEconomics(BaseModel):
    """Net-of-cost economics for ONE window (in-sample or out-of-sample). Net
    figures lead; gross is shown but subordinate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    segment: EvidenceSegment
    trade_count: int
    # Account-level P&L for the window (verbatim / arithmetic on verbatim).
    net_pnl: JsonDecimal  # ending_equity - starting_balance — the engine's real number
    cost_total: JsonDecimal
    gross_pnl: JsonDecimal  # net_pnl + cost_total
    return_pct: JsonDecimal
    # Per-trade edge.
    win_rate: float
    gross_expectancy_per_trade: JsonDecimal  # avg_trade_pnl (gross of costs)
    net_expectancy_per_trade: JsonDecimal | None  # gross - cost/trade; None if 0 trades
    cost_per_trade: JsonDecimal | None
    avg_win: JsonDecimal | None  # gross; None when not cleanly derivable
    avg_loss: JsonDecimal | None  # gross magnitude
    # Cost vs edge.
    cost_to_edge: float | None  # cost_total / gross_pnl, when gross_pnl > 0
    cost_to_edge_label: str
    costs_exceed_gross: bool


class EconomicDecay(BaseModel):
    """HISTORICAL decay: in-sample vs the sealed out-of-sample slice. NOT live."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    in_sample_net_expectancy: JsonDecimal | None
    out_of_sample_net_expectancy: JsonDecimal | None
    oos_expectancy_fraction_of_is: float | None
    oos_return_fraction_of_is: float | None
    note: str = (
        "Historical decay (in-sample → out-of-sample backtest windows). This is "
        "NOT live/forward-test decay — that needs a running champion (not built)."
    )


class CandidateEconomics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    config_hash: str
    candidate_id: str
    pair: str
    in_sample: WindowEconomics
    out_of_sample: WindowEconomics | None
    decay: EconomicDecay | None
    flag: EconomicFlag
    concerns: tuple[EconomicConcern, ...]
    amortized_research_cost_usd: JsonDecimal | None = None


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _decompose_avg_win_loss(
    *, win_rate: float, profit_factor: float | None, gross_expectancy: Decimal, trade_count: int
) -> tuple[Decimal | None, Decimal | None]:
    """Recover GROSS average win and average loss (magnitudes) from the stored
    win_rate / profit_factor / per-trade gross expectancy. Pure arithmetic; the
    engine is never consulted. Returns (None, None) when the split is degenerate
    (e.g. profit_factor == 1, or no winners/losers)."""
    n = Decimal(trade_count)
    if trade_count == 0:
        return None, None
    w = Decimal(str(win_rate))
    winners = w * n
    losers = n - winners
    sum_realized = gross_expectancy * n  # = gross_wins - gross_loss

    if profit_factor is None:  # no losing trades → all gross_loss is 0
        avg_loss = Decimal("0")
        avg_win = (sum_realized / winners) if winners > 0 else None
        return avg_win, avg_loss

    pf = Decimal(str(profit_factor))
    if pf == 1 or winners <= 0 or losers <= 0:
        return None, None
    gross_loss = sum_realized / (pf - 1)  # magnitude (signs cancel for pf<1 too)
    gross_wins = pf * gross_loss
    avg_win = gross_wins / winners
    avg_loss = (gross_loss / losers).copy_abs()
    return avg_win.copy_abs(), avg_loss


def _window_economics(ev: BacktestEvidence) -> WindowEconomics:
    m = ev.metrics
    net_pnl = ev.ending_equity - ev.starting_balance
    cost_total = ev.cost_total
    gross_pnl = net_pnl + cost_total
    return_pct = (net_pnl / ev.starting_balance) if ev.starting_balance != 0 else Decimal("0")

    gross_exp = m.avg_trade_pnl
    cost_per_trade = (cost_total / Decimal(m.trade_count)) if m.trade_count > 0 else None
    net_exp = (gross_exp - cost_per_trade) if cost_per_trade is not None else None
    avg_win, avg_loss = _decompose_avg_win_loss(
        win_rate=m.win_rate,
        profit_factor=m.profit_factor,
        gross_expectancy=gross_exp,
        trade_count=m.trade_count,
    )

    costs_exceed = gross_pnl <= 0
    cte: float | None = None
    if gross_pnl > 0:
        cte = float(cost_total / gross_pnl)
        label = f"Broker takes {cte * 100:.1f}% of gross P&L"
    elif net_pnl <= 0 and cost_total > 0:
        label = "Costs exceed any gross edge — no profit to take a share of"
    else:
        label = "Cost-to-edge not defined (no gross profit)"

    return WindowEconomics(
        segment=ev.segment,
        trade_count=m.trade_count,
        net_pnl=net_pnl,
        cost_total=cost_total,
        gross_pnl=gross_pnl,
        return_pct=return_pct,
        win_rate=m.win_rate,
        gross_expectancy_per_trade=gross_exp,
        net_expectancy_per_trade=net_exp,
        cost_per_trade=cost_per_trade,
        avg_win=avg_win,
        avg_loss=avg_loss,
        cost_to_edge=cte,
        cost_to_edge_label=label,
        costs_exceed_gross=costs_exceed,
    )


def _decay(is_w: WindowEconomics, oos_w: WindowEconomics) -> EconomicDecay:
    is_exp = is_w.net_expectancy_per_trade
    oos_exp = oos_w.net_expectancy_per_trade
    exp_frac: float | None = None
    if is_exp is not None and oos_exp is not None and is_exp > 0:
        exp_frac = float(oos_exp / is_exp)
    is_ret = Decimal(str(is_w.return_pct))
    ret_frac = float(Decimal(str(oos_w.return_pct)) / is_ret) if is_ret > 0 else None
    return EconomicDecay(
        in_sample_net_expectancy=is_exp,
        out_of_sample_net_expectancy=oos_exp,
        oos_expectancy_fraction_of_is=exp_frac,
        oos_return_fraction_of_is=ret_frac,
    )


def _flags(
    is_w: WindowEconomics,
    oos_w: WindowEconomics | None,
    decay: EconomicDecay | None,
    thresholds: EconomicsThresholds,
) -> tuple[EconomicFlag, tuple[EconomicConcern, ...]]:
    # The judged window is OOS when available (the more honest read), else IS.
    judged = oos_w or is_w
    concerns: list[EconomicConcern] = []

    net_exp = judged.net_expectancy_per_trade
    if net_exp is not None and net_exp <= 0:
        concerns.append(
            EconomicConcern(
                level=EconomicFlag.RETIRE,
                reason=f"Net expectancy negative after costs ({judged.segment.value}).",
            )
        )
    if judged.costs_exceed_gross:
        concerns.append(
            EconomicConcern(
                level=EconomicFlag.RETIRE,
                reason=f"Costs exceed gross edge ({judged.segment.value}) — uneconomic.",
            )
        )
    elif judged.cost_to_edge is not None:
        if judged.cost_to_edge >= float(thresholds.cost_to_edge_retire):
            concerns.append(
                EconomicConcern(
                    level=EconomicFlag.RETIRE,
                    reason=judged.cost_to_edge_label + " — costs consume the edge.",
                )
            )
        elif judged.cost_to_edge >= float(thresholds.cost_to_edge_concern):
            concerns.append(
                EconomicConcern(level=EconomicFlag.CONCERN, reason=judged.cost_to_edge_label + ".")
            )

    if decay is not None and decay.oos_expectancy_fraction_of_is is not None:
        frac = decay.oos_expectancy_fraction_of_is
        pct = f"OOS expectancy is {frac * 100:.0f}% of in-sample"
        if frac < float(thresholds.oos_fraction_retire):
            concerns.append(EconomicConcern(level=EconomicFlag.RETIRE, reason=pct + " — collapse."))
        elif frac < float(thresholds.oos_fraction_concern):
            concerns.append(EconomicConcern(level=EconomicFlag.CONCERN, reason=pct + " — decay."))

    if oos_w is not None and oos_w.trade_count < thresholds.min_trade_count:
        concerns.append(
            EconomicConcern(
                level=EconomicFlag.CONCERN,
                reason=(
                    f"Out-of-sample rests on {oos_w.trade_count} trades — below the "
                    f"statistical-power floor of {thresholds.min_trade_count}."
                ),
            )
        )

    if any(c.level == EconomicFlag.RETIRE for c in concerns):
        flag = EconomicFlag.RETIRE
    elif concerns:
        flag = EconomicFlag.CONCERN
    else:
        flag = EconomicFlag.OK
    return flag, tuple(concerns)


def candidate_economics(
    in_sample: BacktestEvidence,
    out_of_sample: BacktestEvidence | None = None,
    *,
    thresholds: EconomicsThresholds = DEFAULT_THRESHOLDS,
    amortized_research_cost_usd: Decimal | None = None,
) -> CandidateEconomics:
    """Derive the full economic read for one candidate from its persisted
    evidence. Deterministic; reads verbatim values; never recomputes via the
    engine."""
    is_w = _window_economics(in_sample)
    oos_w = _window_economics(out_of_sample) if out_of_sample is not None else None
    decay = _decay(is_w, oos_w) if oos_w is not None else None
    flag, concerns = _flags(is_w, oos_w, decay, thresholds)
    return CandidateEconomics(
        config_hash=in_sample.config_hash,
        candidate_id=in_sample.candidate_id,
        pair=in_sample.pair,
        in_sample=is_w,
        out_of_sample=oos_w,
        decay=decay,
        flag=flag,
        concerns=concerns,
        amortized_research_cost_usd=amortized_research_cost_usd,
    )


__all__ = [
    "DEFAULT_THRESHOLDS",
    "CandidateEconomics",
    "EconomicConcern",
    "EconomicDecay",
    "EconomicFlag",
    "EconomicsThresholds",
    "WindowEconomics",
    "candidate_economics",
]
