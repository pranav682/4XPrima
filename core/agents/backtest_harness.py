"""Deterministic backtest harness — runs the Stage-2 engine, NO LLM.

This is the compute half of backtest_agent: it turns a
:class:`core.models.StrategyProposal` into one
:class:`core.models.BacktestEvidence` per candidate by building the real
strategy and running the deterministic Stage-2 engine over the **in-sample**
window. The LLM never runs here and never produces a number — it only
interprets the evidence this module computes.

In-sample only. The held-out OOS slice is taken via
:class:`core.backtest.DataSplit` and **never accessed** — this module holds no
OOS confirmation token and calls no OOS accessor. The OOS evaluation is a
later, sealed stage.

Determinism: identical inputs ⇒ identical ``config_hash`` ⇒ identical evidence
(the engine is deterministic; this module adds no randomness).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from core.backtest import BacktestEngine, CostModel, DataSplit
from core.backtest.metrics import BacktestMetrics
from core.market_data import CandleProvider, MarketDataError
from core.models import (
    BacktestEvidence,
    BacktestMetricsView,
    Candle,
    GateResult,
    RiskConfig,
    StrategyCandidate,
    StrategyProposal,
)
from core.strategy import build_strategy

# A roomy default risk config — enough that ordinary candidates can actually
# trade in-sample (the engine still routes every order through the real
# RiskManager). Callers override for tighter regimes.
DEFAULT_RISK_CONFIG = RiskConfig(
    max_risk_per_trade_pct=Decimal("0.01"),
    max_portfolio_risk_pct=Decimal("0.10"),
    max_concurrent_positions=5,
    max_exposure_per_pair_pct=Decimal("50"),
    max_correlated_exposure_pct=Decimal("100"),
    correlation_groups={},
    daily_loss_limit_pct=Decimal("0.10"),
    max_drawdown_pct=Decimal("0.40"),
)


@dataclass(frozen=True, slots=True)
class BacktestRunConfig:
    """Inputs for the deterministic harness (everything the engine needs)."""

    lookback_count: int = 1000  # bars to fetch per candidate
    oos_fraction: float = 0.2  # held-out tail — sealed, never run here
    starting_balance: Decimal = Decimal("100000")
    cost_model: CostModel = field(default_factory=CostModel)
    risk_config: RiskConfig = DEFAULT_RISK_CONFIG
    day_rollover_utc_hour: int = 21

    # --- Fixed deterministic gate thresholds (NOT the final word — the critic
    #     + OOS + a human decide deployment; these are cheap triage gates) ---
    min_trade_count: int = 30
    max_drawdown_ceiling: Decimal = Decimal("0.30")
    min_profit_factor: Decimal = Decimal("1.0")


# ---------------------------------------------------------------------------
# Metric / gate helpers
# ---------------------------------------------------------------------------


def _finite_or_none(value: float) -> float | None:
    """Map an infinite metric (no downside / no losing trades) to ``None`` so
    nothing is misrepresented and the value round-trips through JSON."""
    return None if math.isinf(value) else value


def metrics_view(metrics: BacktestMetrics) -> BacktestMetricsView:
    """Copy the engine's metrics into the agent-facing view (inf → None)."""
    return BacktestMetricsView(
        total_return_pct=metrics.total_return_pct,
        annualised_return_pct=metrics.annualised_return_pct,
        sharpe_ratio=metrics.sharpe_ratio,
        sortino_ratio=_finite_or_none(metrics.sortino_ratio),
        max_drawdown_pct=metrics.max_drawdown_pct,
        win_rate=metrics.win_rate,
        profit_factor=_finite_or_none(metrics.profit_factor),
        trade_count=metrics.trade_count,
        avg_trade_pnl=metrics.avg_trade_pnl,
        exposure_pct=metrics.exposure_pct,
    )


def _compute_gates(
    metrics: BacktestMetrics,
    *,
    halted: bool,
    config: BacktestRunConfig,
) -> tuple[GateResult, ...]:
    """Deterministic fixed gates. Computed here, never by the LLM."""
    pf = metrics.profit_factor
    pf_ok = math.isinf(pf) or Decimal(str(pf)) >= config.min_profit_factor
    return (
        GateResult(
            name="min_trade_count",
            passed=metrics.trade_count >= config.min_trade_count,
            detail=f"{metrics.trade_count} trades vs floor {config.min_trade_count}",
        ),
        GateResult(
            name="max_drawdown",
            passed=metrics.max_drawdown_pct <= config.max_drawdown_ceiling,
            detail=f"maxDD {metrics.max_drawdown_pct} vs ceiling {config.max_drawdown_ceiling}",
        ),
        GateResult(
            name="not_halted",
            passed=not halted,
            detail="kill switch tripped" if halted else "ran to completion",
        ),
        GateResult(
            name="in_sample_profit_factor",
            passed=pf_ok,
            detail=(
                "no losing trades"
                if math.isinf(pf)
                else f"PF {pf:.3f} vs floor {config.min_profit_factor}"
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class HarnessError(Exception):
    """A candidate could not be run (no data, too short, or build failure).

    The candidate is omitted from the evidence set — only candidates that were
    ACTUALLY run produce evidence (the agent must not reference others)."""


def run_candidate(
    candidate: StrategyCandidate,
    *,
    candle_provider: CandleProvider,
    config: BacktestRunConfig,
) -> BacktestEvidence:
    """Run ONE candidate's in-sample backtest → :class:`BacktestEvidence`.

    Raises :class:`HarnessError` if the candidate cannot be run.
    """
    try:
        candles = candle_provider.get_candles(
            candidate.instrument,
            granularity=candidate.timeframe,
            count=config.lookback_count,
        )
    except (MarketDataError, KeyError, ValueError) as e:
        raise HarnessError(f"no candles for {candidate.instrument}: {e}") from e

    if len(candles) < 4:
        raise HarnessError(f"too few candles for {candidate.instrument} ({len(candles)})")

    in_sample = _in_sample_only(candles, config.oos_fraction)
    if len(in_sample) < 2:
        raise HarnessError("in-sample window too short to backtest")

    try:
        strategy = build_strategy(candidate)
    except Exception as e:  # not constructible → treated as not-run
        raise HarnessError(f"candidate not constructible: {e}") from e

    engine = BacktestEngine(
        bars=list(in_sample),
        strategy=strategy,
        risk_config=config.risk_config,
        cost_model=config.cost_model,
        starting_balance=config.starting_balance,
        day_rollover_utc_hour=config.day_rollover_utc_hour,
    )
    result = engine.run()

    gates = _compute_gates(result.metrics, halted=result.halted_due_to_kill_switch, config=config)
    return BacktestEvidence(
        candidate_id=candidate.candidate_id,
        config_hash=result.config_hash,
        pair=result.pair,
        in_sample_start=in_sample[0].time,
        in_sample_end=in_sample[-1].time,
        bars_total=result.bars_total,
        bars_processed=result.bars_processed,
        halted_due_to_kill_switch=result.halted_due_to_kill_switch,
        halt_reason=result.halt_reason,
        n_signals_proposed=result.n_signals_proposed,
        n_signals_accepted=result.n_signals_accepted,
        n_signals_rejected=result.n_signals_rejected,
        starting_balance=result.starting_balance,
        ending_equity=result.ending_equity,
        cost_total=result.cost_breakdown.total,
        metrics=metrics_view(result.metrics),
        gates=gates,
        gates_all_passed=all(g.passed for g in gates),
    )


def run_proposal(
    proposal: StrategyProposal,
    *,
    candle_provider: CandleProvider,
    config: BacktestRunConfig,
) -> tuple[tuple[BacktestEvidence, ...], tuple[tuple[str, str], ...]]:
    """Run every candidate. Returns ``(evidence, skipped)`` where ``skipped``
    is ``(candidate_id, reason)`` for candidates that could not be run."""
    evidence: list[BacktestEvidence] = []
    skipped: list[tuple[str, str]] = []
    for candidate in proposal.candidates:
        try:
            evidence.append(
                run_candidate(candidate, candle_provider=candle_provider, config=config)
            )
        except HarnessError as e:
            skipped.append((candidate.candidate_id, str(e)))
    return tuple(evidence), tuple(skipped)


def _in_sample_only(candles: Sequence[Candle], oos_fraction: float) -> tuple[Candle, ...]:
    """Return ONLY the in-sample slice. The OOS tail is split off and sealed —
    this function never returns it and never takes the OOS token."""
    split = DataSplit(candles, oos_fraction=oos_fraction)
    return split.in_sample


__all__ = [
    "DEFAULT_RISK_CONFIG",
    "BacktestRunConfig",
    "HarnessError",
    "metrics_view",
    "run_candidate",
    "run_proposal",
]
