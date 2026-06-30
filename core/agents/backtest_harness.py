"""Deterministic backtest harness — runs the Stage-2 engine, NO LLM.

The compute half of backtest_agent AND critic_agent: it builds the real
strategy and runs the deterministic Stage-2 engine, turning a candidate into
:class:`core.models.BacktestEvidence` / :class:`core.models.RobustnessEvidence`.
The LLM never runs here and never produces a number — it only interprets the
evidence this module computes.

OOS discipline. ``run_candidate`` / ``run_proposal`` are IN-SAMPLE only and
never open the holdout. The held-out OOS slice is opened ONLY by
``run_candidate_oos`` / ``run_robustness`` (the critic stage), which supply the
literal confirmation token to :meth:`core.backtest.DataSplit.access_out_of_sample`.
That token (``_OOS_CONFIRMATION``) lives only in this deterministic module —
the LLM never sees or holds it; it only ever receives the resulting evidence.

Determinism: identical inputs ⇒ identical ``config_hash`` ⇒ identical evidence
(the engine is deterministic; this module adds no randomness).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from core.backtest import BacktestEngine, BacktestResult, CostModel, DataSplit
from core.backtest.metrics import BacktestMetrics
from core.market_data import CandleProvider, MarketDataError
from core.models import (
    BacktestArtifact,
    BacktestEvidence,
    BacktestMetricsView,
    Candle,
    CostStressPoint,
    EquityCurvePoint,
    EvidenceSegment,
    GateResult,
    ParamNeighborResult,
    RiskConfig,
    RobustnessEvidence,
    StrategyCandidate,
    StrategyProposal,
    TradeConcentration,
)
from core.strategy import build_strategy

# The literal confirmation token DataSplit requires to open the held-out OOS
# slice. It lives HERE, in deterministic harness code, and is supplied ONLY by
# run_candidate_oos / run_robustness below. The LLM never sees or holds it.
_OOS_CONFIRMATION = "I_AM_DONE_TUNING"

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
    oos_fraction: float = 0.2  # held-out tail — sealed, opened ONLY by the critic
    starting_balance: Decimal = Decimal("100000")
    cost_model: CostModel = field(default_factory=CostModel)
    risk_config: RiskConfig = DEFAULT_RISK_CONFIG
    day_rollover_utc_hour: int = 21
    # Cost multipliers for the cost-sensitivity stress (critic stage).
    cost_multipliers: tuple[Decimal, ...] = (Decimal("1.5"), Decimal("2.0"))

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


def _fetch(
    candidate: StrategyCandidate, *, candle_provider: CandleProvider, config: BacktestRunConfig
) -> list[Candle]:
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
    return candles


def _run_engine(
    bars: Sequence[Candle],
    candidate: StrategyCandidate,
    *,
    config: BacktestRunConfig,
    cost_model: CostModel,
) -> BacktestResult:
    """Build the strategy and run the engine over ``bars``. Pure / deterministic."""
    try:
        strategy = build_strategy(candidate)
    except Exception as e:  # not constructible → treated as not-run
        raise HarnessError(f"candidate not constructible: {e}") from e
    return BacktestEngine(
        bars=list(bars),
        strategy=strategy,
        risk_config=config.risk_config,
        cost_model=cost_model,
        starting_balance=config.starting_balance,
        day_rollover_utc_hour=config.day_rollover_utc_hour,
    ).run()


def _evidence_from_result(
    candidate_id: str,
    bars: Sequence[Candle],
    result: BacktestResult,
    *,
    segment: EvidenceSegment,
    config: BacktestRunConfig,
) -> BacktestEvidence:
    gates = _compute_gates(result.metrics, halted=result.halted_due_to_kill_switch, config=config)
    return BacktestEvidence(
        candidate_id=candidate_id,
        config_hash=result.config_hash,
        pair=result.pair,
        segment=segment,
        window_start=bars[0].time,
        window_end=bars[-1].time,
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


_ARTIFACT_MAX_POINTS = 400


def _downsample(curve: Sequence[object], max_points: int = _ARTIFACT_MAX_POINTS) -> list[int]:
    """Indices of an evenly-spaced subsample of ``curve`` (always including the
    last point), so a persisted equity curve stays small without distorting its
    shape. Returns indices, not points, so callers map them to typed rows."""
    n = len(curve)
    if n <= max_points:
        return list(range(n))
    step = n / max_points
    idx = sorted({min(n - 1, round(i * step)) for i in range(max_points)} | {n - 1})
    return idx


def build_artifact(
    candidate: StrategyCandidate,
    bars: Sequence[Candle],
    result: BacktestResult,
    *,
    segment: EvidenceSegment,
) -> BacktestArtifact:
    """A rich, dashboard-only artifact for ONE backtest window — the (downsampled)
    equity curve plus the headline annotations, all copied VERBATIM from the
    engine's ``BacktestResult``. Nothing here is recomputed downstream."""
    sb = result.starting_balance
    net = result.ending_equity - sb
    ret = (net / sb) if sb != 0 else Decimal("0")
    keep = _downsample(result.equity_curve)
    points = tuple(
        EquityCurvePoint(
            bar_index=result.equity_curve[i].bar_index,
            time=result.equity_curve[i].time,
            equity=result.equity_curve[i].equity,
            drawdown_pct=result.equity_curve[i].drawdown_pct,
        )
        for i in keep
    )
    return BacktestArtifact(
        config_hash=result.config_hash,
        candidate_id=candidate.candidate_id,
        pair=result.pair,
        segment=segment,
        window_start=bars[0].time,
        window_end=bars[-1].time,
        starting_balance=sb,
        ending_balance=result.ending_balance,
        ending_equity=result.ending_equity,
        peak_equity=result.peak_equity,
        net_pnl=net,
        return_pct=ret,
        max_drawdown_pct=result.max_drawdown_pct,
        trade_count=sum(1 for t in result.trade_log if t.realized_pnl is not None),
        cost_total=result.cost_breakdown.total,
        bars_processed=result.bars_processed,
        halted_due_to_kill_switch=result.halted_due_to_kill_switch,
        equity_curve=points,
    )


def run_candidate_artifacts(
    candidate: StrategyCandidate,
    *,
    candle_provider: CandleProvider,
    config: BacktestRunConfig,
) -> tuple[BacktestArtifact, ...]:
    """Build the in-sample and (token-gated) out-of-sample artifacts for ONE
    candidate — each with its full equity curve — for the dashboard.

    Like :func:`run_robustness`, this opens the sealed OOS slice in deterministic
    code only; the token never leaves this module. Raises :class:`HarnessError`
    if the candidate cannot be run at all."""
    candles = _fetch(candidate, candle_provider=candle_provider, config=config)
    split = DataSplit(candles, oos_fraction=config.oos_fraction)
    artifacts: list[BacktestArtifact] = []

    in_sample = split.in_sample
    if len(in_sample) >= 2:
        is_result = _run_engine(in_sample, candidate, config=config, cost_model=config.cost_model)
        artifacts.append(
            build_artifact(candidate, in_sample, is_result, segment=EvidenceSegment.IN_SAMPLE)
        )

    oos = split.access_out_of_sample(token=_OOS_CONFIRMATION)
    if len(oos) >= 2:
        oos_result = _run_engine(oos, candidate, config=config, cost_model=config.cost_model)
        artifacts.append(
            build_artifact(candidate, oos, oos_result, segment=EvidenceSegment.OUT_OF_SAMPLE)
        )

    if not artifacts:
        raise HarnessError("no window long enough to backtest for artifacts")
    return tuple(artifacts)


def run_candidate(
    candidate: StrategyCandidate,
    *,
    candle_provider: CandleProvider,
    config: BacktestRunConfig,
) -> BacktestEvidence:
    """Run ONE candidate's IN-SAMPLE backtest → :class:`BacktestEvidence`.

    The OOS tail is split off and sealed — this never opens it. Raises
    :class:`HarnessError` if the candidate cannot be run.
    """
    candles = _fetch(candidate, candle_provider=candle_provider, config=config)
    in_sample = DataSplit(candles, oos_fraction=config.oos_fraction).in_sample
    if len(in_sample) < 2:
        raise HarnessError("in-sample window too short to backtest")
    result = _run_engine(in_sample, candidate, config=config, cost_model=config.cost_model)
    return _evidence_from_result(
        candidate.candidate_id, in_sample, result, segment=EvidenceSegment.IN_SAMPLE, config=config
    )


def run_proposal(
    proposal: StrategyProposal,
    *,
    candle_provider: CandleProvider,
    config: BacktestRunConfig,
) -> tuple[tuple[BacktestEvidence, ...], tuple[tuple[str, str], ...]]:
    """Run every candidate IN-SAMPLE. Returns ``(evidence, skipped)`` where
    ``skipped`` is ``(candidate_id, reason)`` for candidates that could not run."""
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


# ---------------------------------------------------------------------------
# Robustness evidence (critic stage) — OOS, cost, parameter, concentration.
# All deterministic. The OOS path is the ONLY place the holdout token is used.
# ---------------------------------------------------------------------------


def _scale_cost(model: CostModel, multiplier: Decimal) -> CostModel:
    """Scale the friction components (spread / commission / slippage) for the
    cost-sensitivity stress; swap params are left as-is."""
    return CostModel(
        half_spread=model.half_spread * multiplier,
        commission_per_unit=model.commission_per_unit * multiplier,
        slippage_per_unit=model.slippage_per_unit * multiplier,
        swap_long_per_unit_per_day=model.swap_long_per_unit_per_day,
        swap_short_per_unit_per_day=model.swap_short_per_unit_per_day,
        weekend_swap_multiplier=model.weekend_swap_multiplier,
    )


def _neighbor(candidate: StrategyCandidate, param_name: str, value: Decimal) -> StrategyCandidate:
    new_params = tuple(
        p.model_copy(update={"value": value}) if p.name == param_name else p
        for p in candidate.parameters
    )
    return candidate.model_copy(update={"parameters": new_params})


def _trade_concentration(result: BacktestResult) -> TradeConcentration:
    """How much in-sample profit comes from the few biggest winners."""
    closed = [t for t in result.trade_log if t.realized_pnl is not None]
    wins = sorted(
        (t.realized_pnl for t in closed if t.realized_pnl is not None and t.realized_pnl > 0),
        reverse=True,
    )
    gross = sum(wins, Decimal("0"))
    if gross > 0:
        top1 = float(wins[0] / gross)
        top5 = float(sum(wins[:5], Decimal("0")) / gross)
    else:
        top1 = 0.0
        top5 = 0.0
    return TradeConcentration(
        closed_trade_count=len(closed),
        gross_profit=gross,
        top_trade_profit_share=top1,
        top5_profit_share=top5,
    )


def run_candidate_oos(
    candidate: StrategyCandidate,
    *,
    candle_provider: CandleProvider,
    config: BacktestRunConfig,
) -> BacktestEvidence:
    """Run a candidate on the sealed OUT-OF-SAMPLE slice.

    This is the ONLY function that opens the holdout: it supplies the literal
    confirmation token to :meth:`DataSplit.access_out_of_sample`. The token
    never leaves deterministic code; the LLM only ever sees the resulting
    evidence. Raises :class:`HarnessError` if OOS cannot be run.
    """
    candles = _fetch(candidate, candle_provider=candle_provider, config=config)
    split = DataSplit(candles, oos_fraction=config.oos_fraction)
    oos = split.access_out_of_sample(token=_OOS_CONFIRMATION)
    if len(oos) < 2:
        raise HarnessError("out-of-sample window too short to backtest")
    result = _run_engine(oos, candidate, config=config, cost_model=config.cost_model)
    return _evidence_from_result(
        candidate.candidate_id, oos, result, segment=EvidenceSegment.OUT_OF_SAMPLE, config=config
    )


def run_robustness(
    candidate: StrategyCandidate,
    *,
    candle_provider: CandleProvider,
    config: BacktestRunConfig,
) -> RobustnessEvidence:
    """Compute ALL deterministic robustness evidence for one candidate: the
    in-sample run, the token-gated OOS run, cost-sensitivity, parameter
    sensitivity, and trade-concentration. The critic interprets this; it never
    recomputes any of it. Raises :class:`HarnessError` if the candidate cannot
    be run at all."""
    candles = _fetch(candidate, candle_provider=candle_provider, config=config)
    split = DataSplit(candles, oos_fraction=config.oos_fraction)
    in_sample = split.in_sample
    if len(in_sample) < 2:
        raise HarnessError("in-sample window too short to backtest")

    base_result = _run_engine(in_sample, candidate, config=config, cost_model=config.cost_model)
    in_evidence = _evidence_from_result(
        candidate.candidate_id,
        in_sample,
        base_result,
        segment=EvidenceSegment.IN_SAMPLE,
        config=config,
    )

    # OOS — token-gated, deterministic, the only place the holdout opens.
    oos_evidence: BacktestEvidence | None = None
    oos = split.access_out_of_sample(token=_OOS_CONFIRMATION)
    if len(oos) >= 2:
        oos_result = _run_engine(oos, candidate, config=config, cost_model=config.cost_model)
        oos_evidence = _evidence_from_result(
            candidate.candidate_id,
            oos,
            oos_result,
            segment=EvidenceSegment.OUT_OF_SAMPLE,
            config=config,
        )

    # Cost sensitivity — re-run in-sample with scaled friction.
    cost_stress: list[CostStressPoint] = []
    for mult in config.cost_multipliers:
        stressed = _run_engine(
            in_sample, candidate, config=config, cost_model=_scale_cost(config.cost_model, mult)
        )
        cost_stress.append(
            CostStressPoint(
                cost_multiplier=mult,
                total_return_pct=stressed.metrics.total_return_pct,
                sharpe_ratio=stressed.metrics.sharpe_ratio,
                profit_factor=_finite_or_none(stressed.metrics.profit_factor),
                trade_count=stressed.metrics.trade_count,
            )
        )

    # Parameter sensitivity — perturb each param to its range endpoints.
    ranges = candidate.ranges_as_dict()
    neighbours: list[ParamNeighborResult] = []
    for name, rng in ranges.items():
        for value in (rng.low, rng.high):
            neighbours.append(
                _param_neighbor_result(candidate, name, value, in_sample=in_sample, config=config)
            )

    return RobustnessEvidence(
        candidate_id=candidate.candidate_id,
        in_sample=in_evidence,
        out_of_sample=oos_evidence,
        cost_stress=tuple(cost_stress),
        param_sensitivity=tuple(neighbours),
        trade_concentration=_trade_concentration(base_result),
    )


def _param_neighbor_result(
    candidate: StrategyCandidate,
    name: str,
    value: Decimal,
    *,
    in_sample: Sequence[Candle],
    config: BacktestRunConfig,
) -> ParamNeighborResult:
    neighbour = _neighbor(candidate, name, value)
    try:
        result = _run_engine(in_sample, neighbour, config=config, cost_model=config.cost_model)
    except HarnessError:
        return ParamNeighborResult(
            param_name=name,
            value=value,
            constructible=False,
            total_return_pct=None,
            sharpe_ratio=None,
        )
    return ParamNeighborResult(
        param_name=name,
        value=value,
        constructible=True,
        total_return_pct=result.metrics.total_return_pct,
        sharpe_ratio=result.metrics.sharpe_ratio,
    )


__all__ = [
    "DEFAULT_RISK_CONFIG",
    "BacktestRunConfig",
    "HarnessError",
    "build_artifact",
    "metrics_view",
    "run_candidate",
    "run_candidate_artifacts",
    "run_candidate_oos",
    "run_proposal",
    "run_robustness",
]
