"""Tests for the deterministic net-of-cost economics helper.

Covers expectancy (net = gross - cost/trade), cost-to-edge, the IS->OOS decay
fraction, and every retire/concern flag threshold; determinism; and the hard
rule that this helper never recomputes via the backtest engine.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from core.analysis.economics import (
    EconomicFlag,
    EconomicsThresholds,
    candidate_economics,
)
from core.models import (
    BacktestEvidence,
    BacktestMetricsView,
    EvidenceSegment,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _ev(
    segment: EvidenceSegment,
    *,
    start: str,
    end: str,
    cost: str,
    avg: str,
    wr: float,
    pf: float | None,
    trades: int,
) -> BacktestEvidence:
    metrics = BacktestMetricsView(
        total_return_pct=Decimal("0"),
        annualised_return_pct=Decimal("0"),
        sharpe_ratio=1.0,
        sortino_ratio=1.0,
        max_drawdown_pct=Decimal("0.05"),
        win_rate=wr,
        profit_factor=pf,
        trade_count=trades,
        avg_trade_pnl=Decimal(avg),
        exposure_pct=0.3,
    )
    return BacktestEvidence(
        candidate_id="c1",
        config_hash="hash-" + segment.value,
        pair="USDJPY",
        segment=segment,
        window_start=BASE,
        window_end=BASE + timedelta(days=30),
        bars_total=500,
        bars_processed=500,
        halted_due_to_kill_switch=False,
        halt_reason=None,
        n_signals_proposed=10,
        n_signals_accepted=8,
        n_signals_rejected=2,
        starting_balance=Decimal(start),
        ending_equity=Decimal(end),
        cost_total=Decimal(cost),
        metrics=metrics,
        gates=(),
        gates_all_passed=True,
    )


# ---------------------------------------------------------------------------
# Core arithmetic
# ---------------------------------------------------------------------------


def test_net_expectancy_is_gross_minus_cost_per_trade() -> None:
    ev = _ev(
        EvidenceSegment.IN_SAMPLE,
        start="100000",
        end="110000",
        cost="100",
        avg="50",
        wr=0.5,
        pf=2.0,
        trades=200,
    )
    ec = candidate_economics(ev)
    w = ec.in_sample
    assert w.gross_expectancy_per_trade == Decimal("50")
    assert w.cost_per_trade == Decimal("100") / Decimal("200")  # 0.5
    assert w.net_expectancy_per_trade == Decimal("50") - Decimal("0.5")  # 49.5


def test_cost_to_edge_is_cost_over_gross_pnl() -> None:
    # net 10000, cost 100 -> gross 10100 -> 100/10100 ≈ 0.0099
    ev = _ev(
        EvidenceSegment.IN_SAMPLE,
        start="100000",
        end="110000",
        cost="100",
        avg="50",
        wr=0.5,
        pf=2.0,
        trades=200,
    )
    w = candidate_economics(ev).in_sample
    assert w.cost_to_edge is not None
    assert abs(w.cost_to_edge - 100 / 10100) < 1e-9
    assert "Broker takes" in w.cost_to_edge_label


def test_avg_win_loss_reconstructs_gross_expectancy() -> None:
    w = candidate_economics(
        _ev(
            EvidenceSegment.IN_SAMPLE,
            start="100000",
            end="110000",
            cost="100",
            avg="50",
            wr=0.5,
            pf=2.0,
            trades=200,
        )
    ).in_sample
    assert w.avg_win is not None and w.avg_loss is not None
    # win_rate*avg_win - loss_rate*avg_loss == gross expectancy
    recon = Decimal("0.5") * w.avg_win - Decimal("0.5") * w.avg_loss
    assert abs(recon - w.gross_expectancy_per_trade) < Decimal("0.01")


def test_win_rate_always_accompanied_by_avg_win_loss_or_expectancy() -> None:
    # The model surfaces win_rate together with the per-trade edge fields; the UI
    # relies on this to never show win rate alone.
    w = candidate_economics(
        _ev(
            EvidenceSegment.IN_SAMPLE,
            start="100000",
            end="110000",
            cost="100",
            avg="50",
            wr=0.5,
            pf=2.0,
            trades=200,
        )
    ).in_sample
    assert hasattr(w, "win_rate")
    assert w.gross_expectancy_per_trade is not None


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------


def test_decay_fraction_is_oos_over_is_expectancy() -> None:
    is_ev = _ev(
        EvidenceSegment.IN_SAMPLE,
        start="100000",
        end="120000",
        cost="100",
        avg="50.5",
        wr=0.5,
        pf=2.0,
        trades=200,
    )
    oos_ev = _ev(
        EvidenceSegment.OUT_OF_SAMPLE,
        start="100000",
        end="110000",
        cost="50",
        avg="25.5",
        wr=0.5,
        pf=1.8,
        trades=100,
    )
    ec = candidate_economics(is_ev, oos_ev)
    assert ec.decay is not None
    # IS net_exp = 50.5 - 0.5 = 50 ; OOS net_exp = 25.5 - 0.5 = 25 ; frac = 0.5
    assert ec.decay.oos_expectancy_fraction_of_is is not None
    assert abs(ec.decay.oos_expectancy_fraction_of_is - 0.5) < 1e-9
    assert "NOT live" in ec.decay.note


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def test_flag_ok_when_healthy() -> None:
    is_ev = _ev(
        EvidenceSegment.IN_SAMPLE,
        start="100000",
        end="120000",
        cost="100",
        avg="50",
        wr=0.5,
        pf=2.0,
        trades=200,
    )
    oos_ev = _ev(
        EvidenceSegment.OUT_OF_SAMPLE,
        start="100000",
        end="110000",
        cost="50",
        avg="45",
        wr=0.5,
        pf=1.9,
        trades=60,
    )
    ec = candidate_economics(is_ev, oos_ev)
    assert ec.flag == EconomicFlag.OK
    assert ec.concerns == ()


def test_flag_retire_on_negative_net_expectancy() -> None:
    # OOS gross 5, cost/trade 6 -> net expectancy -1
    is_ev = _ev(
        EvidenceSegment.IN_SAMPLE,
        start="100000",
        end="110000",
        cost="100",
        avg="50",
        wr=0.5,
        pf=2.0,
        trades=200,
    )
    oos_ev = _ev(
        EvidenceSegment.OUT_OF_SAMPLE,
        start="100000",
        end="101000",
        cost="300",
        avg="5",
        wr=0.5,
        pf=1.1,
        trades=50,
    )
    ec = candidate_economics(is_ev, oos_ev)
    assert ec.flag == EconomicFlag.RETIRE
    assert any("Net expectancy negative" in c.reason for c in ec.concerns)


def test_flag_concern_on_high_cost_to_edge() -> None:
    # IS-only: net 200, cost 300 -> gross 500 -> cost-to-edge 0.6 (concern), net exp positive
    ev = _ev(
        EvidenceSegment.IN_SAMPLE,
        start="100000",
        end="100200",
        cost="300",
        avg="10",
        wr=0.5,
        pf=2.0,
        trades=100,
    )
    ec = candidate_economics(ev)
    assert ec.flag == EconomicFlag.CONCERN
    assert any("Broker takes 60" in c.reason for c in ec.concerns)


def test_flag_retire_on_oos_expectancy_collapse() -> None:
    # IS net_exp 50, OOS net_exp 5 -> frac 0.1 < 0.25 (retire); both positive, plenty of trades
    is_ev = _ev(
        EvidenceSegment.IN_SAMPLE,
        start="100000",
        end="120000",
        cost="100",
        avg="50.5",
        wr=0.5,
        pf=2.0,
        trades=200,
    )
    oos_ev = _ev(
        EvidenceSegment.OUT_OF_SAMPLE,
        start="100000",
        end="105000",
        cost="50",
        avg="5.5",
        wr=0.5,
        pf=1.5,
        trades=100,
    )
    ec = candidate_economics(is_ev, oos_ev)
    assert ec.flag == EconomicFlag.RETIRE
    assert any("% of in-sample" in c.reason for c in ec.concerns)


def test_flag_concern_on_thin_oos_trade_count() -> None:
    is_ev = _ev(
        EvidenceSegment.IN_SAMPLE,
        start="100000",
        end="120000",
        cost="100",
        avg="50",
        wr=0.5,
        pf=2.0,
        trades=200,
    )
    oos_ev = _ev(
        EvidenceSegment.OUT_OF_SAMPLE,
        start="100000",
        end="110000",
        cost="50",
        avg="45",
        wr=0.5,
        pf=1.9,
        trades=5,
    )
    ec = candidate_economics(is_ev, oos_ev)
    assert ec.flag == EconomicFlag.CONCERN
    assert any("statistical-power floor" in c.reason for c in ec.concerns)


def test_thresholds_are_configurable() -> None:
    ev = _ev(
        EvidenceSegment.IN_SAMPLE,
        start="100000",
        end="100200",
        cost="300",
        avg="10",
        wr=0.5,
        pf=2.0,
        trades=100,
    )
    # raise the concern ceiling above 0.6 -> no longer flagged
    lenient = EconomicsThresholds(cost_to_edge_concern=Decimal("0.9"))
    assert candidate_economics(ev, thresholds=lenient).flag == EconomicFlag.OK


# ---------------------------------------------------------------------------
# Determinism + no-engine guarantee
# ---------------------------------------------------------------------------


def test_deterministic() -> None:
    is_ev = _ev(
        EvidenceSegment.IN_SAMPLE,
        start="100000",
        end="120000",
        cost="100",
        avg="50",
        wr=0.5,
        pf=2.0,
        trades=200,
    )
    oos_ev = _ev(
        EvidenceSegment.OUT_OF_SAMPLE,
        start="100000",
        end="110000",
        cost="50",
        avg="45",
        wr=0.5,
        pf=1.9,
        trades=60,
    )
    assert candidate_economics(is_ev, oos_ev) == candidate_economics(is_ev, oos_ev)


def test_helper_never_recomputes_via_the_engine() -> None:
    src = (Path(__file__).resolve().parents[1] / "core" / "analysis" / "economics.py").read_text()
    import_lines = [
        line for line in src.splitlines() if line.strip().startswith(("import ", "from "))
    ]
    assert import_lines  # sanity: we actually read the module
    # The helper must not import the engine (it reads persisted values only).
    assert not any("core.backtest" in line for line in import_lines)
    assert not any("BacktestEngine" in line for line in import_lines)
    # And at runtime the module exposes no engine symbol.
    import core.analysis.economics as econ

    assert not hasattr(econ, "BacktestEngine")
