"""End-to-end tests for the BacktestEngine.

These pin the engine's three promises:

1. No look-ahead — signals at bar t fill at t+1's open, never same-bar, and a
   strategy that physically reaches past bar t is stopped by the view.
2. Honest fills — costs drag returns; a costed run ends below a zero-cost run.
3. One risk gate — the SAME RiskManager the live loop uses resizes/rejects
   orders, and a drawdown breach trips the kill switch and flattens the run.

Plus result-level determinism: identical inputs => identical config_hash AND
identical BacktestResult.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal

from core.backtest import BacktestEngine, CostModel
from core.models import (
    AccountState,
    Candle,
    Direction,
    OrderRequest,
    RiskConfig,
)
from core.strategy import MovingAverageCrossover, PointInTimeView

# ---------------------------------------------------------------------------
# Test strategies (tiny, deterministic stubs implementing the Strategy protocol)
# ---------------------------------------------------------------------------


class OneShotStrategy:
    """Emits exactly one order the first time the visible window reaches
    ``trigger_len`` bars, then stays silent forever."""

    name = "one_shot"

    def __init__(
        self,
        *,
        pair: str = "EURUSD",
        direction: Direction = Direction.LONG,
        size: Decimal = Decimal("1000"),
        stop_distance: Decimal = Decimal("0.0050"),
        trigger_len: int = 2,
    ) -> None:
        self._pair = pair
        self._direction = direction
        self._size = size
        self._stop_distance = stop_distance
        self._trigger_len = trigger_len
        self._fired = False

    def decide(
        self,
        bars: PointInTimeView,
        account: AccountState,
        *,
        as_of: datetime,
    ) -> list[OrderRequest]:
        if self._fired or len(bars) != self._trigger_len:
            return []
        self._fired = True
        close = bars.latest.close
        stop = (
            close - self._stop_distance
            if self._direction is Direction.LONG
            else close + self._stop_distance
        )
        return [
            OrderRequest(
                pair=self._pair,
                direction=self._direction,
                size=self._size,
                entry_price=close,
                stop_price=stop,
            )
        ]


class PeekingStrategy:
    """A malicious strategy that tries to read the next (future) bar.

    The PointInTimeView only contains bars 0..t, so ``bars[len(bars)]`` raises
    LookAheadError. The engine should catch it and halt the run.
    """

    name = "peeking"

    def decide(
        self,
        bars: PointInTimeView,
        account: AccountState,
        *,
        as_of: datetime,
    ) -> list[OrderRequest]:
        _ = bars[len(bars)]  # look-ahead — boom
        return []


# ---------------------------------------------------------------------------
# 1. Look-ahead / next-bar-open fill
# ---------------------------------------------------------------------------


def test_signal_fills_at_next_bar_open_not_same_bar(
    make_bars: Callable[..., list[Candle]],
    roomy_risk_config: RiskConfig,
    zero_cost_model: CostModel,
) -> None:
    # Decision fires at index 2 (visible window length 3); fill is at bars[3].
    closes = ["1.10", "1.10", "1.10", "1.20", "1.20", "1.20"]
    opens = ["1.10", "1.10", "1.10", "1.20", "1.20", "1.20"]
    bars = make_bars(closes, opens=opens)

    strat = OneShotStrategy(trigger_len=3)
    result = BacktestEngine(
        bars=bars,
        strategy=strat,
        risk_config=roomy_risk_config,
        cost_model=zero_cost_model,
        starting_balance=Decimal("10000"),
    ).run()

    assert len(result.trade_log) == 1
    trade = result.trade_log[0]
    # Filled at the NEXT bar's open (1.20), never the decision bar's close (1.10).
    assert trade.entry_time == bars[3].time
    assert trade.entry_price == Decimal("1.20")
    assert trade.entry_price != bars[2].close
    # Single open position left at end → no exit recorded.
    assert trade.is_closed is False


def test_strategy_reaching_past_bar_t_halts_the_run(
    make_bars: Callable[..., list[Candle]],
    roomy_risk_config: RiskConfig,
    zero_cost_model: CostModel,
) -> None:
    bars = make_bars(["1.10", "1.11", "1.12", "1.13"])
    result = BacktestEngine(
        bars=bars,
        strategy=PeekingStrategy(),
        risk_config=roomy_risk_config,
        cost_model=zero_cost_model,
        starting_balance=Decimal("10000"),
    ).run()

    assert result.halted_due_to_kill_switch is True
    assert result.halt_reason is not None
    assert "LookAheadError" in result.halt_reason
    assert len(result.trade_log) == 0


# ---------------------------------------------------------------------------
# 2. Costs drag returns
# ---------------------------------------------------------------------------


def _oscillating_closes(n: int = 60) -> list[str]:
    closes: list[str] = []
    price = Decimal("1.10")
    for i in range(n):
        price = price + (Decimal("0.001") if (i // 5) % 2 == 0 else Decimal("-0.001"))
        closes.append(str(price))
    return closes


def test_costed_run_ends_below_zero_cost_run(
    make_bars: Callable[..., list[Candle]],
    roomy_risk_config: RiskConfig,
    zero_cost_model: CostModel,
    costed_model: CostModel,
) -> None:
    bars = make_bars(_oscillating_closes())

    def run(model: CostModel):
        strat = MovingAverageCrossover(
            pair="EURUSD",
            fast_period=3,
            slow_period=8,
            size=Decimal("1000"),
            stop_distance=Decimal("0.005"),
        )
        return BacktestEngine(
            bars=bars,
            strategy=strat,
            risk_config=roomy_risk_config,
            cost_model=model,
            starting_balance=Decimal("10000"),
        ).run()

    zero = run(zero_cost_model)
    costed = run(costed_model)

    assert zero.n_signals_accepted > 0
    assert zero.cost_breakdown.total == Decimal("0")
    assert costed.cost_breakdown.total > 0
    # Costs can only drag equity down relative to a frictionless run.
    assert costed.ending_equity < zero.ending_equity


# ---------------------------------------------------------------------------
# 3. The one risk gate — resize, reject, and the kill switch
# ---------------------------------------------------------------------------


def _config(**overrides) -> RiskConfig:
    base = dict(
        max_risk_per_trade_pct=Decimal("0.10"),
        max_portfolio_risk_pct=Decimal("0.50"),
        max_concurrent_positions=10,
        max_exposure_per_pair_pct=Decimal("100"),
        max_correlated_exposure_pct=Decimal("100"),
        correlation_groups={},
        daily_loss_limit_pct=Decimal("0.50"),
        max_drawdown_pct=Decimal("0.50"),
    )
    base.update(overrides)
    return RiskConfig(**base)


def test_oversized_order_is_resized_by_the_real_risk_manager(
    make_bars: Callable[..., list[Candle]],
    zero_cost_model: CostModel,
) -> None:
    # Per-trade cap 0.1% of 10000 = $10 of risk. Stop distance 0.005 => the
    # largest allowed size is 10 / 0.005 = 2000, well below the requested 100k.
    cfg = _config(max_risk_per_trade_pct=Decimal("0.001"))
    bars = make_bars(["1.10", "1.10", "1.10", "1.10"])
    strat = OneShotStrategy(size=Decimal("100000"), stop_distance=Decimal("0.005"), trigger_len=2)
    result = BacktestEngine(
        bars=bars,
        strategy=strat,
        risk_config=cfg,
        cost_model=zero_cost_model,
        starting_balance=Decimal("10000"),
    ).run()

    assert result.n_signals_accepted == 1
    assert len(result.trade_log) == 1
    # Resized down to exactly the per-trade-cap size, not the requested size.
    assert result.trade_log[0].size == Decimal("2000")
    assert result.trade_log[0].size < Decimal("100000")


def test_order_breaching_a_cap_is_rejected_by_the_real_risk_manager(
    make_bars: Callable[..., list[Candle]],
    zero_cost_model: CostModel,
) -> None:
    # Per-pair notional cap is a sliver ($1), so any real-sized order is rejected
    # — exactly as it would be live.
    cfg = _config(max_exposure_per_pair_pct=Decimal("0.0001"))
    bars = make_bars(["1.10", "1.10", "1.10", "1.10"])
    strat = OneShotStrategy(size=Decimal("1000"), trigger_len=2)
    result = BacktestEngine(
        bars=bars,
        strategy=strat,
        risk_config=cfg,
        cost_model=zero_cost_model,
        starting_balance=Decimal("10000"),
    ).run()

    assert result.n_signals_proposed == 1
    assert result.n_signals_accepted == 0
    assert result.n_signals_rejected == 1
    assert len(result.trade_log) == 0
    assert result.halted_due_to_kill_switch is False


def test_drawdown_breach_trips_kill_switch_and_flattens(
    make_bars: Callable[..., list[Candle]],
    zero_cost_model: CostModel,
) -> None:
    # Big long, then a sharp adverse move blows past the 5% drawdown cap.
    cfg = _config(
        max_drawdown_pct=Decimal("0.05"),
        max_risk_per_trade_pct=Decimal("1"),
        max_portfolio_risk_pct=Decimal("1"),
    )
    closes = ["1.10", "1.10", "1.10", "1.05", "1.05", "1.05"]
    bars = make_bars(closes, opens=closes)
    strat = OneShotStrategy(size=Decimal("100000"), stop_distance=Decimal("0.05"), trigger_len=2)
    result = BacktestEngine(
        bars=bars,
        strategy=strat,
        risk_config=cfg,
        cost_model=zero_cost_model,
        starting_balance=Decimal("10000"),
    ).run()

    assert result.halted_due_to_kill_switch is True
    assert result.halted_at_bar_index == 3
    assert result.halt_reason is not None
    assert "kill switch" in result.halt_reason
    assert "drawdown" in result.halt_reason
    # The run flattened on the way out: the trade is closed, none left open.
    assert len(result.trade_log) == 1
    assert result.trade_log[0].is_closed is True
    assert result.equity_curve[-1].open_positions == 0
    # Realized the modelled loss: (1.05 - 1.10) * 100000 = -5000 (zero cost).
    assert result.trade_log[0].realized_pnl == Decimal("-5000.00")


# ---------------------------------------------------------------------------
# 4. Determinism + reference strategy end-to-end
# ---------------------------------------------------------------------------


def _run_ma(bars, cfg, model):
    strat = MovingAverageCrossover(
        pair="EURUSD",
        fast_period=3,
        slow_period=8,
        size=Decimal("1000"),
        stop_distance=Decimal("0.005"),
    )
    return BacktestEngine(
        bars=bars,
        strategy=strat,
        risk_config=cfg,
        cost_model=model,
        starting_balance=Decimal("10000"),
    ).run()


def test_identical_inputs_give_identical_config_hash_and_result(
    make_bars: Callable[..., list[Candle]],
    roomy_risk_config: RiskConfig,
    costed_model: CostModel,
) -> None:
    bars = make_bars(_oscillating_closes())
    r1 = _run_ma(bars, roomy_risk_config, costed_model)
    r2 = _run_ma(bars, roomy_risk_config, costed_model)

    assert r1.config_hash == r2.config_hash
    # Full structural equality — trade ids included (no uuid4 anywhere).
    assert r1 == r2
    assert [t.trade_id for t in r1.trade_log] == [t.trade_id for t in r2.trade_log]
    if r1.trade_log:
        assert r1.trade_log[0].trade_id == "t000000"


def test_reference_strategy_runs_end_to_end_on_a_fixture(
    make_bars: Callable[..., list[Candle]],
    roomy_risk_config: RiskConfig,
    costed_model: CostModel,
) -> None:
    bars = make_bars(_oscillating_closes())
    result = _run_ma(bars, roomy_risk_config, costed_model)

    assert result.halted_due_to_kill_switch is False
    assert result.bars_processed == len(bars)
    assert result.n_signals_proposed > 0
    # One equity sample per bar (the final sample re-marks the last bar).
    assert len(result.equity_curve) == len(bars)
    assert result.pair == "EURUSD"
