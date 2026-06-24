"""Tests for ExecutionEngine.

These tests verify the *invariant*: an order reaches the broker only through
the risk-gated path. They cover approval, rejection, resize (with explicit
assertion that the resized size — not the original — is what hits the
broker), kill-switch refusal, and broker-side exception → kill-switch trip.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from core.execution import ExecutionEngine
from core.market_data import StubPriceProvider
from core.models import (
    AccountState,
    DecisionKind,
    Direction,
    Fill,
    OrderRequest,
    RejectionReason,
    RiskConfig,
)
from core.paper_broker import PaperBroker, PaperBrokerConfig
from core.risk_manager import RiskManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 6, 24, 12, 0, tzinfo=UTC)


@pytest.fixture
def prices() -> StubPriceProvider:
    pp = StubPriceProvider()
    pp.set_price("EURUSD", bid=Decimal("1.0848"), ask=Decimal("1.0850"))
    return pp


@pytest.fixture
def broker(prices: StubPriceProvider, now: datetime) -> PaperBroker:
    cfg = PaperBrokerConfig(
        starting_balance=Decimal("10000"),
        commission_per_unit=Decimal("0.0001"),
        extra_half_spread=Decimal("0.0001"),
    )
    return PaperBroker(cfg, prices, now_fn=lambda: now)


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(
        max_risk_per_trade_pct=Decimal("0.01"),
        max_portfolio_risk_pct=Decimal("0.05"),
        max_concurrent_positions=5,
        max_exposure_per_pair_pct=Decimal("30"),
        max_correlated_exposure_pct=Decimal("50"),
        daily_loss_limit_pct=Decimal("0.03"),
        max_drawdown_pct=Decimal("0.20"),
    )


@pytest.fixture
def engine(broker: PaperBroker, risk_config: RiskConfig) -> ExecutionEngine:
    return ExecutionEngine(broker=broker, risk_manager=RiskManager(risk_config))


def make_order(
    *,
    size: Decimal = Decimal("1000"),
    entry: Decimal = Decimal("1.0850"),
    stop: Decimal = Decimal("1.0800"),
) -> OrderRequest:
    return OrderRequest(
        pair="EURUSD",
        direction=Direction.LONG,
        size=size,
        entry_price=entry,
        stop_price=stop,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_approved_order_opens_position_and_charges_commission(
    engine: ExecutionEngine, broker: PaperBroker
) -> None:
    result = engine.submit(make_order(size=Decimal("1000")))
    assert result.placed
    assert result.decision.kind is DecisionKind.APPROVE
    assert result.fill is not None
    assert result.placed_order is not None
    assert result.placed_order.size == Decimal("1000")
    # Cash dropped by exactly the commission.
    assert broker.cash == Decimal("10000") - Decimal("0.10000000")
    # One position open at the buy-side fill price.
    positions = broker.get_open_positions()
    assert len(positions) == 1
    assert positions[0].entry_price == Decimal("1.0851")


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------


def test_rejected_order_places_nothing(
    broker: PaperBroker, risk_config: RiskConfig
) -> None:
    # Tighten portfolio cap so the order rejects.
    cfg = risk_config.model_copy(
        update={"max_portfolio_risk_pct": Decimal("0.00001")}
    )
    eng = ExecutionEngine(broker=broker, risk_manager=RiskManager(cfg))
    cash_before = broker.cash
    positions_before = list(broker.get_open_positions())

    result = eng.submit(make_order())

    assert result.placed is False
    assert result.fill is None
    assert result.decision.kind is DecisionKind.REJECT
    assert result.decision.limiting_rule is RejectionReason.PORTFOLIO_RISK_CAP
    # Account is byte-identical to before submission.
    assert broker.cash == cash_before
    assert broker.get_open_positions() == positions_before


# ---------------------------------------------------------------------------
# Resize — the critical assertion: the BROKER sees the resized size, not the original
# ---------------------------------------------------------------------------


def test_resize_sends_the_resized_order_to_broker(
    broker: PaperBroker, risk_config: RiskConfig
) -> None:
    """Per-trade cap shrinks the order. The size that the broker fills must
    be the RESIZED size (the whole point of resize as a path through to
    execution). Equity headroom: 10_000; max_risk_per_trade_pct=0.001 → cap 10.
    Original order: size 50_000 at stop_dist 0.005 → risk 250.
    Resized size: 10 / 0.005 = 2_000."""
    cfg = risk_config.model_copy(update={"max_risk_per_trade_pct": Decimal("0.001")})
    eng = ExecutionEngine(broker=broker, risk_manager=RiskManager(cfg))

    result = eng.submit(make_order(size=Decimal("50000")))

    assert result.decision.kind is DecisionKind.RESIZE
    assert result.placed_order is not None
    assert result.placed_order.size == Decimal("2000")
    assert result.fill is not None
    # Size on the fill matches the resized size, not 50_000.
    assert result.fill.size == Decimal("2000")
    # Exactly one new position, at the broker-side entry.
    positions = broker.get_open_positions()
    assert len(positions) == 1
    assert positions[0].size == Decimal("2000")


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


def test_kill_switch_refuses_before_any_broker_call(
    broker: PaperBroker, risk_config: RiskConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the kill switch is engaged, the engine must refuse WITHOUT calling
    the broker. We monkeypatch get_account_state to blow up so any sneaky
    call would surface as an exception."""
    rm = RiskManager(risk_config)
    rm.trip("test trip", tripped_by="manual")
    eng = ExecutionEngine(broker=broker, risk_manager=rm)

    def boom(*_a: Any, **_kw: Any) -> AccountState:  # pragma: no cover - asserted via test
        raise AssertionError("get_account_state must not be called when kill switch is engaged")

    monkeypatch.setattr(broker, "get_account_state", boom)

    def boom_place(*_a: Any, **_kw: Any) -> Fill:  # pragma: no cover
        raise AssertionError("place_order must not be called when kill switch is engaged")

    monkeypatch.setattr(broker, "place_order", boom_place)

    result = eng.submit(make_order())
    assert result.placed is False
    assert result.decision.kind is DecisionKind.REJECT
    assert result.decision.limiting_rule is RejectionReason.KILL_SWITCH


def test_kill_switch_tripped_mid_run_refuses_next_submit(
    engine: ExecutionEngine,
) -> None:
    engine.risk_manager.trip("operator intervention", tripped_by="manual")
    result = engine.submit(make_order())
    assert result.placed is False
    assert result.decision.limiting_rule is RejectionReason.KILL_SWITCH


# ---------------------------------------------------------------------------
# Broker exceptions trip the kill switch (fail-closed)
# ---------------------------------------------------------------------------


def test_get_account_state_exception_trips_kill_switch(
    broker: PaperBroker,
    risk_config: RiskConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = ExecutionEngine(broker=broker, risk_manager=RiskManager(risk_config))

    def boom() -> AccountState:
        raise RuntimeError("simulated broker outage")

    monkeypatch.setattr(broker, "get_account_state", boom)

    result = eng.submit(make_order())
    assert result.placed is False
    assert result.decision.limiting_rule is RejectionReason.KILL_SWITCH
    assert eng.risk_manager.kill_switch_engaged
    assert eng.risk_manager.kill_switch_state.tripped_by == "exception"


def test_place_order_exception_trips_kill_switch(
    broker: PaperBroker,
    risk_config: RiskConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = ExecutionEngine(broker=broker, risk_manager=RiskManager(risk_config))

    def boom(order: OrderRequest) -> Fill:
        raise RuntimeError("simulated venue rejection")

    monkeypatch.setattr(broker, "place_order", boom)

    result = eng.submit(make_order())
    assert result.placed is False
    assert eng.risk_manager.kill_switch_engaged
    assert eng.risk_manager.kill_switch_state.tripped_by == "exception"
    # The engine did try to place the order — the placed_order field is set
    # even though no fill came back, so the audit shows what was attempted.
    assert result.placed_order is not None
    assert result.fill is None


# ---------------------------------------------------------------------------
# End-to-end equity / peak / drawdown propagation
# ---------------------------------------------------------------------------


def test_round_trip_updates_account_state_for_next_decision(
    engine: ExecutionEngine,
    broker: PaperBroker,
    prices: StubPriceProvider,
    now: datetime,
) -> None:
    # Open
    first = engine.submit(make_order(size=Decimal("10000")))
    assert first.placed

    # Move price up; query account state — peak should have advanced.
    prices.set_price(
        "EURUSD", bid=Decimal("1.0898"), ask=Decimal("1.0900"), timestamp=now
    )
    mid_state = broker.get_account_state()
    assert mid_state.equity > Decimal("10000")
    assert mid_state.peak_equity == mid_state.equity

    # Move price back down (small drawdown).
    prices.set_price(
        "EURUSD", bid=Decimal("1.0860"), ask=Decimal("1.0862"), timestamp=now
    )
    later_state = broker.get_account_state()
    assert later_state.peak_equity > later_state.equity
    assert later_state.drawdown_pct > 0
