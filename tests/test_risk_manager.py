"""Unit tests for the deterministic risk manager.

These tests are the safety keystone of the fast loop. Coverage targets:

- sizing math (incl. resize-down when per-trade cap exceeded)
- each cap rejecting independently
- kill-switch latch + every tripping path + reset
- edge cases (zero/negative equity, zero stop distance, drawdown right at cap)
- ``RiskConfig`` immutability

The test fixtures intentionally use round numbers and small magnitudes so the
arithmetic stays readable in the assertions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from core.models import (
    AccountState,
    DecisionKind,
    Direction,
    OrderRequest,
    Position,
    RejectionReason,
    RiskConfig,
)
from core.risk_manager import RiskManager, _hash_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 6, 24, 12, 0, tzinfo=UTC)


@pytest.fixture
def cfg() -> RiskConfig:
    """A spacious-but-realistic default config.

    Numbers chosen so a 'normal' order at 10k equity easily fits, leaving plenty
    of headroom for tests that deliberately violate one cap at a time.
    """
    # Exposure caps modelled on a 30:1 leverage allowance — realistic for
    # retail forex and roomy enough that the sizing-focused tests aren't
    # confounded by accidental per-pair cap hits.
    return RiskConfig(
        max_risk_per_trade_pct=Decimal("0.01"),         # 1% per trade
        max_portfolio_risk_pct=Decimal("0.05"),         # 5% total open risk
        max_concurrent_positions=5,
        max_exposure_per_pair_pct=Decimal("30"),        # 30x notional per pair
        max_correlated_exposure_pct=Decimal("50"),      # 50x across a group
        correlation_groups={
            "USD_QUOTE": ("EURUSD", "GBPUSD", "AUDUSD"),
            "USD_BASE": ("USDJPY", "USDCAD"),
        },
        daily_loss_limit_pct=Decimal("0.03"),           # 3% daily
        max_drawdown_pct=Decimal("0.20"),               # 20% max DD
    )


@pytest.fixture
def account(now: datetime) -> AccountState:
    return AccountState(
        balance=Decimal("10000"),
        equity=Decimal("10000"),
        peak_equity=Decimal("10000"),
        day_start_equity=Decimal("10000"),
        as_of=now,
    )


def make_order(
    *,
    pair: str = "EURUSD",
    direction: Direction = Direction.LONG,
    size: Decimal = Decimal("1000"),
    entry: Decimal = Decimal("1.0850"),
    stop: Decimal = Decimal("1.0800"),
) -> OrderRequest:
    return OrderRequest(
        pair=pair,
        direction=direction,
        size=size,
        entry_price=entry,
        stop_price=stop,
    )


def make_position(
    *,
    pair: str = "EURUSD",
    direction: Direction = Direction.LONG,
    size: Decimal = Decimal("1000"),
    entry: Decimal = Decimal("1.0850"),
    stop: Decimal = Decimal("1.0800"),
) -> Position:
    return Position(
        pair=pair,
        direction=direction,
        size=size,
        entry_price=entry,
        stop_price=stop,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_approves_normal_order(cfg: RiskConfig, account: AccountState) -> None:
    rm = RiskManager(cfg)
    order = make_order()  # risk_at_stop = 1000 * 0.005 = 5; cap = 0.01 * 10_000 = 100
    decision = rm.evaluate(order, account)

    assert decision.kind == DecisionKind.APPROVE
    assert decision.accepted is True
    assert decision.sized_order == order
    assert decision.rejected_by == ()
    assert decision.limiting_rule is None
    assert decision.config_hash == rm.config_hash
    assert decision.as_of == account.as_of


# ---------------------------------------------------------------------------
# Sizing math + resize-down
# ---------------------------------------------------------------------------


def test_per_trade_cap_resizes_down(cfg: RiskConfig, account: AccountState) -> None:
    # stop distance 0.005; equity 10_000; cap 1% → max risk 100
    # request size 50_000 → risk = 250, well above 100. Expect resize to 20_000.
    order = make_order(size=Decimal("50000"))
    rm = RiskManager(cfg)
    decision = rm.evaluate(order, account)

    assert decision.kind == DecisionKind.RESIZE
    assert decision.accepted is True
    assert decision.limiting_rule is RejectionReason.PER_TRADE_CAP
    assert decision.sized_order is not None
    # max_risk / stop_distance = 100 / 0.005 = 20_000
    assert decision.sized_order.size == Decimal("20000")
    # The resized order's risk should equal the cap exactly.
    assert decision.sized_order.risk_at_stop == Decimal("100.000")


def test_sizing_exact_boundary_does_not_resize(
    cfg: RiskConfig, account: AccountState
) -> None:
    # Risk lands EXACTLY at the per-trade cap → should approve, not resize.
    # cap=100, stop_dist=0.005 → size that risks exactly 100 = 20_000
    order = make_order(size=Decimal("20000"))
    rm = RiskManager(cfg)
    decision = rm.evaluate(order, account)
    assert decision.kind == DecisionKind.APPROVE
    assert decision.sized_order is not None
    assert decision.sized_order.size == Decimal("20000")


# ---------------------------------------------------------------------------
# Each cap independently
# ---------------------------------------------------------------------------


def test_rejects_when_max_concurrent_positions_reached(
    cfg: RiskConfig, account: AccountState
) -> None:
    positions = tuple(
        make_position(pair="EURUSD", size=Decimal("100"))
        for _ in range(cfg.max_concurrent_positions)
    )
    account = account.model_copy(update={"open_positions": positions})
    rm = RiskManager(cfg)
    decision = rm.evaluate(make_order(), account)
    assert decision.kind == DecisionKind.REJECT
    assert decision.limiting_rule is RejectionReason.MAX_CONCURRENT_POSITIONS
    assert RejectionReason.MAX_CONCURRENT_POSITIONS in decision.rejected_by


def test_rejects_when_per_pair_exposure_cap_exceeded(
    cfg: RiskConfig, account: AccountState, now: datetime
) -> None:
    # Tighten per-pair cap to 5% notional → 500 USD at equity 10_000.
    cfg = cfg.model_copy(update={"max_exposure_per_pair_pct": Decimal("0.05")})
    # An order with notional 1000*1.0850 = 1085 > 500. Resize to risk-cap first
    # would not apply (risk is 5), so per-pair fires.
    order = make_order(size=Decimal("1000"))
    rm = RiskManager(cfg)
    decision = rm.evaluate(order, account)
    assert decision.kind == DecisionKind.REJECT
    assert decision.limiting_rule is RejectionReason.PER_PAIR_EXPOSURE_CAP


def test_rejects_when_correlated_exposure_cap_exceeded(
    cfg: RiskConfig, account: AccountState
) -> None:
    # Correlated cap to 5% notional → 500 USD at equity 10_000.
    cfg = cfg.model_copy(update={"max_correlated_exposure_pct": Decimal("0.05")})
    # Open a position on GBPUSD (in the USD_QUOTE group). Notional ~ 300.
    pos = make_position(
        pair="GBPUSD",
        size=Decimal("200"),
        entry=Decimal("1.2700"),  # notional 254
        stop=Decimal("1.2650"),
    )
    account = account.model_copy(update={"open_positions": (pos,)})
    rm = RiskManager(cfg)
    # New order on EURUSD also in USD_QUOTE — combined notional 254 + 1085 > 500.
    decision = rm.evaluate(make_order(size=Decimal("1000")), account)
    assert decision.kind == DecisionKind.REJECT
    assert decision.limiting_rule is RejectionReason.CORRELATED_EXPOSURE_CAP


def test_rejects_when_portfolio_risk_cap_exceeded(
    cfg: RiskConfig, account: AccountState
) -> None:
    # Tighten portfolio cap to 0.001% → 0.1 USD; basically zero headroom.
    cfg = cfg.model_copy(update={"max_portfolio_risk_pct": Decimal("0.00001")})
    rm = RiskManager(cfg)
    # Small order with risk_at_stop = 5 → far above the tiny cap.
    decision = rm.evaluate(make_order(), account)
    assert decision.kind == DecisionKind.REJECT
    assert decision.limiting_rule is RejectionReason.PORTFOLIO_RISK_CAP


def test_rejects_and_trips_on_daily_loss_breach(
    cfg: RiskConfig, account: AccountState
) -> None:
    # daily_loss_limit_pct = 3% of day_start_equity (10_000) = 300 loss
    # Set equity such that loss = 350.
    account = account.model_copy(update={"equity": Decimal("9650")})
    rm = RiskManager(cfg)
    decision = rm.evaluate(make_order(), account)
    assert decision.kind == DecisionKind.REJECT
    assert RejectionReason.DAILY_LOSS_LIMIT in decision.rejected_by
    assert RejectionReason.KILL_SWITCH in decision.rejected_by
    assert rm.kill_switch_engaged
    assert rm.kill_switch_state.tripped_by == "daily_loss"


def test_rejects_and_trips_on_drawdown_breach(
    cfg: RiskConfig, account: AccountState
) -> None:
    # max_drawdown_pct = 20%; set equity 7000 against peak 10_000 → DD = 30%.
    account = account.model_copy(
        update={
            "equity": Decimal("7000"),
            # Pull day_start down so we breach DD without first tripping daily-loss.
            "day_start_equity": Decimal("7000"),
        }
    )
    rm = RiskManager(cfg)
    decision = rm.evaluate(make_order(), account)
    assert decision.kind == DecisionKind.REJECT
    assert RejectionReason.DRAWDOWN_CAP in decision.rejected_by
    assert RejectionReason.KILL_SWITCH in decision.rejected_by
    assert rm.kill_switch_engaged
    assert rm.kill_switch_state.tripped_by == "drawdown"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_rejects_when_stop_distance_zero(
    cfg: RiskConfig, account: AccountState
) -> None:
    order = make_order(entry=Decimal("1.0850"), stop=Decimal("1.0850"))
    rm = RiskManager(cfg)
    decision = rm.evaluate(order, account)
    assert decision.kind == DecisionKind.REJECT
    assert decision.limiting_rule is RejectionReason.STOP_DISTANCE_NONPOSITIVE


def test_rejects_when_equity_nonpositive(
    cfg: RiskConfig, account: AccountState
) -> None:
    account = account.model_copy(update={"equity": Decimal("0")})
    rm = RiskManager(cfg)
    decision = rm.evaluate(make_order(), account)
    assert decision.kind == DecisionKind.REJECT
    assert decision.limiting_rule is RejectionReason.NONPOSITIVE_EQUITY


def test_drawdown_right_at_cap_trips(
    cfg: RiskConfig, account: AccountState
) -> None:
    # Equity at 8000, peak 10_000 → DD exactly 20% = cap. ``>=`` means it trips.
    account = account.model_copy(
        update={"equity": Decimal("8000"), "day_start_equity": Decimal("8000")}
    )
    rm = RiskManager(cfg)
    decision = rm.evaluate(make_order(), account)
    assert decision.kind == DecisionKind.REJECT
    assert rm.kill_switch_engaged


def test_order_below_drawdown_cap_passes_drawdown_gate(
    cfg: RiskConfig, account: AccountState
) -> None:
    # DD 10% < cap 20%, so the gate must pass and the order go through.
    account = account.model_copy(
        update={"equity": Decimal("9000"), "day_start_equity": Decimal("9000")}
    )
    rm = RiskManager(cfg)
    decision = rm.evaluate(make_order(), account)
    assert decision.accepted is True
    assert not rm.kill_switch_engaged


def test_negative_equity_short_circuits_before_caps(
    cfg: RiskConfig, account: AccountState
) -> None:
    account = account.model_copy(update={"equity": Decimal("-100")})
    rm = RiskManager(cfg)
    decision = rm.evaluate(make_order(), account)
    assert decision.kind == DecisionKind.REJECT
    assert decision.limiting_rule is RejectionReason.NONPOSITIVE_EQUITY


# ---------------------------------------------------------------------------
# Kill switch — latching, tripping paths, reset
# ---------------------------------------------------------------------------


def test_kill_switch_blocks_all_orders_when_engaged(
    cfg: RiskConfig, account: AccountState
) -> None:
    rm = RiskManager(cfg)
    rm.trip("anomaly detected", tripped_by="manual")
    for _ in range(3):
        decision = rm.evaluate(make_order(), account)
        assert decision.kind == DecisionKind.REJECT
        assert decision.limiting_rule is RejectionReason.KILL_SWITCH


def test_kill_switch_latches_after_drawdown_trip(
    cfg: RiskConfig, account: AccountState, now: datetime
) -> None:
    # Trigger via drawdown.
    breached = account.model_copy(
        update={"equity": Decimal("7000"), "day_start_equity": Decimal("7000")}
    )
    rm = RiskManager(cfg)
    rm.evaluate(make_order(), breached)
    assert rm.kill_switch_engaged

    # Even with a now-healthy account, the switch stays engaged: it LATCHES.
    healthy = AccountState(
        balance=Decimal("12000"),
        equity=Decimal("12000"),
        peak_equity=Decimal("12000"),
        day_start_equity=Decimal("12000"),
        as_of=now,
    )
    decision = rm.evaluate(make_order(), healthy)
    assert decision.kind == DecisionKind.REJECT
    assert decision.limiting_rule is RejectionReason.KILL_SWITCH


def test_manual_trip_records_reason_and_source(cfg: RiskConfig) -> None:
    rm = RiskManager(cfg)
    rm.trip("connectivity loss", tripped_by="execution_health_check")
    state = rm.kill_switch_state
    assert state.engaged is True
    assert state.reason == "connectivity loss"
    assert state.tripped_by == "execution_health_check"
    assert state.tripped_at is not None


def test_re_tripping_preserves_original_cause(cfg: RiskConfig) -> None:
    rm = RiskManager(cfg)
    rm.trip("first cause", tripped_by="manual")
    rm.trip("second cause", tripped_by="other")
    # The first cause is the root in the audit; second is logged but not stored.
    state = rm.kill_switch_state
    assert state.reason == "first cause"
    assert state.tripped_by == "manual"


def test_reset_requires_confirmation_token(cfg: RiskConfig) -> None:
    rm = RiskManager(cfg)
    rm.trip("test", tripped_by="manual")
    with pytest.raises(PermissionError):
        rm.reset(operator="alice", confirmation="please")
    # Still engaged after the failed reset.
    assert rm.kill_switch_engaged


def test_reset_with_correct_confirmation_clears(
    cfg: RiskConfig, account: AccountState
) -> None:
    rm = RiskManager(cfg)
    rm.trip("test", tripped_by="manual")
    rm.reset(operator="alice", confirmation="I_UNDERSTAND_RESET")
    assert not rm.kill_switch_engaged

    # Normal orders flow again.
    decision = rm.evaluate(make_order(), account)
    assert decision.accepted is True


def test_reset_when_not_engaged_is_a_no_op(cfg: RiskConfig) -> None:
    rm = RiskManager(cfg)
    rm.reset(operator="alice", confirmation="I_UNDERSTAND_RESET")
    assert not rm.kill_switch_engaged


# ---------------------------------------------------------------------------
# RiskConfig immutability
# ---------------------------------------------------------------------------


def test_risk_config_is_frozen(cfg: RiskConfig) -> None:
    with pytest.raises(ValidationError):
        cfg.max_risk_per_trade_pct = Decimal("0.99")  # type: ignore[misc]
    with pytest.raises(ValidationError):
        cfg.max_concurrent_positions = 999  # type: ignore[misc]


def test_risk_config_propose_new_via_model_copy(cfg: RiskConfig) -> None:
    # The supported pattern: build a *new* config; the live one is untouched.
    proposed = cfg.model_copy(update={"max_risk_per_trade_pct": Decimal("0.005")})
    assert proposed.max_risk_per_trade_pct == Decimal("0.005")
    assert cfg.max_risk_per_trade_pct == Decimal("0.01")
    assert _hash_config(proposed) != _hash_config(cfg)


def test_account_state_is_frozen(account: AccountState) -> None:
    with pytest.raises(ValidationError):
        account.equity = Decimal("99999")  # type: ignore[misc]


def test_order_request_is_frozen() -> None:
    o = make_order()
    with pytest.raises(ValidationError):
        o.size = Decimal("99999")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Resize-flows-through-downstream-gates (classic risk-engine bug)
# ---------------------------------------------------------------------------


def test_resize_is_evaluated_against_downstream_caps(
    cfg: RiskConfig, account: AccountState
) -> None:
    """When step 6 resizes the order, steps 7-9 must use the RESIZED notional
    and risk_at_stop — not the original. Otherwise an order would be rejected
    by a downstream cap that the resized version comfortably satisfies.

    Setup:
      - equity = 10_000
      - per-trade cap 0.1% → max risk 10
      - per-pair cap 50% notional → 5_000
      - request: size 100_000, entry 1.0, stop 0.95 → stop_distance 0.05
        * original risk 100_000 * 0.05 = 5_000  (≫ 10, triggers resize)
        * resized size 10 / 0.05 = 200
        * resized notional 200 * 1.0 = 200      (well under per-pair 5_000)

    If the implementation checked the ORIGINAL notional (100_000) against the
    per-pair cap (5_000), this would REJECT. If it correctly uses the resized
    notional (200), it APPROVES the resized order.
    """
    cfg = cfg.model_copy(
        update={
            "max_risk_per_trade_pct": Decimal("0.001"),
            "max_exposure_per_pair_pct": Decimal("0.5"),
        }
    )
    order = make_order(
        size=Decimal("100000"),
        entry=Decimal("1.0"),
        stop=Decimal("0.95"),
    )
    rm = RiskManager(cfg)
    decision = rm.evaluate(order, account)

    assert decision.kind == DecisionKind.RESIZE, (
        f"expected RESIZE (downstream gates seeing the resized order); got "
        f"{decision.kind} reason={decision.reason!r}"
    )
    assert decision.sized_order is not None
    assert decision.sized_order.size == Decimal("200")
    assert decision.sized_order.notional == Decimal("200.00")
    # The whole point of this test: the size hitting downstream caps is the
    # resized one, not the originally-requested 100_000.
    assert decision.sized_order.size < order.size


def test_resize_then_correlated_cap_uses_resized_notional(
    cfg: RiskConfig, account: AccountState
) -> None:
    """Same property for the correlated-group gate."""
    # Tight correlated cap; existing exposure brings the group close to it so
    # only the small RESIZED notional can fit.
    cfg = cfg.model_copy(
        update={
            "max_risk_per_trade_pct": Decimal("0.001"),
            "max_correlated_exposure_pct": Decimal("0.5"),
        }
    )
    existing = make_position(
        pair="GBPUSD",
        size=Decimal("4500"),
        entry=Decimal("1.0"),  # notional 4500; under the 5000 cap with room for 500
        stop=Decimal("0.999"),
    )
    account = account.model_copy(update={"open_positions": (existing,)})
    order = make_order(
        size=Decimal("100000"),
        entry=Decimal("1.0"),
        stop=Decimal("0.95"),
    )
    rm = RiskManager(cfg)
    decision = rm.evaluate(order, account)
    # original notional 100_000 + existing 4500 ≫ 5000 (would reject)
    # resized notional 200 + existing 4500 = 4700 ≤ 5000 (must approve)
    assert decision.kind == DecisionKind.RESIZE
    assert decision.sized_order is not None
    assert decision.sized_order.size == Decimal("200")


def test_aggregate_caps_reject_not_resize(
    cfg: RiskConfig, account: AccountState
) -> None:
    """Spec contract: per-trade is the only cap that resizes. Portfolio,
    per-pair, and correlated reject — they do not silently downsize."""
    cfg = cfg.model_copy(update={"max_portfolio_risk_pct": Decimal("0.00001")})
    # Below the per-trade cap (5 ≪ 100), so per-trade does not fire — yet
    # portfolio is so tight that it must reject, not resize.
    rm = RiskManager(cfg)
    decision = rm.evaluate(make_order(), account)
    assert decision.kind == DecisionKind.REJECT
    assert decision.limiting_rule is RejectionReason.PORTFOLIO_RISK_CAP
    assert decision.sized_order is None


# ---------------------------------------------------------------------------
# Hash / determinism
# ---------------------------------------------------------------------------


def test_config_hash_is_stable_and_distinguishes_configs(cfg: RiskConfig) -> None:
    h1 = _hash_config(cfg)
    h2 = _hash_config(cfg)
    assert h1 == h2

    altered = cfg.model_copy(update={"max_drawdown_pct": Decimal("0.10")})
    assert _hash_config(altered) != h1


def test_same_inputs_yield_same_decision_modulo_id(
    cfg: RiskConfig, account: AccountState
) -> None:
    rm = RiskManager(cfg)
    a = rm.evaluate(make_order(), account)
    b = rm.evaluate(make_order(), account)
    assert a.kind == b.kind
    assert a.reason == b.reason
    assert a.sized_order == b.sized_order
    assert a.config_hash == b.config_hash
    # decision_ids are UUIDs and so do differ — that's the only non-deterministic piece.
    assert a.decision_id != b.decision_id
