# risk_manager (deterministic code — NOT an agent)

> The risk manager is **deterministic Python in the fast loop**. It is *not* an LLM agent, and the slow loop must not configure it at runtime. This spec exists so every agent and human knows what the risk manager guarantees and what its inputs are.

## Purpose

Be the single, deterministic, fail-safe gate between a *signal* and an *order*. Enforce the hard invariants from `CLAUDE.md`. Maintain and respect the kill switch.

## Responsibilities

**Intended vs realized risk.** Every cap that is enforced *before* the fill (per-trade, per-pair, correlated, portfolio) measures the **intended** risk computed from `stop_distance` — i.e. the loss that would occur if the stop filled exactly. The **realized** risk (what actually happens when a stop gaps through) is bounded by the *backstop* gates: the drawdown and daily-loss auto-trips. Together they enforce the safety story even when a single fill exceeds its modelled stop loss.

**Reject vs resize.** Only the per-trade cap RESIZES. Portfolio, per-pair, and correlated exposure caps REJECT. This is intentional: resizing one new order to fit an aggregate cap would silently obscure that the *portfolio* is already at its limit. The strategy should see the rejection and decide whether to close something else (the slow-loop optimizer may propose this) — the risk manager doesn't make that call.

For each order produced by a strategy, evaluate the gates **in this order**:

1. **Kill switch** — if engaged, reject immediately.
2. **Input validity** — `stop_distance > 0`, `equity > 0`. Otherwise reject as `STOP_DISTANCE_NONPOSITIVE` / `NONPOSITIVE_EQUITY`.
3. **Drawdown cap** — if current drawdown (equity vs `peak_equity`) ≥ `max_drawdown_pct`, **trip the kill switch** and reject.
4. **Daily loss limit** — if today's loss (equity vs `day_start_equity`) ≥ `daily_loss_limit_pct`, **trip the kill switch** and reject.
5. **Max concurrent positions** — reject if `len(open_positions) ≥ max_concurrent_positions`.
6. **Per-trade risk cap** — if `risk_at_stop > max_risk_per_trade_pct × equity`, **resize the order down** so risk equals the cap exactly. (This is the only cap that resizes; the rest reject.)
7. **Per-pair notional exposure cap** — reject if combined notional on this pair would exceed `max_exposure_per_pair_pct × equity`.
8. **Correlated exposure cap** — for every `correlation_group` containing the pair, reject if the group's combined notional would exceed `max_correlated_exposure_pct × equity`.
9. **Aggregate portfolio risk cap** — reject if total open `risk_at_stop` would exceed `max_portfolio_risk_pct × equity`.
10. **Audit** — every decision (approved, resized, OR rejected) is logged structurally with the inputs it used and the `config_hash` in force.

On any unhandled exception inside the risk manager, the kill switch flips to `engaged` automatically. **Fail safe means fail closed.**

## Inputs (with types)

Canonical types live in `core/models.py`. Summary:

```python
class RiskConfig:                            # frozen pydantic model; reload = new instance
    schema_version: str
    account_currency: str
    max_risk_per_trade_pct: Decimal          # loss-side, in (0, 1]
    max_portfolio_risk_pct: Decimal          # loss-side, in (0, 1]
    max_concurrent_positions: int            # ≥ 1
    max_exposure_per_pair_pct: Decimal       # notional; may exceed 1 under leverage
    max_correlated_exposure_pct: Decimal     # notional, per correlation group
    correlation_groups: dict[str, tuple[str, ...]]  # group_name → pairs in group
    daily_loss_limit_pct: Decimal            # loss-side, in (0, 1]
    max_drawdown_pct: Decimal                # loss-side, in (0, 1]

class OrderRequest:
    pair: str
    direction: Direction                     # LONG | SHORT
    size: Decimal                            # requested size, base-currency units
    entry_price: Decimal
    stop_price: Decimal

class AccountState:
    balance: Decimal
    equity: Decimal
    open_positions: tuple[Position, ...]
    realized_pnl_today: Decimal
    unrealized_pnl: Decimal
    peak_equity: Decimal                     # rolling peak, for drawdown
    day_start_equity: Decimal                # for daily loss limit
    as_of: datetime                          # UTC, tz-aware
```

`RiskManager.evaluate(order, account) → RiskDecision`. There is no `RiskDecisionRequest` wrapper at the runtime layer; the upstream signal-to-order translator is what shapes a `Signal` into an `OrderRequest`.

## Outputs (strict structured schema)

```python
class RiskDecision:
    decision_id: str
    kind: DecisionKind                       # APPROVE | RESIZE | REJECT
    sized_order: OrderRequest | None         # set iff APPROVE or RESIZE
    rejected_by: tuple[RejectionReason, ...] # may be non-empty even on RESIZE? No — empty unless REJECT.
    reason: str                              # human-readable
    limiting_rule: RejectionReason | None    # the gate that drove the outcome
    config_hash: str                         # which RiskConfig version made the call
    as_of: datetime                          # UTC, mirrors AccountState.as_of
```

`accepted` is a derived property: True iff `kind ∈ {APPROVE, RESIZE}`.

`RejectionReason` (closed enum):
`KILL_SWITCH`, `PER_TRADE_CAP`, `MAX_CONCURRENT_POSITIONS`, `PORTFOLIO_RISK_CAP`, `PER_PAIR_EXPOSURE_CAP`, `CORRELATED_EXPOSURE_CAP`, `DAILY_LOSS_LIMIT`, `DRAWDOWN_CAP`, `INVALID_INPUT`, `STOP_DISTANCE_NONPOSITIVE`, `NONPOSITIVE_EQUITY`.

## What the slow loop is and is NOT allowed to do

- **Allowed:** *propose* a new `RiskConfig` (a candidate file) for human review via `reporting_agent`. The proposal is just a file; it does nothing until a human approves and the deterministic deploy step writes it to the live path.
- **Allowed:** read the audit log (read-only).
- **Forbidden:** mutate `RiskConfig` at runtime, even temporarily. There is no API for this; the slow loop has no credentials to write the live config file.
- **Forbidden:** set or clear the kill switch. The kill switch is set by deterministic code (on exception or per the operator) and cleared by a deterministic operator command.

## Kill switch

- **Latching.** Once engaged, every subsequent `evaluate` call rejects with `KILL_SWITCH` until a human resets it. Multiple "trip" calls preserve the *first* reason — the root cause — and log the rest as `kill_switch_already_engaged` warnings.
- Engages automatically on:
  - drawdown ≥ `max_drawdown_pct` (`tripped_by="drawdown"`),
  - daily loss ≥ `daily_loss_limit_pct` (`tripped_by="daily_loss"`),
  - any unhandled exception in the fast loop (`tripped_by="exception"`),
  - any failure to reload `RiskConfig` or write the audit log (`tripped_by="audit"` / `tripped_by="config_reload"`).
- Engages manually via `RiskManager.trip(reason, tripped_by=...)`.
- Resets only via `RiskManager.reset(operator=..., confirmation="I_UNDERSTAND_RESET")` — the confirmation token blocks accidental resets from automated callers.
- Flatten policy (`flatten_immediately` | `let_stops_work`) executes deterministically by the order router, not by the risk manager itself.

## Testing

- Pure-function gates: 100% unit-test coverage with property-based tests (Hypothesis) for sizing math and cap arithmetic.
- Fuzz the kill switch: random injection of broker errors / config reload failures must never produce an `accepted=True` decision.
- Determinism test: given a fixed sequence of `RiskDecisionRequest`s, output bytes are identical across runs.

## Hand-offs

- **Upstream:** strategy engine produces `Signal`s.
- **Downstream:** approved `Order`s go to the order router; rejected decisions are audited and dropped.
- **Audit log** is the slow loop's read-only view into what happened.
