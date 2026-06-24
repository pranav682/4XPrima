# risk_manager (deterministic code — NOT an agent)

> The risk manager is **deterministic Python in the fast loop**. It is *not* an LLM agent, and the slow loop must not configure it at runtime. This spec exists so every agent and human knows what the risk manager guarantees and what its inputs are.

## Purpose

Be the single, deterministic, fail-safe gate between a *signal* and an *order*. Enforce the hard invariants from `CLAUDE.md`. Maintain and respect the kill switch.

## Responsibilities

For each `Signal` produced by a strategy:

1. **Position sizing** — compute size from per-trade risk budget, stop distance, account equity, and instrument tick value.
2. **Per-trade cap** — reject if proposed risk exceeds `max_risk_per_trade_pct` of equity.
3. **Per-symbol cap** — reject if it would breach `max_exposure_per_symbol_pct`.
4. **Portfolio cap** — reject if total open risk would exceed `max_portfolio_risk_pct`.
5. **Correlation cap** — reject if the new position correlates above `max_correlation` with existing exposure beyond `max_correlated_exposure_pct`.
6. **Drawdown cap** — if rolling drawdown (over `drawdown_window`) exceeds `max_drawdown_pct`, deny ALL new opens until reset; existing positions are managed normally.
7. **Kill switch** — if set, deny all opens and trigger the configured flatten policy (`flatten_immediately` or `let_stops_work`).
8. **Audit** — every decision (approved AND denied) is written to the append-only audit log with the inputs it used.

On any unhandled exception inside the risk manager, the kill switch flips to `engaged` automatically. **Fail safe means fail closed.**

## Inputs (with types)

```python
class RiskConfig:                            # loaded at startup from file; reload on SIGHUP
    schema_version: str
    account_currency: str
    max_risk_per_trade_pct: Decimal          # e.g. 0.5
    max_exposure_per_symbol_pct: Decimal
    max_portfolio_risk_pct: Decimal
    max_correlation: Decimal                 # 0.0..1.0
    max_correlated_exposure_pct: Decimal
    max_drawdown_pct: Decimal
    drawdown_window: timedelta
    kill_switch: KillSwitchConfig
    correlation_matrix_source: str           # path or vendor key

class RiskDecisionRequest:
    signal: Signal                           # symbol, side, entry, stop, target
    account: AccountSnapshot                 # equity, currency, open positions
    market: MarketSnapshot                   # quote, spread, last fill
    correlation_matrix: CorrelationMatrix
    now: datetime                            # UTC
```

## Outputs (strict structured schema)

```python
class RiskDecision:
    decision_id: str
    accepted: bool
    sized_order: Order | None                # set iff accepted
    rejected_by: list[RejectionReason]       # named per-gate
    notes: str                               # neutral, machine-parseable
    config_hash: str                         # which RiskConfig version made the call
```

`RejectionReason` is one of an enum: `PER_TRADE_CAP`, `PER_SYMBOL_CAP`, `PORTFOLIO_CAP`, `CORRELATION_CAP`, `DRAWDOWN_CAP`, `KILL_SWITCH`, `INVALID_INPUT`, `STOP_DISTANCE_NONPOSITIVE`.

## What the slow loop is and is NOT allowed to do

- **Allowed:** *propose* a new `RiskConfig` (a candidate file) for human review via `reporting_agent`. The proposal is just a file; it does nothing until a human approves and the deterministic deploy step writes it to the live path.
- **Allowed:** read the audit log (read-only).
- **Forbidden:** mutate `RiskConfig` at runtime, even temporarily. There is no API for this; the slow loop has no credentials to write the live config file.
- **Forbidden:** set or clear the kill switch. The kill switch is set by deterministic code (on exception or per the operator) and cleared by a deterministic operator command.

## Kill switch

- File-backed or env-backed flag, checked **on every decision**.
- When engaged: every `RiskDecisionRequest` returns `accepted=False, rejected_by=[KILL_SWITCH]`.
- Flatten policy (`flatten_immediately` | `let_stops_work`) executes deterministically by the order router, not by the risk manager itself.
- Engages automatically on: any unhandled exception in the fast loop; any failure to reload `RiskConfig`; any audit-log write failure.

## Testing

- Pure-function gates: 100% unit-test coverage with property-based tests (Hypothesis) for sizing math and cap arithmetic.
- Fuzz the kill switch: random injection of broker errors / config reload failures must never produce an `accepted=True` decision.
- Determinism test: given a fixed sequence of `RiskDecisionRequest`s, output bytes are identical across runs.

## Hand-offs

- **Upstream:** strategy engine produces `Signal`s.
- **Downstream:** approved `Order`s go to the order router; rejected decisions are audited and dropped.
- **Audit log** is the slow loop's read-only view into what happened.
