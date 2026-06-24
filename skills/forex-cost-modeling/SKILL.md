---
name: forex-cost-modeling
description: Models the real-world cost of a forex trade — spread, slippage, commission, swap/rollover, weekend gaps — for use in backtesting and gate evaluation.
---

# forex-cost-modeling

## When to use

Any backtest, any parameter sweep, any cost-sensitivity check. Costs that don't model these factors will systematically over-state edge.

## The method

A simulated fill on instrument `i` at time `t` costs the strategy:

```
total_cost = spread_cost + commission + slippage + swap (if held over rollover)
```

### 1. Spread

- **Source:** the live or replay quote stream's bid/ask. Do **not** use mid-quote and ignore spread.
- **Model:** per-instrument typical spread schedule by **session** (Asia / London / NY / overlap) and by **calendar context** (pre-news widening, holiday thin liquidity, weekend re-open).
- **Widening rules:**
  - News window (`±15 min` around tier-1 event): spread = 2× session typical.
  - Holiday / low-liquidity day: spread = 1.5× session typical.
  - 30 min before and 30 min after the Sunday open: spread = 3× session typical.

### 2. Commission

- Per-instrument, per-lot, applied to **both legs** (open and close).
- Configured in `CostModelConfig.commission_per_lot` (per side, account-currency-denominated).

### 3. Slippage

- Modelled as a function of:
  - **Order type** (market vs limit): limit orders that fill — no slippage; market orders — slipped.
  - **Pre-trade volatility** (rolling N-bar ATR): higher ATR → higher slippage.
  - **Order size relative to typical session volume**.
- Default: `slippage_pips = max(0.3, 0.05 * ATR_pips * size_ratio)`. Tunable per instrument in `CostModelConfig.slippage_model`.

### 4. Swap / rollover

- Applies to positions held over **17:00 New York time** (broker convention).
- Triple swap on **Wednesday** rollover (covers weekend).
- Per-instrument long/short swap points in `CostModelConfig.swap_table`.
- Backtests **must** carry swap if any candidate strategy holds overnight; ignoring swap on a carry-sensitive pair (e.g. USDTRY, ZAR pairs) is a disqualifying error.

### 5. Weekend gaps

- For any position held over the weekend, simulate a **gap fill** at Sunday open based on the realised gap distribution from history (per instrument).
- Stops are honoured at the **post-gap price** — they do not magically execute at the pre-gap stop level.
- If `RiskConfig.no_weekend_hold = true`, the backtester must close any open position by `Friday 16:55 NY` and audit the close.

## Cost-sensitivity test (input to the critic)

Re-run the OOS pass at:

- **1.0×** (base) — primary metric for gate evaluation.
- **1.5×** — small adverse environment.
- **2.0×** — adversarial environment.

The critic compares all three. A strategy whose Sharpe collapses below 0 at 1.5× is fragile to cost and **must** be flagged.

## Checklist when configuring a backtest's cost model

- [ ] Per-instrument spread schedule by session is loaded.
- [ ] News-window widening rule is enabled.
- [ ] Commission per lot is set per broker tier (paper config = our intended live broker).
- [ ] Slippage model is enabled for market orders, off for limit fills.
- [ ] Swap table is loaded; triple-Wednesday flag is on.
- [ ] Weekend-gap distribution is loaded for each instrument held overnight.
- [ ] Cost-sensitivity multipliers (1.0, 1.5, 2.0) are configured.

## What this skill does NOT cover

- Risk-based position sizing — see `skills/risk-management-rules`.
- Strategy-level entry/exit logic — see `skills/backtesting-methodology`.
