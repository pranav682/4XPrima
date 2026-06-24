---
name: backtesting-methodology
description: Canonical method for backtesting 4xPrima strategies — walk-forward, out-of-sample holdout, gate thresholds, champion/challenger promotion. Used by backtest_agent, optimization_agent, critic_agent.
---

# backtesting-methodology

## When to use

Any time a `StrategyCandidate` or `ParameterProposal` is being evaluated. Backtests configured outside this method are not eligible for proposal to a human.

## The method

### 1. Split the data, *once*, at config time

- **In-sample (IS):** the older portion of the available history. Used for fitting / parameter selection inside walk-forward folds.
- **Out-of-sample (OOS) holdout:** the most recent contiguous tail. **Never touched** during fitting or optimization. Sized at ≥ 20% of total history, with a hard minimum that yields at least the gate's `minimum_trade_count`.
- The split is recorded in `BacktestRunConfig` and immutable for that candidate's lifetime.

### 2. Walk-forward on the IS portion

- Rolling **anchored** walk-forward: train window grows, test window slides forward by `step`.
- Default: `train_window = 6 months`, `test_window = 1 month`, `step = 1 month`. Override per timeframe.
- Each fold reports its own `GateResult`. Aggregate stats are computed across folds (not across the concatenated test slices) so a single great fold can't carry the rest.

### 3. Single OOS pass at the end

- Apply the *finalised* configuration once to the OOS holdout. **No re-fitting** on OOS.
- If OOS is touched more than once during a candidate's lifetime, mark the candidate `oos_burned` and retire it.

### 4. Cost model

Use `skills/forex-cost-modeling`. Run the final OOS pass at **base cost** AND at **1.5×** and **2×** cost — these are the cost-sensitivity inputs the critic uses.

### 5. Gate thresholds (default)

A `GateResult` *passes* if **all** of the following hold:

| Metric | Pass threshold |
| --- | --- |
| Trade count | ≥ `minimum_trade_count` (default 100 in-sample, 30 OOS) |
| Sharpe (annualised, net of cost) | ≥ 0.8 IS; ≥ 0.5 OOS |
| Profit factor | ≥ 1.25 IS; ≥ 1.10 OOS |
| Max drawdown | ≤ `max_drawdown_pct` from `RiskConfig` |
| OOS / IS Sharpe ratio | ≥ 0.5 (i.e. OOS retains half of IS Sharpe) |
| Walk-forward fold pass rate | ≥ 60% of folds pass the IS thresholds individually |

Mark `overall = "marginal"` if one of OOS Sharpe, OOS profit factor, or fold pass rate is within 10% of its threshold and the others pass. Otherwise `pass` or `fail`.

### 6. Champion / challenger promotion

A challenger may **only** propose to replace the champion if all of:

1. Its OOS gate passes the thresholds above.
2. Its OOS Sharpe exceeds the **champion's recent live Sharpe** by ≥ 20%, measured over a window of at least 30 trading days (or the champion's age, whichever is shorter).
3. Its max drawdown is no greater than 1.25× the champion's realised max drawdown.
4. `critic_agent` returns `accept`.
5. A human approves via `reporting_agent`'s approval surface.

A failed promotion does **not** kill the challenger — it returns to the candidate pool for another cycle. After three failed promotion attempts, the challenger is archived to prevent endless re-test loops.

## Checklist before submitting a backtest

- [ ] OOS holdout is the most recent contiguous tail, ≥ 20% of history.
- [ ] Walk-forward params are set; folds will produce at least 6 test slices.
- [ ] Cost model uses base + 1.5× + 2× spreads (see `skills/forex-cost-modeling`).
- [ ] `minimum_trade_count` is set per timeframe.
- [ ] `seeds` is set for any stochastic step.
- [ ] OOS slice will be touched exactly once.

## What this skill does NOT cover

- The cost model parameters themselves — see `skills/forex-cost-modeling`.
- The adversarial checks beyond simple gates — see `skills/overfitting-checklist`.
- The risk-management constraints applied to live execution — see `skills/risk-management-rules`.
