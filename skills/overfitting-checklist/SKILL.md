---
name: overfitting-checklist
description: The adversarial checklist the critic_agent runs against every StrategyCandidate and ParameterProposal. Default verdict is reject; this is the list of things that have to pass for an accept.
---

# overfitting-checklist

## When to use

`critic_agent` runs this on every proposal. Other agents *may* read it for self-checks before submitting a proposal (cheaper to self-reject than to pay the critic's Opus tokens).

## How to use it

Treat each numbered check as a named test. Each must produce a `CriticCheckResult` with `{name, pass, evidence_citation, severity}`. **Any `pass = false` collapses the verdict to `reject`** (or `needs_more_evidence` if the check is uncomputable from the provided report). There is no "weighted score" — gates are hard.

## The checks

### 1. Out-of-sample decay

- **Pass:** `oos_sharpe / is_sharpe ≥ 0.5` AND OOS profit factor ≥ 1.10 AND OOS trade count ≥ 30.
- **Why:** large IS→OOS gap is the signature of overfitting; we tolerate decay but not collapse.

### 2. Walk-forward stability

- **Pass:** at least **60%** of walk-forward folds individually pass the IS gates.
- **Pass:** per-fold Sharpe coefficient of variation `CV(sharpe) ≤ 1.0`.
- **Why:** a strategy that works on average but is a roller-coaster fold-to-fold is fitting noise per regime.

### 3. Cost sensitivity

- **Pass:** OOS Sharpe at **1.5× cost** ≥ 0.3 AND profit factor ≥ 1.05.
- **Pass:** OOS Sharpe at **2.0× cost** > 0 (not necessarily strong, but positive).
- **Why:** if a tiny widening of spread kills the strategy, the edge is cost-arbitrage, not signal.

### 4. Regime dependence

- **Pass:** no single `RegimeLabel` in `regime_breakdown` contributes more than `regime_dependence_threshold` (default **0.6**) of total OOS P&L.
- **Pass:** the strategy has at least one non-trivial regime where it is *not* losing.
- **Why:** a strategy that only works in one regime is one regime change away from blowing up.

### 5. Parameter sensitivity

- **Pass:** small perturbations of each parameter (±10% within `parameter_ranges`) keep OOS Sharpe within 30% of nominal.
- **Pass:** the *neighbourhood* of the parameter point in IS space is not a needle in haystack — visualised as a smooth-ish region, not a spike.
- **Why:** sharp optima are overfit; robust optima are robust.

### 6. Survivorship / look-ahead / data-snooping smells

- **Pass:** no indicator uses a window that exceeds the available history at its evaluation time.
- **Pass:** no rule references a future bar (close-on-open, etc.).
- **Pass:** no metric depends on the survivorship of the watchlist (instruments delisted mid-history are accounted for).
- **Pass:** the candidate's structure is not obviously derived from inspecting the OOS slice (e.g. parameters that snap to "interesting" recent events).
- **Why:** these are the classic mechanical errors that silently inflate Sharpe.

### 7. Trade-count sanity

- **Pass:** in-sample trade count ≥ `minimum_trade_count` (default 100).
- **Pass:** OOS trade count ≥ 30.
- **Pass:** at the same time, trade frequency × commission ≤ 30% of gross profit (not fee-bled to death).
- **Why:** too few trades → no statistical power; too many trades → all edge eaten by fees.

### 8. Drawdown shape

- **Pass:** max drawdown ≤ `RiskConfig.max_drawdown_pct`.
- **Pass:** drawdown recovery time ≤ 2× the average winning-period duration (no eternal underwater).
- **Why:** survivability matters as much as ending P&L.

## Output format

```python
class CriticCheckResult:
    name: str                                # one of the eight above
    pass: bool
    evidence_citation: str                   # quotes the metric from the BacktestInterpretation
    severity: Literal["info", "warn", "fail"]
```

If a check is uncomputable (e.g. the report lacks a regime breakdown for check #4), `pass = False` and the verdict is `needs_more_evidence`, with a note explaining what's missing. **Do not pass a check on a guess.**

## Checklist for the critic_agent at runtime

- [ ] All 8 checks evaluated.
- [ ] Each check cites a specific field from the `BacktestInterpretation`.
- [ ] `verdict = "accept"` only if every check is `pass`.
- [ ] `kill_reasons` enumerates every failing check.
- [ ] Did **not** ask for another backtest (use `needs_more_evidence`).
