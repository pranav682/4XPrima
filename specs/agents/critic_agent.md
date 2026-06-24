# critic_agent

## Purpose

**Adversarial.** Try to **kill** every `StrategyCandidate` and every `ParameterProposal` using the overfitting checklist before it ever reaches a human. Default verdict is `reject` — the proposal must clear every check to flip to `accept`.

## Responsibilities

- Read the proposal/candidate, its `BacktestInterpretation`, and the relevant `MarketContextReport`.
- Run the `overfitting-checklist` (see skill) as a sequence of named tests:
  - Out-of-sample decay.
  - Walk-forward instability across folds.
  - Cost sensitivity (1.5× and 2× spread/commission).
  - Regime dependence (does it only work in one regime?).
  - Parameter sensitivity (small perturbations should not collapse the result).
  - Survivorship / look-ahead / data-snooping smells.
  - Trade-count sanity (too few trades → no statistical power; too many → fee bleed).
- For each test, emit a structured result with the **specific evidence** in the backtest report.
- Render a final `CriticVerdict` of `accept` (all gates pass), `reject` (one or more fail), or `needs_more_evidence` (a gate cannot be evaluated from the provided report).

## Inputs (with types)

```python
class CriticRequest:
    run_id: str
    subject: StrategyCandidate | ParameterProposal
    backtest_interpretation: BacktestInterpretation
    market_context: MarketContextReport
    minimum_trade_count: int                 # gate per skills/overfitting-checklist
    regime_dependence_threshold: float       # max share of P&L from a single regime
    cost_multipliers: list[float]            # e.g. [1.5, 2.0]
```

## Outputs (strict structured schema)

```python
class CriticVerdict:
    verdict_id: str
    run_id: str
    subject_id: str                          # candidate_id or proposal_id
    schema_version: str = "1.0"
    checks: list[CriticCheckResult]          # one per checklist item
    verdict: Literal["accept", "reject", "needs_more_evidence"]
    kill_reasons: list[str]                  # required if verdict != "accept"
    confidence: float                        # 0..1
    notes: str                               # ≤ 600 chars, neutral
```

Each `CriticCheckResult`: `{name, pass, evidence_citation, severity}`.

## Tools & Skills used

**Tools:** none at runtime. The deterministic backtester has already produced everything the critic needs; if information is missing, the answer is `needs_more_evidence`, **not** "run another backtest".

**Skills:**
- `overfitting-checklist` — the canonical list of tests and pass/fail bands.
- `backtesting-methodology` — to reason about walk-forward and OOS construction.
- `forex-cost-modeling` — for the cost sensitivity test.
- `prompt-caching-and-token-budget`.

## Model tier and why

**Opus** (`claude-opus-4-7`). This is the highest-stakes single decision in the slow loop — a wrong `accept` lets a curve-fit go in front of a human under a false halo of validation. Opus's adversarial reasoning is the budget we explicitly pay for. Cache the (long, stable) checklist prompt aggressively; per-call inputs are small.

## System-prompt structure

```
[tools]                                          ← empty
[system]
  - identity: "you are an adversarial critic. Default to reject." ← stable
  - schema for CriticVerdict                     ← stable
  - overfitting-checklist skill (full text)      ← stable, large
  - backtesting-methodology skill                ← stable
  - forex-cost-modeling skill                    ← stable
  -------------- cache_control breakpoint ---------
  - none
[messages]
  - user: { subject, backtest_interpretation, market_context, thresholds }
```

Opus cache minimum is 4k tokens — the stable system block above is well above that, so caching is real.

## Token budget

- **Input target:** ≤ 18k (stable ~10–12k, volatile up to 6k).
- **Output target:** ≤ 3k (structured checks + verdict).
- Hard cap: 25k in / 6k out.

## Guardrails

- **Default verdict is `reject`.** The agent must positively cite passing evidence to flip to `accept`.
- **Must not** mark `accept` if any check fails — collapse to `reject` (or `needs_more_evidence` if a check is uncomputable).
- **Must not** ask for more backtests — `needs_more_evidence` is signal to the orchestrator that *deterministic* code (more data, longer OOS) is required.
- **Must not** restate the strategy's promise; the job is to attack, not summarise.
- **Must not** be swayed by the `interpretation` field of the backtest — it reasons over the raw metrics in `GateResult`s.

## Hand-offs

- `CriticVerdict` is consumed by:
  - `orchestrator_agent` (state machine: only `accept` advances to human review),
  - `reporting_agent` (the human sees both the proposal AND the critic verdict, side by side).
- Persisted under `data/runs/<run_id>/verdicts/<verdict_id>.json`.
