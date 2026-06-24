# backtest_agent

## Purpose

Configure backtests, drive the deterministic backtester (which lives in `backtest/`), and **interpret** the results. The agent does no heavy compute — it sets up runs, reads the structured output, and writes a verdict.

## Responsibilities

- For each `StrategyCandidate` (or `ParameterProposal`), build a `BacktestRunConfig` honouring `skills/backtesting-methodology` (walk-forward, out-of-sample split, cost model from `skills/forex-cost-modeling`).
- Submit the run via the `backtest_run` tool.
- Read the `BacktestRunReport` (deterministic JSON from `backtest/`).
- Produce a `BacktestInterpretation` with explicit go / no-go on each gate (in-sample, out-of-sample, walk-forward, cost sensitivity).

## Inputs (with types)

```python
class BacktestRequest:
    run_id: str
    target: StrategyCandidate | ParameterProposal
    data_window: DataWindow                  # full date range available
    walk_forward: WalkForwardConfig          # window / step / n_folds
    oos_holdout: OOSConfig                   # held-out tail, never touched in fitting
    cost_model: CostModelConfig              # spread, slippage, commission, swap
    seeds: list[int]                         # for any stochastic component
```

## Outputs (strict structured schema)

```python
class BacktestInterpretation:
    run_id: str
    target_id: str                           # candidate_id or proposal_id
    schema_version: str = "1.0"
    in_sample: GateResult                    # metrics + pass/fail vs thresholds
    out_of_sample: GateResult
    walk_forward: WalkForwardResult          # per-fold + aggregate
    cost_sensitivity: CostSensitivityResult  # P&L under 1.5x and 2x cost
    regime_breakdown: dict[RegimeLabel, GateResult]
    overall: Literal["pass", "fail", "marginal"]
    fail_reasons: list[str]                  # empty if pass
    interpretation: str                      # ≤ 800 chars, neutral, no recommendations
    raw_report_path: str                     # pointer to deterministic JSON for audit
```

`GateResult` includes: total return, Sharpe, Sortino, max drawdown, hit rate, profit factor, average R, trade count, exposure pct, t-stat vs zero — see `skills/backtesting-methodology`.

## Tools & Skills used

**Tools (eager):**
- `backtest_run` — submits a `BacktestRunConfig`, returns a `BacktestRunReport` (deterministic).
- `read_strategy_spec` — fetches a `StrategyCandidate` by id (idempotent).

**Tools (deferred):**
- `backtest_export_plots` — only when reporting agent later asks for visuals.

**Skills:**
- `backtesting-methodology` — gate thresholds and walk-forward rules.
- `forex-cost-modeling` — for the cost model config.
- `overfitting-checklist` — sanity-checks before delivering pass.
- `prompt-caching-and-token-budget`.

## Model tier and why

**Sonnet** (`claude-sonnet-4-6`). Interpreting a structured report against fixed gates is high-stakes but well-bounded — the heavy reasoning is done by the deterministic backtester. Opus is reserved for the critic.

## System-prompt structure

```
[tools]                                          ← stable (backtest_run, read_strategy_spec eager)
[system]
  - identity & objective                         ← stable
  - gate thresholds (pass/fail bands)            ← stable
  - schema for BacktestInterpretation            ← stable
  - backtesting-methodology skill                ← stable
  - forex-cost-modeling skill                    ← stable
  - overfitting-checklist skill                  ← stable
  -------------- cache_control breakpoint ---------
  - dynamic gate overrides (rare; e.g. tightened Sharpe threshold for live promotion)
[messages]
  - user: { target spec, data window, walk_forward, oos, cost_model, seeds, run_id }
```

Long-running variant (running many candidates in a session) enables `context_management.clear_tool_uses_20250919` per `docs/llm-conventions.md` §2.

## Token budget

- **Input target:** ≤ 15k (stable ~8k + target spec + backtester report excerpt).
- **Output target:** ≤ 3k.
- Hard cap: 25k in / 6k out — if the report excerpt is bigger, summarise via the deterministic side first.

## Guardrails

- **Must not** alter the `BacktestRunConfig` after submission (no retrying with friendlier costs).
- **Must not** mark `overall = "pass"` if any gate fails — only `marginal` or `fail` are allowed in that case.
- **Must not** invent metrics; everything in `interpretation` cites the report.
- **Must not** propose strategy or parameter changes — that's `optimization_agent`.
- **Must not** call `backtest_export_plots` unless explicitly requested in the input.

## Hand-offs

- `BacktestInterpretation` is consumed by:
  - `optimization_agent` (if `pass` or `marginal`, drives parameter exploration),
  - `critic_agent` (always),
  - `orchestrator_agent` (state update),
  - `reporting_agent` (human summary).
- Persisted under `data/runs/<run_id>/backtests/<target_id>.json`.
