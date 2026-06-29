# backtest_agent

## Purpose

Turn a `StrategyProposal` into honest, judged backtest results: build each strategy, run the deterministic Stage-2 engine, and **interpret** the output into a verdict. The agent does no heavy compute and **never produces or alters a number** — it interprets metrics the deterministic code computed.

> **Build status (Stage 3).** Implemented in the **in-sample-only, harness-based** form below, reconciled to the real Stage-2 engine (`core/backtest/`):
> - The deterministic **harness** (`core/agents/backtest_harness.py`) — NOT the LLM and NOT a runtime tool — does all compute: `build_strategy(candidate)` → the engine over the **in-sample** window → a frozen `BacktestEvidence` per candidate (verbatim metrics, cost breakdown, `config_hash`, fixed-gate flags). The agent consumes the evidence already computed; it calls no tools at runtime (keeps the cache hot).
> - **In-sample only.** The OOS holdout stays sealed — no path in this agent or harness supplies the OOS token. The richer gate set sketched later in this doc (**out-of-sample, walk-forward, cost-sensitivity, regime-breakdown**) is a **future extension**, deferred until the harness orchestrates those runs. The engine's metric pack also has no t-stat / average-R yet; the verdict reports the metrics the engine actually computes.
> - Profitability is decided later by the OOS slice + the critic, never here. This agent triages.

## Responsibilities

- For each `StrategyCandidate` (or `ParameterProposal`), build a `BacktestRunConfig` honouring `skills/backtesting-methodology` (walk-forward, out-of-sample split, cost model from `skills/forex-cost-modeling`).
- Submit the run via the `backtest_run` tool.
- Read the `BacktestRunReport` (deterministic JSON from `backtest/`).
- Produce a `BacktestInterpretation` with explicit go / no-go on each gate (in-sample, out-of-sample, walk-forward, cost sensitivity).

## Inputs (with types) — as built

```python
@dataclass(frozen=True)
class BacktestAgentRequest:        # core.agents.backtest_agent
    run_id: str
    proposal: StrategyProposal     # from strategy_lab
    evidence: tuple[BacktestEvidence, ...]   # ALREADY computed by the harness
    tier: ModelTier = ModelTier.DEFAULT
```

The deterministic harness is configured separately:

```python
@dataclass(frozen=True)
class BacktestRunConfig:           # core.agents.backtest_harness
    lookback_count: int = 1000     # bars per candidate
    oos_fraction: float = 0.2      # held-out tail — sealed, never run here
    starting_balance: Decimal
    cost_model: CostModel
    risk_config: RiskConfig        # every order routes through the real RiskManager
    # fixed deterministic gate thresholds:
    min_trade_count: int = 30
    max_drawdown_ceiling: Decimal = Decimal("0.30")
    min_profit_factor: Decimal = Decimal("1.0")
```

## Outputs (strict structured schema) — as built

```python
class BacktestVerdictSet(BaseModel):    # core.models, frozen, extra=forbid
    run_id: str
    schema_version: str = "1.0"
    verdicts: tuple[BacktestVerdict, ...]

class BacktestVerdict(BaseModel):       # core.models, frozen, extra=forbid
    candidate_id: str
    config_hash: str                    # must match the run's evidence
    metrics: BacktestMetricsView        # copied VERBATIM from the evidence
    gates: tuple[GateResult, ...]       # copied verbatim
    assessment: str                     # ≤ 800 chars, honest in-sample read
    concerns: tuple[str, ...]           # overfit smells, fragility
    triage: BacktestTriage              # advance_to_critic | reject | needs_different_params
    caveats: str                        # ≤ 500 chars, in-sample-only / not-predictive
```

`BacktestMetricsView` carries exactly the engine's metric pack (total/annualised
return, Sharpe, Sortino, max drawdown, win rate, profit factor, trade count, avg
trade PnL, exposure); `sortino_ratio` / `profit_factor` are `null` when the
deterministic value is infinite (no downside / no losing trades). There is **NO
out-of-sample metric and NO live/deploy field** anywhere.

## Tools & Skills used

**Tools: none at runtime** (as built). The backtest is run by the deterministic
harness BEFORE the agent call; the agent receives the evidence as data. This
keeps the cache hot and removes any path for the LLM to re-run or alter a
backtest. (The spec originally imagined `backtest_run` / `read_strategy_spec`
tools; the harness supersedes them.)

**Skills:**
- `backtesting-methodology` — gate thresholds and walk-forward rules.
- `forex-cost-modeling` — for the cost model config.
- `overfitting-checklist` — sanity-checks before delivering pass.
- `prompt-caching-and-token-budget`.

## Model tier and why

**DEFAULT tier** (`ModelTier.DEFAULT` → `gpt-5.4`). Interpreting a structured report against fixed gates is high-stakes but well-bounded — the heavy reasoning is done by the deterministic harness. The HEAVY tier is reserved for the critic.

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
