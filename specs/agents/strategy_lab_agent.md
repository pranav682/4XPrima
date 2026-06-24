# strategy_lab_agent

## Purpose

Propose candidate strategy *specifications* (entry/exit logic, indicators, timeframe, instruments) as structured objects that the backtester can consume verbatim. The agent **does not run code**; it writes the spec the deterministic engine will execute.

## Responsibilities

- Read `MarketContextReport` and the current champion's recent performance.
- Generate one or more `StrategyCandidate`s consistent with the regime tags.
- Tag each candidate with hypothesis ("trend continuation in EURUSD during Asia session"), indicator stack, timeframe, and stop/target rules.
- Avoid duplicates of recent candidates in the same run window (orchestrator passes a dedupe set).

## Inputs (with types)

```python
class StrategyLabRequest:
    run_id: str
    market_context: MarketContextReport
    champion_summary: ChampionSummary        # pair, params, recent P&L, drawdown
    recent_candidate_hashes: set[str]        # dedupe
    n_candidates: int                        # default 3, max 8
    allowed_instruments: list[CurrencyPair]
    allowed_timeframes: list[Timeframe]      # ["M5", "M15", "H1", "H4"]
    indicator_palette: list[str]             # whitelist (no LLM-invented indicators)
```

## Outputs (strict structured schema)

```python
class StrategyCandidate:
    candidate_id: str                        # deterministic hash of the spec
    run_id: str
    hypothesis: str                          # 1–2 sentences
    instrument: CurrencyPair
    timeframe: Timeframe
    session_filter: SessionFilter | None     # Asia/London/NY, optional
    entry_rules: list[Rule]                  # structured AST, not free text
    exit_rules: list[Rule]
    stop_rule: StopRule                      # fixed pips OR ATR-multiple
    target_rule: TargetRule                  # fixed / ATR / trailing
    indicators: list[IndicatorSpec]          # from indicator_palette only
    parameter_ranges: dict[str, ParamRange]  # bounded ranges optimizer may explore
    expected_trade_frequency: str            # qualitative; for plausibility check
    risk_notes: str                          # ≤ 300 chars
```

A `Rule` is a typed AST node, not natural language — examples in `skills/backtesting-methodology`.

## Tools & Skills used

**Tools:** none at runtime. All inputs are passed in. (Keeps the cache hot and forbids the agent from fetching new market data — that's `market_context_agent`'s job.)

**Skills:**
- `backtesting-methodology` — the rule grammar the backtester accepts.
- `forex-cost-modeling` — so candidates aren't proposed at timeframes where spread dominates edge.
- `prompt-caching-and-token-budget`.

## Model tier and why

**Sonnet** (`claude-sonnet-4-6`) by default. Promote to **Opus** (`claude-opus-4-7`) only for explicit "novel strategy design" sessions invoked by the orchestrator (set by env / config flag). Opus is not justified for routine candidate generation.

## System-prompt structure

```
[tools]                                              ← empty (none used)
[system]
  - identity & objective                             ← stable
  - rule grammar reference (compact)                 ← stable
  - indicator palette enumerated                     ← stable
  - backtesting-methodology skill                    ← stable
  - forex-cost-modeling skill                        ← stable
  -------------- cache_control breakpoint ------------
  - none (no per-call policy overrides)
[messages]
  - user: { market_context, champion_summary, dedupe set, n, allowed_* } ← volatile
```

## Token budget

- **Input target:** ≤ 12k (stable ~7k + volatile market context + champion summary).
- **Output target:** ≤ 4k for the typical 3-candidate response.
- Hard cap: 16k in / 8k out.

## Guardrails

- **Must not** invent indicators outside `indicator_palette`.
- **Must not** propose parameter values *outside* the ranges defined in `parameter_ranges` — those are the optimizer's job.
- **Must not** propose timeframes where the modelled cost (spread + commission) exceeds 25% of the candidate's expected edge per trade — flag and skip.
- **Must not** call any tool at runtime.
- **Must not** output candidates that hash-collide with `recent_candidate_hashes` (orchestrator re-checks; agent must self-dedupe first).

## Hand-offs

- Each `StrategyCandidate` is consumed by `backtest_agent`.
- Persisted under `data/runs/<run_id>/candidates/<candidate_id>.json`.
- The `candidate_id` is the join key the critic and orchestrator use later.
