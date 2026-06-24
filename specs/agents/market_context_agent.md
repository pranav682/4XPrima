# market_context_agent

## Purpose

Build a structured snapshot of the *current* macro and microstructure environment that downstream agents (strategy lab, optimization, critic) and the human can rely on. **Never** outputs trade calls or directional opinions framed as recommendations.

## Responsibilities

- Pull macro indicators (rates, CPI, NFP, central-bank statements) via tool calls.
- Pull the upcoming **economic calendar** for the agent's watchlist currencies for the next 24h / 7d windows.
- Pull recent news and sentiment for the watchlist currencies.
- Classify the **regime** (trending / mean-reverting / event-driven / risk-off / risk-on) per pair.
- Flag scheduled events likely to widen spreads or invalidate intraday strategies.
- Emit a single `MarketContextReport` (schema in `skills/market-context-format`).

## Inputs (with types)

```python
class MarketContextRequest:
    run_id: str                              # set by orchestrator
    as_of: datetime                          # UTC, included in volatile tail
    watchlist: list[CurrencyPair]            # e.g. ["EURUSD", "GBPUSD", "USDJPY"]
    lookback_days: int                       # macro/news window, default 7
    calendar_horizon_hours: int              # default 168 (7 days)
```

## Outputs (strict structured schema)

```python
class MarketContextReport:
    run_id: str
    as_of: datetime                          # UTC
    schema_version: str = "1.0"
    regime: dict[CurrencyPair, RegimeLabel]  # {"EURUSD": "trending_up", ...}
    regime_confidence: dict[CurrencyPair, float]   # 0.0–1.0
    scheduled_events: list[ScheduledEvent]   # ordered by start_time
    surprise_vs_expectation: list[SurpriseEvent]  # released - consensus
    sentiment: dict[CurrencyPair, SentimentScore] # -1.0..+1.0 + source counts
    risk_flags: list[RiskFlag]               # e.g. "FOMC_T+18h", "thin_liquidity_window"
    notes: str                               # free-form, ≤ 500 chars, neutral
    citations: list[Citation]                # source URLs / data tickers, deduped
```

See `skills/market-context-format/SKILL.md` for the canonical JSON schema and field-level rules.

## Tools & Skills used

**Tools (eager):**
- `economic_calendar_lookup` — calendar API wrapper.
- `news_search` — vendor news search restricted to the watchlist currencies.
- `macro_series_lookup` — fixed list of macro tickers (FRED-style).

**Tools (deferred via Tool Search Tool):**
- Per-vendor sentiment APIs, alt-data sources, optional web fetch — loaded on demand.

**Skills:**
- `market-context-format` — output schema and field rules.
- `prompt-caching-and-token-budget` — to keep the agent within budget.

## Model tier and why

**Sonnet** (`claude-sonnet-4-6`). The task is multi-source synthesis into a fixed schema; Sonnet hits the cost/accuracy sweet spot. Haiku is too brittle for nuanced regime classification; Opus is overkill for what is essentially summarisation + classification.

## System-prompt structure

```
[tools]                                                ← stable (per agent version)
[system]
  - identity & objective                               ← stable
  - "you are NOT allowed to recommend trades"          ← stable
  - market-context-format skill block                  ← stable
  - prompt-caching skill block (compact)               ← stable
  - regime taxonomy & risk-flag taxonomy               ← stable
  -------------- cache_control breakpoint --------------
  - dynamic policy overrides (e.g. holiday window)     ← volatile, AFTER breakpoint
[messages]
  - user: { run_id, as_of, watchlist, horizons, recent fast-loop audit summary } ← volatile
```

Single explicit `cache_control: ephemeral` at end of the stable system block.

## Token budget

- **Input target:** ≤ 8k tokens (cached prefix ~5–6k, volatile tail ≤ 2k).
- **Output target:** ≤ 2k tokens (structured JSON; long prose forbidden by schema).
- Hard cap: 12k in / 4k out — exceeds → fail the run and alert orchestrator.

## Guardrails (what it must NOT do)

- **Must not** output trade calls, directional bias couched as advice, or position sizing.
- **Must not** invent macro numbers or calendar entries — every datum cites a tool result.
- **Must not** include opinions about other market participants (no "smart money is…").
- **Must not** dynamically add tools mid-call (would invalidate cache and tool list).
- **Must not** emit free text outside the `notes` field (schema-violating → orchestrator fails the run).

## Hand-offs

- `MarketContextReport` is consumed by:
  - `strategy_lab_agent` (to shape candidate generation),
  - `optimization_agent` (regime tag conditions which parameter set is in play),
  - `critic_agent` (cost / regime-sensitivity checks),
  - `reporting_agent` (human summary).
- Persisted under `data/runs/<run_id>/market_context.json` for replayability.
