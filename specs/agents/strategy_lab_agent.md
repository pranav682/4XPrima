# strategy_lab_agent

## Purpose

Propose candidate strategy *specifications* as structured objects that the backtester can construct and run verbatim. The agent **does not run code**, backtest, optimize, execute, deploy, or self-approve anything; it writes bounded specs the deterministic engine will execute and the critic will judge.

> **Build status (Stage 3).** This spec is implemented in the **registry-based, constructible** form below: a candidate names an archetype from the fixed `core.strategy.STRATEGY_REGISTRY` and supplies concrete parameters + bounded `parameter_ranges`. The richer free-form **Rule-AST** grammar sketched later in this doc (typed entry/exit `Rule` nodes, `IndicatorSpec`, `StopRule`/`TargetRule`, session filters) is a **future extension** — it is deferred until the backtest engine can construct arbitrary rule ASTs. Today the engine only constructs registry archetypes (MA crossover), and the agent may propose only what the engine can build. Keep the registry and this spec in lockstep: add an archetype to the code registry first, then widen the catalog here.

## Responsibilities

- Read `MarketContextReport` and the current champion's recent performance.
- Generate one or more `StrategyCandidate`s consistent with the regime tags.
- Tag each candidate with hypothesis ("trend continuation in EURUSD during Asia session"), indicator stack, timeframe, and stop/target rules.
- Avoid duplicates of recent candidates in the same run window (orchestrator passes a dedupe set).

## Inputs (with types) — as built

```python
@dataclass(frozen=True)
class StrategyLabRequest:          # core.agents.strategy_lab_agent
    run_id: str
    market_context: MarketContextReport
    allowed_universe: tuple[str, ...]        # screened pairs, passed in (not hardcoded)
    allowed_timeframes: tuple[Granularity, ...]  # e.g. (H1, H4, D)
    n_candidates: int = 3                    # propose 1..n, never more
    tier: ModelTier = ModelTier.DEFAULT      # HEAVY for novel-design sessions
```

*Deferred to the orchestrator stage (not Stage 3):* `champion_summary` and a
`recent_candidate_hashes` dedupe set. The orchestrator will pass these in and
re-check dedupe; the agent need not until then.

## Outputs (strict structured schema) — as built

```python
class StrategyProposal(BaseModel):           # core.models, frozen, extra=forbid
    run_id: str
    as_of: datetime
    schema_version: str = "1.0"
    candidates: tuple[StrategyCandidate, ...]  # length 1..n_candidates

class StrategyCandidate(BaseModel):          # core.models, frozen, extra=forbid
    candidate_id: str
    run_id: str
    archetype: StrategyArchetype             # ONLY from core.strategy.STRATEGY_REGISTRY
    instrument: str                          # six-letter pair from allowed_universe
    timeframe: Granularity
    parameters: tuple[StrategyParam, ...]    # {name, value} per archetype param
    parameter_ranges: tuple[ParamRange, ...] # {name, low, high} bounded sandbox
    rationale: str                           # ≤ 500 chars, tied to the context
```

Notes:
- `parameters` / `parameter_ranges` are **tuples of named entries**, not open
  dicts, so the OpenAI Structured-Outputs schema stays strict-mode clean.
- There is **NO execution / deployment / live-trading field** anywhere; the
  `rationale` is scrubbed for execution intent at construction AND re-checked
  in Tier-1 (defence-in-depth, like `market_context`'s no-trade-call check).
- Every candidate must be **constructible** into a real `core.strategy.Strategy`
  via `core.strategy.build_strategy(candidate)` before the output is accepted.

The free-form `Rule` AST (typed entry/exit nodes) is the future extension noted
under Purpose — not part of the Stage-3 build.

## Tools & Skills used

**Tools:** none at runtime. All inputs are passed in. (Keeps the cache hot and forbids the agent from fetching new market data — that's `market_context_agent`'s job.)

**Skills:**
- `backtesting-methodology` — the rule grammar the backtester accepts.
- `forex-cost-modeling` — so candidates aren't proposed at timeframes where spread dominates edge.
- `prompt-caching-and-token-budget`.

## Model tier and why

**DEFAULT tier** (`ModelTier.DEFAULT` → `gpt-5.4`, the Sonnet-equivalent) by default. Promote to the **HEAVY tier** (`ModelTier.HEAVY` → `gpt-5.5`) only for explicit "novel strategy design" sessions invoked by the orchestrator (set via the `tier` field on `StrategyLabRequest`). The heavy tier is not justified for routine candidate generation.

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
