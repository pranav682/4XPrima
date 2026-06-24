# 4xPrima — Plan

This document is the index for the staged build-out. **No trading logic yet.** This pass is specs, skills, and typed stubs.

## Stage 0 — Context & standards (this pass)

Goal: every later pass starts from a complete, written-down spec for what to build and how.

- [x] `CLAUDE.md` — project memory, invariants, conventions.
- [x] `docs/architecture.md` — two-loop architecture, agent roster, hand-offs, approval gates.
- [x] `docs/llm-conventions.md` — single source of truth for prompt caching, context management, tool search, model routing, Agent SDK vs custom loop.
- [x] `specs/agents/*.md` — one spec per slow-loop agent (template fixed in `CLAUDE.md`).
  - `market_context_agent.md`
  - `strategy_lab_agent.md`
  - `backtest_agent.md`
  - `optimization_agent.md`
  - `critic_agent.md`
  - `orchestrator_agent.md`
  - `reporting_agent.md`
- [x] `specs/components/risk_manager.md` — risk manager is **deterministic code**, not an agent.
- [x] `skills/*/SKILL.md` — referenced from agent specs.
  - `backtesting-methodology`
  - `forex-cost-modeling`
  - `overfitting-checklist`
  - `risk-management-rules`
  - `market-context-format`
  - `prompt-caching-and-token-budget`
- [x] `core/llm_client.py` — shared LLM wrapper stub (no real calls yet).
- [x] `core/usage_accounting.py` — per-agent token + cache-hit log stub.
- [x] `pyproject.toml`, `.gitignore`, `.env.example` — project scaffolding.

## Stage 1 — Deterministic fast loop (next)

Implement the components that must work without any LLM, in this order:

1. Market data adapter (paper feed first; broker SDK behind an interface).
2. Strategy interface + one reference strategy (moving-average crossover, for end-to-end testing).
3. `risk/` — position sizing, per-trade cap, portfolio cap, drawdown cap, correlation cap, kill switch.
4. Paper-trading order router with an audit log.
5. End-to-end "tick → signal → risk gate → paper order" smoke test.

**Exit criteria:** fast loop runs unattended on paper for ≥24h with zero exceptions and a complete audit trail.

## Stage 2 — Backtester (deterministic)

1. Bar/tick replay engine with realistic spread/slippage/commission/swap (see `skills/forex-cost-modeling`).
2. Walk-forward + out-of-sample split utilities (see `skills/backtesting-methodology`).
3. Standard report format consumed by `backtest_agent`.

**Exit criteria:** identical reference strategy produces identical metrics across two seeded runs; walk-forward report passes schema validation.

## Stage 3 — Slow loop agents

Build agents in dependency order, all going through `core/llm_client.py`:

1. `market_context_agent` (no deps on other agents).
2. `strategy_lab_agent` (consumes market context).
3. `backtest_agent` (drives the Stage 2 backtester).
4. `optimization_agent` (consumes backtest output, bounded ranges only).
5. `critic_agent` (Opus; tries to kill every proposal — see `skills/overfitting-checklist`).
6. `orchestrator_agent` (champion/challenger state machine).
7. `reporting_agent` (human-facing; mediates the approval gate).

Each agent gets its prompt assembly, schema tests, and a token-budget regression test before it ships.

## Stage 4 — Approval gates & deployment

1. Human-approval CLI/UI surface (no agent can self-promote a strategy).
2. Champion/challenger promotion mechanics in the fast loop (versioned strategy artefacts).
3. Live-trading switch — explicit, multi-step, audit-logged, off by default.

## Live trading

**Not on the plan.** Live trading requires an explicit, dated approval by the user that flips a clearly-named env flag, and at that point we revisit risk limits, broker creds handling, and incident response separately.
