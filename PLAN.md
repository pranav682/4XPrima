# 4xPrima — Plan

This document is the index for the staged build-out and the **source of truth for stage numbering**.

## Stage numbering (canonical)

The stage numbers in THIS file are canonical:

> Stage 0 context · Stage 1 fast loop · **Stage 2 backtester** · **Stage 3 slow-loop agents** · Stage 4 approval gates.

**Build-order note.** Work shipped out of numeric order, and two commits are mislabelled. Commits `6ceba41` and `d457989` are titled "Stage 2 part 1/2" but they built the **slow-loop agents**, which are **Stage 3** here. The in-code references already use this canonical scheme (`core/strategy.py` calls the strategy lab "Stage 3 part 2"). The deterministic **backtester (Stage 2)** was actually started *after* the first agent and is still uncommitted. Read the "Stage 2" commit labels as "Stage 3 part 1". Git history is immutable; this note is the reconciliation.

## Current status (2026-06-26)

- **Stage 0** — done (committed).
- **Stage 1 (fast loop)** — risk manager, domain models, paper execution/router, OANDA + context-data adapters all committed. The strategy interface + reference strategy (`core/strategy.py`) are now committed and tested as part of the Stage 2 commit; the ≥24h unattended-paper exit criterion is still not demonstrated.
- **Stage 2 (backtester)** — committed and tested under `core/backtest/`: engine, costs, metrics, walk-forward/OOS split run end-to-end through the real `RiskManager`. Trade IDs are now a deterministic per-run counter, so identical inputs ⇒ identical `config_hash` **and** identical `BacktestResult` (verified by test). **Only remaining Stage 2 scope: the standard report module (item 3 below).**
- **Stage 3 (agents)** — `core/llm_client.py`, `market_context_agent`, the agent runner, and the two-tier evaluation gate are committed and tested. Remaining six agents not started.
- **Stage 4** — not started.
- Test suite: **209 passed, 6 skipped** (live-only) — 167 prior + 42 new backtester/strategy tests.

> **Layout note.** The whole codebase lives flat under `core/` (e.g. `core/risk_manager.py`, `core/execution.py`, `core/backtest/`). The boundary table in `CLAUDE.md` and the stale `backtest/` reference in `docs/architecture.md` were reconciled to this flat layout in the Stage 2 commit. The fast/slow split is enforced in code (no LLM imports in fast-loop modules), not by directory nesting.

## Stage 0 — Context & standards

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

## Stage 1 — Deterministic fast loop

Implement the components that must work without any LLM, in this order:

1. [x] Market data adapter — `core/market_data.py` + OANDA v20 read-only adapter; in-process paper broker in `core/paper_broker.py`.
2. [ ] Strategy interface + one reference strategy (MA crossover) — code exists in **uncommitted** `core/strategy.py`; no tests yet.
3. [x] Risk manager — `core/risk_manager.py`: position sizing, per-trade / portfolio / per-pair / correlation caps, drawdown + daily-loss kill switch.
4. [x] Paper-trading order router with an audit log — `core/execution.py` + `core/paper_broker.py`.
5. [ ] End-to-end "tick → signal → risk gate → paper order" smoke test — not yet demonstrated.

**Exit criteria:** fast loop runs unattended on paper for ≥24h with zero exceptions and a complete audit trail. *(Not yet met.)*

## Stage 2 — Backtester (deterministic)

Committed and tested under `core/backtest/` (+ `core/strategy.py`). 42 hermetic tests cover look-ahead, costs, metrics, the risk gate, walk-forward/OOS, and determinism.

1. [x] Bar/tick replay engine with realistic spread/slippage/commission/swap (see `skills/forex-cost-modeling`) — `core/backtest/engine.py` + `costs.py`; routes every order through the real `RiskManager`; deterministic trade IDs.
2. [x] Walk-forward + out-of-sample split utilities (see `skills/backtesting-methodology`) — `core/backtest/walkforward.py` (`DataSplit` token-gated OOS + `walk_forward_windows`).
3. [ ] **Standard report format consumed by `backtest_agent` — NOT started; the only remaining Stage 2 scope.** No `report` module exists yet (docstrings in `engine.py`/`types.py` reference one); deferred to a separate pass.

**Exit criteria:** identical reference strategy produces identical metrics across two seeded runs ✅ (now full `BacktestResult` equality — trade IDs are a deterministic counter, no `uuid4`); walk-forward report passes schema validation ⏳ (pending the report module in item 3).

## Stage 3 — Slow loop agents

Shared infrastructure committed + tested: `core/llm_client.py` (provider-agnostic), `core/agents/runner.py`, and the two-tier evaluation gate (`core/agents/evaluation.py`, `cost.py`).

Build agents in dependency order, all going through `core/llm_client.py`:

1. [x] `market_context_agent` (no deps on other agents) — committed + tested. *Shipped early; mislabelled "Stage 2 part 1" in commit `6ceba41`.*
2. [ ] `strategy_lab_agent` (consumes market context).
3. [ ] `backtest_agent` (drives the Stage 2 backtester).
4. [ ] `optimization_agent` (consumes backtest output, bounded ranges only).
5. [ ] `critic_agent` (Opus; tries to kill every proposal — see `skills/overfitting-checklist`).
6. [ ] `orchestrator_agent` (champion/challenger state machine).
7. [ ] `reporting_agent` (human-facing; mediates the approval gate).

Each agent gets its prompt assembly, schema tests, and a token-budget regression test before it ships.

## Stage 4 — Approval gates & deployment

*(Not started.)*

1. [ ] Human-approval CLI/UI surface (no agent can self-promote a strategy).
2. [ ] Champion/challenger promotion mechanics in the fast loop (versioned strategy artefacts).
3. [ ] Live-trading switch — explicit, multi-step, audit-logged, off by default.

## Live trading

**Not on the plan.** Live trading requires an explicit, dated approval by the user that flips a clearly-named env flag, and at that point we revisit risk limits, broker creds handling, and incident response separately.
