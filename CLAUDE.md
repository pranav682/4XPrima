# 4xPrima — Project Memory

> **Read this file at the start of every session.** It is the source of truth for what 4xPrima is, what cannot change, and how to make changes safely.

## What 4xPrima is

A multi-agent algorithmic forex trading system. It is split into two **decoupled** loops:

- **Fast loop** — deterministic Python. Market-data ingestion → strategy signal generation → risk manager → order execution. **No LLM is ever in this path.** It must run, fail safe, and respect the kill switch on its own.
- **Slow loop** — the LLM agent system, run on a schedule. Produces **proposals and reports only**: market context, strategy candidates, backtests, parameter sweeps, adversarial critique, orchestration, and a human-facing reporting agent. It cannot place orders.

The reporting agent + explicit approval gates are the bridge. Anything that changes live risk requires a human "yes".

## Hard invariants (do not violate, ever)

1. **No LLM call is in the live trade-execution path.** Signals reach execution only through deterministic code.
2. **Risk limits and the kill switch are sacred.** The optimization/improvement loop **cannot** override them — they are enforced by deterministic code that the slow loop does not configure.
3. **Paper trading only** until a human explicitly approves live trading. Live credentials must be physically absent from the default environment.
4. **Never commit secrets.** All API keys, broker creds, etc. live in `.env` (gitignored). Use `pydantic-settings` to load them.
5. **Every strategy/parameter change must pass walk-forward + out-of-sample validation AND the critic** before it can be proposed for deployment. No exceptions, no "small" tweaks.

## Coding conventions

- **Python 3.11+** (Agent SDK requires ≥3.10; we go 3.11 for `Self`, exception groups, perf).
- **Full type hints** on every function and class. `mypy --strict` clean is the goal.
- **Formatting:** `black`. **Linting:** `ruff` (includes import sort, pyupgrade, bugbear, etc.).
- **Tests:** `pytest` with `pytest-asyncio`. Backtest math has unit tests; agent specs have schema tests.
- **Logging:** structured (JSON) via `structlog`. Every LLM call logs `input_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `output_tokens`, `model`, `agent_name`, `run_id`.
- **Config:** `pydantic-settings` reading from environment / `.env`. No hardcoded paths, hosts, or limits in code.
- **Data:** time series are timezone-aware UTC; prices are `Decimal` at boundaries, `float64` numpy inside hot loops with explicit narrowing.

## Required reading before implementing anything

- **Before implementing an agent:** read its spec in `specs/agents/<name>.md` AND every skill it references in `skills/`. The spec defines its inputs, outputs, tools, model tier, and guardrails. Don't add fields or tools the spec doesn't list — update the spec first.
- **Before adding any LLM call:** read `docs/llm-conventions.md`. It is the single source of truth for prompt caching, context management, model routing, and token accounting.

## The LLM client rule

**ALL runtime LLM calls MUST go through `core/llm_client.py`.** Never `import anthropic` and call `client.messages.create(...)` directly anywhere else. The shared client is what enforces:

- the stable cacheable-prefix message layout,
- model tier routing,
- context-management / tool-search settings,
- per-agent token + cache-hit accounting.

A direct SDK call bypasses all of that and breaks the caching economics.

## Two-loop boundaries (where things live)

| Concern | Lives in | Why |
| --- | --- | --- |
| Live price feed, broker adapter, order router | `execution/` (fast loop) | Must run without LLM. |
| Strategy signal evaluation at runtime | `strategies/` (fast loop) | Deterministic, fast, testable. |
| Risk manager + kill switch | `risk/` (fast loop) | Sacred — see invariant #2. |
| Backtest engine | `backtest/` (deterministic) | Heavy compute; called BY the slow loop, not part of it. |
| Agent prompts, agent orchestration, reports | `agents/` (slow loop) | LLM-driven, scheduled, proposes only. |
| Shared LLM client, usage accounting | `core/` | Used only by the slow loop. |

## When in doubt

If a change would let an LLM directly affect a live order, **stop and ask the user**. Same for anything that touches the risk limits, the kill switch, or production credentials. Optimization can propose; only a human can deploy.
