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
4. **Never commit secrets, and never echo or interpolate real secret values into commands, logs, or commit messages.** All API keys, broker creds, etc. live in `.env` (gitignored). Use `pydantic-settings` to load them (`SecretStr`). See "Security hygiene" below for the rule and the scan routine.
5. **Every strategy/parameter change must pass walk-forward + out-of-sample validation AND the critic** before it can be proposed for deployment. No exceptions, no "small" tweaks.

## Security hygiene

> **The rule.** Never echo or interpolate a real secret value into a command, a log line, a commit message, a test fixture, a docstring, or any file or output. The bash history, your terminal scrollback, and any captured tool output count as outputs — once a secret lands there, it has leaked.

What this means in practice:

- **Safety scans are PATTERN-based, never literal-value-based.** A grep like `grep -E 'sk-svcacct-G-clgxF6...' tracked_files/` writes the real key into shell history and into any output that captures the command. The correct check uses a regex pattern that matches the *shape* of the token, not the value: `grep -E 'sk-(ant-|svcacct-|proj-)?[A-Za-z0-9_-]{20,}'`. Run `scripts/safety_scan.sh` — it does this for OpenAI, Anthropic, GitHub, AWS, Slack, GCP key, and JWT formats, plus asserts `.env` is gitignored and untracked.
- **Reports never include matched text.** When the scan finds a token-shaped string, it prints `file:line` only — never the line itself — so even the *failure* output is safe.
- **`SecretStr` everywhere at the boundary.** Pydantic-settings models wrap every key in `SecretStr` so accidental `repr()` / `str()` / structlog dumps print `**********`, not the value. Call `.get_secret_value()` only at the single point of HTTP construction and never store the result.
- **Errors never re-include URLs or headers.** Provider wrappers (OANDA, FRED, OpenAI, etc.) catch SDK errors and re-raise with the error TYPE + safe message, never the request URL (which can carry the key in a query string) or the request headers.
- **Prefer a real scanner as a pre-commit hook.** `gitleaks` or `trufflehog` catch entropy-based secrets that pattern matching misses. `scripts/safety_scan.sh` calls `gitleaks detect --no-banner --redact` automatically when it's installed. Install: `brew install gitleaks`.

If you find yourself about to type a literal token into a command line, STOP. Load it from `.env` (`set -a; . ./.env; set +a`) and reference the variable name only.

## Coding conventions

- **Python 3.11+** (Agent SDK requires ≥3.10; we go 3.11 for `Self`, exception groups, perf).
- **Full type hints** on every function and class. Enforced via the quality gate below (`mypy --strict`).
- **Formatting:** `black`. **Linting:** `ruff` (includes import sort, pyupgrade, bugbear, etc.). Both enforced via the quality gate below.
- **Tests:** `pytest` with `pytest-asyncio`. Backtest math has unit tests; agent specs have schema tests.
- **Logging:** structured (JSON) via `structlog`. Every LLM call logs `input_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `output_tokens`, `model`, `agent_name`, `run_id`.
- **Config:** `pydantic-settings` reading from environment / `.env`. No hardcoded paths, hosts, or limits in code.
- **Data:** time series are timezone-aware UTC; prices are `Decimal` at boundaries, `float64` numpy inside hot loops with explicit narrowing.

## Quality gate (`scripts/check.sh`) — a ratchet, not a blanket claim

**Run `scripts/check.sh` before every commit.** It runs `ruff`, `black --check`, and `mypy --strict` over **one shared enforced-clean set** and fails if any tool errors on any of those files. None of the three is repo-wide clean and never has been (don't claim otherwise) — we enforce them as a **ratchet** on completed code, all over the *same* file set so the gates can't drift apart.

- **Enforced-clean set** — must stay simultaneously ruff-, black-, and `--strict`-clean; `scripts/check.sh` fails on any new error here:
  `core/models.py`, `core/strategy.py`, `core/backtest/`, `core/analysis/`, `core/agents/strategy_lab_agent.py`, `core/broker.py`, `core/config.py`, `core/usage_accounting.py`.
- **Every NEW module must land clean under all three** and be added to the set (`ENFORCED` in `scripts/check.sh`).
- **A file joins the set only when it is BOTH black-clean AND mypy-clean (and ruff-clean)** — not one or the other.
- **Known debt — NOT yet enforced:** the rest of the fast loop (`core/risk_manager.py`, `core/execution.py`, `core/paper_broker.py`, `core/market_data.py`), the data/LLM adapters (`core/context_data.py`, `core/llm_client.py`), the slow-loop agents (`core/agents/*` except the already-clean `cost.py`/`types.py`), and most of `tests/`. Full inventory (type **and** formatting debt) + ratchet in **`docs/quality-debt.md`**.

As a file is cleaned under all three tools, add it to `ENFORCED` and delete its row from `docs/quality-debt.md`. Do not add a file while it still fails any tool.

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

The codebase is a **single flat package under `core/`** — there are no top-level `execution/`, `strategies/`, `risk/`, `backtest/`, or `agents/` directories. The fast/slow split is a **code** boundary, not a directory one: the fast-loop modules are pure deterministic Python and **never import an LLM client**. That rule (invariant #1) is enforced by the code itself and its tests, not by where files sit. Keep new modules under `core/` and preserve the no-LLM-in-the-fast-path boundary regardless of layout.

| Concern | Lives in | Loop / role | Why |
| --- | --- | --- | --- |
| Live price feed, broker adapter, order router | `core/market_data.py`, `core/broker.py`, `core/paper_broker.py`, `core/execution.py` | fast | Must run without LLM. |
| Strategy signal evaluation at runtime | `core/strategy.py` | fast | Deterministic, fast, testable. |
| Risk manager + kill switch | `core/risk_manager.py` | fast | Sacred — see invariant #2. |
| Backtest engine | `core/backtest/` | deterministic | Heavy compute; called BY the slow loop, not part of it. |
| Agent prompts, agent orchestration, reports | `core/agents/` | slow | LLM-driven, scheduled, proposes only. |
| Shared LLM client, usage accounting | `core/llm_client.py`, `core/usage_accounting.py` | slow | Used only by the slow loop. |

## When in doubt

If a change would let an LLM directly affect a live order, **stop and ask the user**. Same for anything that touches the risk limits, the kill switch, or production credentials. Optimization can propose; only a human can deploy.
