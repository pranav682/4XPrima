# Type debt inventory (`mypy --strict`)

> **Status: 96 errors across 19 files (2026-06-28).** This is tracked debt, not a
> standard we pretend to meet. `mypy --strict` is **not** repo-wide clean. We
> enforce it as a **ratchet**: a fixed enforced-clean set that CI/pre-commit
> won't let regress (`scripts/typecheck.sh`), and each package below gets
> cleaned and promoted into that set as its code stabilises.

See `CLAUDE.md` → "Type checking" for the policy. The enforced-clean set today is
`core/models.py`, `core/strategy.py`, `core/backtest/`, `core/broker.py`,
`core/config.py`, `core/usage_accounting.py` — **0 errors, must stay 0.**

## How to read this

- Each row = a package/area with its current `--strict` error count.
- **Promote** = clean it to 0, add its paths to `ENFORCED` in
  `scripts/typecheck.sh`, and delete its row here.
- Don't sink time into the slow-loop agent / LLM / test rows that are slated for
  rewrite in Stage 3 — typing throwaway code is wasted work. Clean the stable
  deterministic code first.

## Debt by area

### Source (`core/`) — 24 errors

| Area | Files (errors) | Dominant error types | Ratchet priority |
| --- | --- | --- | --- |
| Fast-loop deterministic | `risk_manager.py` (1), `execution.py` (1), `paper_broker.py` (1) | `no-any-return` (untyped `structlog.get_logger`) | **High** — stable code, ~3 one-line `cast`s; promote next. |
| Market-data + context adapters | `market_data.py` (7), `context_data.py` (4) | `index`, `arg-type`, `no-any-return` (provider JSON shapes) | Medium — stable, but needs typed response models. |
| LLM client | `llm_client.py` (2) | `no-any-return` | Medium — small; pairs with the agent cleanup. |
| Slow-loop agents | `agents/market_context_agent.py` (5), `agents/runner.py` (2), `agents/evaluation.py` (1) | `arg-type`, `unused-ignore`, `no-any-return` | **Low** — being reworked in Stage 3; clean as part of that. `agents/cost.py` and `agents/types.py` are already clean. |

### Tests (`tests/`) — 72 errors

| Files (errors) | Dominant error types | Ratchet priority |
| --- | --- | --- |
| `test_context_data.py` (14), `test_oanda_provider.py` (11), `test_market_context_agent.py` (9), `test_paper_broker.py` (5), `test_llm_client.py` (5), `test_risk_manager.py` (4), `test_agent_evaluation.py` (3), `test_agent_runner.py` (1) | `no-untyped-def` (untyped fixture params / helpers), `arg-type`, `call-arg` | **Low** — test typing follows its module; clean alongside each promoted package. |
| `test_backtest_engine.py` (10), `test_backtest_costs.py` (10) | `no-untyped-def` (untyped fixtures `zero_cost_model` etc.), `arg-type` (`_config(**dict)`) | **Medium** — these cover already-enforced source; worth typing so the Stage 2 tests match the Stage 2 source bar. |

## Error-type totals (whole repo, for reference)

```
24  no-untyped-def     untyped function/param (mostly test fixtures + helpers)
20  arg-type           wrong arg type (provider JSON, **dict unpacking)
14  index              indexing a value of wrong/loose type (provider parsing)
13  no-any-return      returning Any (untyped structlog / SDK calls)
 5  unused-ignore      stale `# type: ignore`
 5  operator           op on loosely-typed value
 5  call-arg           missing/extra call argument
 3  type-arg           missing generic type args
 3  no-untyped-call    calling an untyped function from typed code
 2  attr-defined       attribute not known on type
 1  unreachable        narrowing false-positive (debt module)
 1  assignment         incompatible assignment (test)
```

## Suggested ratchet order

1. **Fast-loop deterministic** (`risk_manager`, `execution`, `paper_broker`) — 3 trivial fixes; promote to enforced.
2. **Stage 2 test files** (`test_backtest_engine`, `test_backtest_costs`) — type the fixtures so the tests meet the source bar.
3. **Market-data + context adapters** (+ their tests) — once typed response models land.
4. **LLM client + slow-loop agents** — fold into the Stage 3 agent rework, not before.
