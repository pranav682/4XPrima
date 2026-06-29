# Quality debt inventory (`ruff` + `black` + `mypy --strict`)

> **Status (2026-06-28): 96 `mypy --strict` errors across 19 files, and 18
> `black`-dirty files.** This is tracked debt, not a standard we pretend to
> meet. None of the three tools is repo-wide clean. We enforce them as a
> **ratchet** over ONE shared enforced-clean set (`scripts/check.sh`), and each
> file is promoted into that set only when it is **simultaneously ruff-clean,
> black-clean, and mypy --strict-clean.**

See `CLAUDE.md` → "Quality gate" for the policy. The enforced-clean set today is
`core/models.py`, `core/strategy.py`, `core/backtest/`, `core/analysis/`,
`core/agents/strategy_lab_agent.py`, `core/agents/backtest_harness.py`,
`core/agents/backtest_agent.py`, `core/agents/critic_agent.py`,
`core/broker.py`, `core/config.py`, `core/usage_accounting.py` — clean under all
three tools, and `scripts/check.sh` keeps it that way. (The four `core/agents/*`
entries are single-file promotions from an otherwise debt-laden `core/agents/`
package.)

## How to read this

- **mypy** column = current `--strict` error count for that file.
- **black** column = `clean` / `dirty` under `black --check`.
- **Promote** = get the file to ruff-clean + black-clean + mypy-clean, add its
  path to `ENFORCED` in `scripts/check.sh`, and delete its row here.
- A file with `0` mypy errors that is still `black: dirty` is NOT promotable yet
  — it must pass *both*. (`core/models.py` was exactly this case and was
  reformatted into the enforced set.)
- Don't sink time into the slow-loop agent / LLM / throwaway-test rows that are
  slated for rewrite in Stage 3 — clean the stable deterministic code first.

## Debt by area

ruff is repo-wide clean today, so it isn't broken out per file below; the
columns that vary are mypy and black.

### Source (`core/`)

| File | mypy errors | black | Notes / dominant fixes | Ratchet priority |
| --- | --- | --- | --- | --- |
| `risk_manager.py` | 1 | dirty | `no-any-return` (untyped `structlog`) + reformat | **High** — stable fast-loop code; ~1 cast + black. |
| `execution.py` | 1 | dirty | `no-any-return` + reformat | **High** — same shape. |
| `paper_broker.py` | 1 | dirty | `no-any-return` + reformat | **High** — same shape. |
| `market_data.py` | 7 | dirty | `index`/`arg-type` (provider JSON shapes) + reformat | Medium — needs typed response models. |
| `context_data.py` | 4 | dirty | `arg-type`/`index` (provider JSON) + reformat | Medium — same. |
| `llm_client.py` | 2 | dirty | `no-any-return` (SDK) + reformat | Medium — pairs with agent cleanup. |
| `agents/market_context_agent.py` | 5 | dirty | `arg-type`, stale `unused-ignore` + reformat | **Low** — reworked in Stage 3. |
| `agents/runner.py` | 2 | dirty | `no-any-return` + reformat | **Low** — Stage 3. |
| `agents/evaluation.py` | 1 | dirty | `no-any-return` + reformat | **Low** — Stage 3. |

`core/agents/cost.py` and `core/agents/types.py` are already clean under all
three tools but live in a debt-laden package; promote them when the rest of
`agents/` is cleaned in Stage 3.

### Tests (`tests/`)

| File | mypy errors | black | Notes | Ratchet priority |
| --- | --- | --- | --- | --- |
| `test_backtest_engine.py` | 10 | clean | `no-untyped-def` (untyped fixtures), `arg-type` (`_config(**dict)`) | **Medium** — covers already-enforced Stage 2 source; type it to match. |
| `test_backtest_costs.py` | 10 | clean | `no-untyped-def` (untyped fixtures) | **Medium** — same. |
| `test_context_data.py` | 14 | dirty | `no-untyped-def`, `arg-type` | Low |
| `test_oanda_provider.py` | 11 | dirty | `no-untyped-def`, `index` | Low |
| `test_market_context_agent.py` | 9 | dirty | `no-untyped-def`, `call-arg` | Low — Stage 3. |
| `test_paper_broker.py` | 5 | dirty | `no-untyped-def` | Low |
| `test_llm_client.py` | 5 | dirty | `no-untyped-def`, `assignment` | Low |
| `test_risk_manager.py` | 4 | dirty | `no-untyped-def` | Low |
| `test_agent_evaluation.py` | 3 | dirty | `no-untyped-def` | Low — Stage 3. |
| `test_agent_runner.py` | 1 | dirty | `no-untyped-def` | Low — Stage 3. |
| `test_execution.py` | 0 | dirty | black-only debt (reformat) | Low |

## mypy error-type totals (whole repo, for reference)

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

1. **Fast-loop deterministic** (`risk_manager`, `execution`, `paper_broker`) — ~1 mypy cast each + `black`; promote to enforced.
2. **Stage 2 test files** (`test_backtest_engine`, `test_backtest_costs`) — already black-clean; type the fixtures so the tests meet the source bar.
3. **Market-data + context adapters** (+ their tests) — once typed response models land.
4. **LLM client + slow-loop agents** (+ their tests) — fold into the Stage 3 agent rework, not before.
