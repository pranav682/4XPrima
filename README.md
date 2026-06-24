# 4xPrima

Multi-agent algorithmic forex trading system. Two decoupled loops:

- **Fast loop** — deterministic Python, no LLM in path: market data → strategy → risk manager → execution.
- **Slow loop** — LLM agents producing proposals and reports; human approval required for any change to live risk.

Paper-trading only until a human explicitly approves live trading.

See [`CLAUDE.md`](./CLAUDE.md) for the project memory and hard invariants, [`PLAN.md`](./PLAN.md) for the staged plan, and [`docs/architecture.md`](./docs/architecture.md) for the architecture.
