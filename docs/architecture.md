# 4xPrima — Architecture

## Two loops, one bridge

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          FAST LOOP (deterministic Python, NO LLM)            │
│                                                                              │
│  market data ──► strategy engine ──► risk manager ──► order router ──► broker│
│   (feed)         (signal eval)        (caps + kill)    (paper or live)       │
│        ▲                                   │                                 │
│        │                                   ▼ (denied / approved + audit log) │
│  config & versioned strategy artefacts  ──── kill switch ◄──── human / file  │
└──────────────────────────────────────────────────────────────────────────────┘
                                ▲                                 │
                                │ (champion / challenger swap,    │ (status,
                                │  via human-approved deploy)     │  fills,
                                │                                 │  P&L)
                                │                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       SLOW LOOP (LLM agents, proposes only)                  │
│                                                                              │
│   market_context_agent ──► strategy_lab_agent ──► backtest_agent             │
│            │                       │                    │                    │
│            └──────────┬────────────┘                    │                    │
│                       ▼                                 ▼                    │
│                 orchestrator_agent  ◄────  optimization_agent                │
│                       │                                 │                    │
│                       ▼                                 ▼                    │
│                  critic_agent  (Opus, adversarial — tries to kill proposals) │
│                       │                                                      │
│                       ▼                                                      │
│                 reporting_agent  ─── conversation ───►  ┌──────────────────┐ │
│                                                          │  HUMAN OPERATOR │ │
│                                                          └─────────┬────────┘ │
│                                                                    │          │
└────────────────────────────────────────────────────────────────────┼──────────┘
                                                                     │
                                          explicit, audited approval ▼
                                              (champion swap, live toggle, etc.)
```

The two loops share **data** (market data, strategy artefacts, fills, P&L) but not **control**. The slow loop reads and proposes; the fast loop executes. The reporting agent + the human-approval gate is the *only* path from a slow-loop proposal to a fast-loop change.

## Fast loop — components and contract

| Component | Responsibility | Hard rule |
| --- | --- | --- |
| **Market data adapter** | Stream/replay normalized OHLC + tick data with UTC timestamps. | One uniform interface; broker-specific code stays behind it. |
| **Strategy engine** | Load versioned strategy artefacts; on each bar/tick, compute signals. | Pure functions over typed inputs; no I/O inside signal eval. |
| **Risk manager** (see `specs/components/risk_manager.md`) | Position sizing; per-trade, per-symbol, portfolio risk caps; drawdown cap; correlation cap; kill switch. | **Deterministic code, no LLM.** The slow loop **cannot** raise its limits. |
| **Order router** | Send orders to paper or live broker; reconcile fills; persist an audit log. | Default is paper. Live requires explicit env flag + human approval. |
| **Audit log** | Append-only, JSON-line, signed sequence of (signal → risk decision → order → fill). | Read by the slow loop; never written by it. |

The fast loop runs as a long-lived process. If anything in it throws, the kill switch flips, open positions are flattened to a safe state per the risk manager's runbook, and the process exits with a non-zero status that the supervisor picks up. **A crash in the slow loop has no effect on the fast loop.**

## Slow loop — agent roster

Each agent has its own spec in `specs/agents/<name>.md`. Summary here only — go to the spec before implementing.

| Agent | Role | Model tier | Consumes | Produces |
| --- | --- | --- | --- | --- |
| **market_context_agent** | Read macro data, news, the economic calendar, sentiment. | Sonnet | raw feeds (via tools) | `MarketContextReport` (structured) |
| **strategy_lab_agent** | Propose candidate strategy specs (entry/exit, indicators, timeframe). | Sonnet (Opus for novel design sessions) | `MarketContextReport`, champion stats | `StrategyCandidate` (structured) |
| **backtest_agent** | Configure backtests; interpret results. Heavy compute is in deterministic `backtest/` code. | Sonnet | `StrategyCandidate`, backtest runner output | `BacktestInterpretation` (structured) |
| **optimization_agent** | Propose parameter changes **within bounded ranges only**. Every proposal is validation-gated. | Sonnet | `BacktestInterpretation`, parameter bounds | `ParameterProposal` (structured) |
| **critic_agent** | Adversarial. Tries to kill each proposal via the overfitting checklist. | **Opus** | every proposal | `CriticVerdict` (`accept` / `reject` + reasons) |
| **orchestrator_agent** | Schedule the slow loop; route work; hold run state; manage champion/challenger. | Sonnet (Haiku for routing-only ticks) | all agent outputs + run state | `SlowLoopRunState` + dispatch decisions |
| **reporting_agent** | Human-facing. Summarize P&L, explain proposed changes, mediate approval. | Sonnet | run state + critic verdicts | conversational report + approval requests |

## Hand-offs

A normal slow-loop cycle:

1. **orchestrator** wakes (cron / schedule), reads fast-loop audit log + current champion stats, decides whether to run a full cycle or just a status check.
2. **market_context** runs → `MarketContextReport`.
3. **strategy_lab** receives the report → `StrategyCandidate`s.
4. **backtest_agent** drives the backtester for each candidate → `BacktestInterpretation`s.
5. **optimization** may propose tweaks to the current champion or to a candidate → `ParameterProposal`s.
6. **critic** sees every proposal (candidate + tweak) and renders a verdict.
7. **orchestrator** updates run state and selects what to *propose* to the human.
8. **reporting** generates the human-facing summary and opens an approval request.
9. Human approves (or rejects) → if approved, a new versioned strategy artefact is committed; fast loop picks it up at its next safe rollover.

Nothing in steps 1–8 changes the running system.

## Approval gates (the bridge)

There are three approval gates between the slow loop and the live system:

1. **Champion swap** — any change to the strategy currently authorising fast-loop signals.
2. **Parameter change** — any change to a parameter inside an existing champion.
3. **Mode change** — any flip of `TRADING_MODE` (paper ↔ live) or `KILL_SWITCH_OVERRIDE`.

Every gate requires:

- a `ParameterProposal` or `StrategyCandidate` that passed walk-forward + out-of-sample,
- a `CriticVerdict` of `accept` (Opus-evaluated),
- a `reporting_agent` summary the human reads,
- an explicit human "approve" action (CLI command or signed file write — not a chat reply),
- an audit-log entry recording who approved and what artefact was deployed.

The slow loop **does not have credentials** to flip any of these. The approval action is what writes the new artefact / flag.

## Data flow at a glance

- **Read by slow loop:** market data archive, fast-loop audit log, backtest reports, prior agent outputs.
- **Written by slow loop:** agent run records, proposals, critic verdicts, reporting transcripts, usage-accounting log.
- **Read by fast loop:** market data, strategy artefacts (versioned), risk config.
- **Written by fast loop:** audit log, fill stream, P&L stream.

Strategy artefacts and risk config are the only objects both loops touch, and **only the fast loop reads them at runtime**; the slow loop merely *proposes* new versions for the human to deploy.

## Venue / instrument layer must stay swappable

Both the price provider (`core.market_data.PriceProvider`) and the broker (`core.broker.Broker`) are Protocols. The OANDA practice adapter is the *first* implementation, not the contract — the eventual live venue can and likely will be a different broker (e.g. a SEBI-registered Indian exchange-traded-currency broker, to match operator residency). Design constraint: no execution-path code may import an OANDA-specific symbol; all venue specifics live behind the Protocols.
