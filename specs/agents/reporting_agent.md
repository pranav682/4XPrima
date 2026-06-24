# reporting_agent

## Purpose

The **only** human-facing surface. Summarise P&L and risk events in plain language; explain what the system wants to change and **why**; present approval decisions clearly. Never carries authority — the human's reply is the deciding action.

## Responsibilities

- Render the latest fast-loop status (P&L, drawdown, open positions, any kill-switch event) in plain English.
- Walk through pending `ApprovalRequest`s: what is being proposed, what the critic said, what the backtests showed (in plain language, with links to the structured artefacts for audit).
- Explain trade-offs honestly: "this proposal raises Sharpe but increases max drawdown by X%".
- Surface failures, `needs_more_evidence` verdicts, and stale challengers.
- Format the response so the human can act in one of a small, named set of ways: `approve <approval_id>`, `reject <approval_id> [reason]`, `defer <approval_id>`, `ask <approval_id> <question>`.

## Inputs (with types)

```python
class ReportingRequest:
    run_id: str
    audience: Literal["operator_daily", "operator_event", "operator_approval"]
    audit_excerpt: AuditExcerpt
    champion_summary: ChampionSummary
    pending_approvals: list[ApprovalRequest]    # the proposals waiting for a human
    recent_failures: list[AgentFailure]
    prior_human_messages: list[HumanTurn]       # for continuity, ≤ N turns
```

## Outputs (strict structured schema)

```python
class ReportingOutput:
    run_id: str
    schema_version: str = "1.0"
    summary_markdown: str                       # ≤ 4k chars, neutral, plain language
    sections: list[ReportSection]               # ordered: status, proposals, asks, footnotes
    approval_surface: list[ApprovalAction]      # the named actions the human may take
    artefact_links: list[ArtefactLink]          # paths under data/runs/<run_id>/
    open_questions: list[str]                   # things the human should clarify
```

The `summary_markdown` is what gets shown; `approval_surface` is the structured action vocabulary that the approval CLI/UI parses out of the human's reply.

## Tools & Skills used

**Tools (eager):**
- `read_artefact` — pull a `BacktestInterpretation`, `CriticVerdict`, etc. by id, read-only.
- `format_pnl_table` — deterministic helper, not an LLM call.

**Tools (deferred):**
- Plot-rendering tool — only if the human asks for a chart.

**Skills:**
- `market-context-format` — to summarise context faithfully (read, don't reinterpret).
- `risk-management-rules` — for kill-switch / risk vocabulary.
- `prompt-caching-and-token-budget`.

## Model tier and why

**Sonnet** (`claude-sonnet-4-6`) for approval messages and event summaries (clarity matters). **Haiku** (`claude-haiku-4-5-20251001`) for routine daily status summaries — cheap and adequate when the only job is to format the audit table.

Selected by `audience` field.

## System-prompt structure

```
[tools]                                          ← stable per tier
[system]
  - identity: "human-facing reporter; honest about trade-offs; you do not approve" ← stable
  - approval action vocabulary (approve/reject/defer/ask)         ← stable
  - reporting style guide (tone, length, no emojis unless asked)  ← stable
  - market-context-format skill                  ← stable
  - risk-management-rules skill (read-only)      ← stable
  -------------- cache_control breakpoint ---------
  - none
[messages]
  - user: { request, prior human messages, pending approvals (full), failures, audit excerpt }
```

Conversation continuity: pass prior human turns in `messages` (after the breakpoint). Do **not** mutate `system` between turns — see `docs/llm-conventions.md` §5.

## Token budget

- **Routine status (Haiku):** ≤ 4k in / ≤ 1.2k out.
- **Approval message (Sonnet):** ≤ 12k in (proposals + critic verdicts can be hefty) / ≤ 3k out.
- Hard cap: 20k in / 6k out.

## Guardrails

- **Must not** state recommendations as obligations ("you should approve…") — surface trade-offs and let the human decide.
- **Must not** approve or reject a proposal itself — `approval_surface` is the only mechanism, and it requires a human reply.
- **Must not** invent numbers — every metric cites an `ArtefactLink`.
- **Must not** alter or paraphrase the `CriticVerdict.kill_reasons` — quote them.
- **Must not** include trade calls or directional advice — the slow loop *proposes systems*, not trades.
- **Must not** soften critic rejections to be more palatable.

## Hand-offs

- Human reply → parsed by approval CLI/UI → if `approve`, the named action runs (deterministic code writes the new strategy artefact / flips the env flag) → fast loop picks up at next safe rollover.
- Conversation transcript persisted under `data/runs/<run_id>/reports/<timestamp>.json`.
