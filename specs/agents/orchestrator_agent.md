# orchestrator_agent

## Purpose

Schedule and route the slow loop. Hold run state across an end-to-end cycle. Manage the **champion / challenger** improvement protocol: who is the current champion, what challengers are in flight, which gates each challenger has cleared.

## Responsibilities

- On schedule (cron or event), decide whether to run a **full cycle** (market context → candidates → backtests → optimization → critic → report) or a **status check** (read fast-loop audit + champion stats only).
- For a full cycle: invoke each agent in order, passing prior outputs as inputs.
- Enforce the dependency: a proposal cannot reach `reporting_agent` without an `accept` from `critic_agent`.
- Maintain `SlowLoopRunState` for the cycle (which candidates were tried, which proposals were generated, which verdicts were rendered, what's left).
- On error from any agent, decide: retry once, abort the cycle, or fall through to reporting with the failure noted.
- **Never** mutate live config or deploy a proposal directly. Outputs an approval-request artefact for the human.

## Inputs (with types)

```python
class OrchestrationTick:
    tick_id: str                             # one per scheduled fire
    now: datetime                            # UTC
    cycle_kind: Literal["full", "status_check", "challenger_resume"]
    prior_run_state: SlowLoopRunState | None # for resume
    fast_loop_audit_excerpt: AuditExcerpt    # last 24h fills, P&L, kill-switch events
    champion: ChampionSummary
    challengers_in_flight: list[ChallengerState]
```

## Outputs (strict structured schema)

```python
class OrchestrationOutcome:
    tick_id: str
    schema_version: str = "1.0"
    run_state: SlowLoopRunState              # updated state, persisted
    actions_taken: list[AgentInvocation]     # which sub-agents were called, with token totals
    pending_human_approvals: list[ApprovalRequest]   # zero or more
    failures: list[AgentFailure]
    next_tick_hint: NextTickHint             # when/why to wake again
```

`SlowLoopRunState` includes: cycle id, phase (`context | lab | backtest | optimize | critique | report | idle`), challenger registry with per-challenger gate progress, and the champion handle.

## Tools & Skills used

**Tools (eager):**
- `invoke_agent` — typed dispatcher to a sub-agent (returns its structured output).
- `read_run_state` / `write_run_state` — persistent state on disk.
- `read_audit_log` — fast-loop audit, read-only.

**Tools (deferred):**
- `invoke_subagent` variants for less-common agents — loaded via Tool Search Tool when needed.

**Skills:**
- `prompt-caching-and-token-budget`.
- `risk-management-rules` (read-only — for vocabulary when summarising kill-switch events).

## Model tier and why

**Sonnet** (`claude-sonnet-4-6`) by default for full-cycle ticks (multi-step routing with state).
**Haiku** (`claude-haiku-4-5-20251001`) for `status_check` ticks (read audit + champion stats, decide whether anything needs a full cycle). Haiku is fine for the easy case and lets us tick frequently for cheap.

The tier is selected per `OrchestrationTick.cycle_kind`. **Switching tiers invalidates the cache** — keep the two tiers as separate cache lanes, do not interleave.

## System-prompt structure

```
[tools]                                          ← stable per tier
[system]
  - identity & objective                         ← stable
  - cycle protocol (phase machine)               ← stable
  - dependency rule: "no proposal to reporting without critic accept" ← stable
  - schema for OrchestrationOutcome              ← stable
  - risk-management-rules skill (read-only)      ← stable
  -------------- cache_control breakpoint ---------
  - none
[messages]
  - user: { tick, prior run_state, audit excerpt, champion, challengers }
[memory] (optional, opt-in)
  - cross-tick scratch via memory_20250818       ← for long champion/challenger arcs
```

For long-running ticks: `context_management` enabled per `docs/llm-conventions.md` §2.

## Token budget

- **Status-check tick (Haiku):** ≤ 6k in / ≤ 1k out.
- **Full cycle tick (Sonnet):** ≤ 25k in (cumulative across agent invocations is tracked separately by `usage_accounting`) / ≤ 4k out.
- Hard cap on a single dispatch: 40k in / 8k out.

## Guardrails

- **Must not** deploy any proposal. It can only file an `ApprovalRequest`.
- **Must not** call sub-agents in a different order than the cycle protocol allows (no skipping the critic).
- **Must not** raise risk limits or alter the kill switch — those are deterministic-code only.
- **Must not** retry a sub-agent more than once per tick; on second failure, mark `AgentFailure` and fall through.
- **Must not** drop a `needs_more_evidence` critic verdict silently — must surface it to reporting.

## Hand-offs

- `OrchestrationOutcome.pending_human_approvals` → `reporting_agent` (which renders the human-facing message and the approval surface).
- `OrchestrationOutcome.run_state` → persisted; used by the next tick as `prior_run_state`.
- `OrchestrationOutcome.actions_taken` → `core/usage_accounting.py` for token economics.
