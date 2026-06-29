# orchestrator_agent

## Purpose

Sequence and route the slow loop. Hold run state across an end-to-end cycle. Manage the **champion / challenger** protocol and route survivors to a human approval queue.

> **Build status (Stage 3).** Reconciled and implemented as **deterministic coordination code, NOT an LLM agent** (`core/orchestration/`):
> - **The orchestrator makes NO LLM call.** It invokes the four worker agents (`market_context → strategy_lab → backtest_agent → critic`), which each make their own calls through `AgentRunner`, and passes typed outputs between them. Sequencing and state are a deterministic state machine, not a model — so the spec's "Sonnet for full cycles / Haiku for status ticks" LLM summarisation does **not** belong here; it belongs to **`reporting_agent`** (deferred). There is no model tier, system prompt, or structured LLM output for this component.
> - **It is one pass:** `Orchestrator.run_cycle(universe, *, cycle_id)`. There is NO scheduler / daemon / unattended loop — running on a schedule is a deliberate later decision; a one-shot CLI is the only runner.
> - **TWO HARD WALLS (structural).** (1) It **cannot promote or trade** — surviving the critic routes a candidate to the append-only approval QUEUE (a pending entry awaiting a human), and that is the terminus; there is no path that promotes a challenger to champion, sets a strategy live, or flips the paper/live flag (promotion is a human-only Stage-4 action, not built). (2) It **cannot touch sacred config** — it READS `RiskConfig` to pass into backtests but never mutates it, trips/resets the kill switch, or changes the paper/live flag.
> - Same metrics-verbatim integrity as the worker agents: the orchestrator only moves their typed outputs around; it fabricates no numbers and runs the deterministic harness for in-sample + the token-gated OOS evidence.

## Responsibilities

- On schedule (cron or event), decide whether to run a **full cycle** (market context → candidates → backtests → optimization → critic → report) or a **status check** (read fast-loop audit + champion stats only).
- For a full cycle: invoke each agent in order, passing prior outputs as inputs.
- Enforce the dependency: a proposal cannot reach `reporting_agent` without an `accept` from `critic_agent`.
- Maintain `SlowLoopRunState` for the cycle (which candidates were tried, which proposals were generated, which verdicts were rendered, what's left).
- On error from any agent, decide: retry once, abort the cycle, or fall through to reporting with the failure noted.
- **Never** mutate live config or deploy a proposal directly. Outputs an approval-request artefact for the human.

## Inputs / API — as built

Deterministic, not an LLM request. Collaborators are injected (so the cycle is
fully testable with mocks):

```python
class Orchestrator:                # core.orchestration.orchestrator
    def __init__(self, *, market_context_agent, strategy_lab_agent,
                 backtest_agent, critic_agent, runner: AgentRunner,
                 candle_provider: CandleProvider,
                 registry: ChampionChallengerRegistry,
                 approval_queue: ApprovalQueue,
                 config: OrchestratorConfig | None = None) -> None: ...

    def run_cycle(self, universe: tuple[str, ...], *, cycle_id: str) -> CycleResult: ...

@dataclass(frozen=True)
class OrchestratorConfig:
    max_cost_per_cycle_usd: Decimal      # hard per-cycle budget (cost-side kill switch)
    per_call_cost_ceiling_usd: Decimal   # budgeted ceiling checked before each agent call
    n_candidates: int
    allowed_timeframes: tuple[Granularity, ...]
    backtest_config: BacktestRunConfig   # READ-only RiskConfig is inside here
```

## Outputs — as built

```python
class CycleResult(BaseModel):        # core.orchestration.orchestrator
    cycle_id: str
    outcome: CycleOutcome            # completed | aborted_budget | aborted_failure
    started_at: datetime; ended_at: datetime; duration_seconds: float
    total_cost_usd: Decimal; stage_costs_usd: dict[str, str]
    candidates_proposed: int; candidates_killed: int; candidates_queued: int
    queued_identities: tuple[str, ...]
    abort_reason: str | None
```

Persistent state lives in two JSON-backed stores (same spirit as
`usage_accounting`): `ChampionChallengerRegistry` (keyed by **strategy
identity** — the params hash — so re-proposing the same strategy is recognised,
not duplicated; states `PROPOSED | BACKTESTED | KILLED | SURVIVED_FOR_NOW |
QUEUED_FOR_APPROVAL`, with `APPROVED | CHAMPION | LIVE` reserved for a human and
**not orchestrator-writable**), and an append-only `ApprovalQueue` (pending
entries; the orchestrator can only append, never approve/reject).

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

**None — this component makes no LLM call.** The orchestrator is deterministic coordination code; the workers it invokes own their own tiers. The spec's earlier "Sonnet for full cycles / Haiku for status ticks" describes **`reporting_agent`'s** human-facing summarisation, which is deferred and built separately. (Per-worker tiers today: `market_context`/`strategy_lab`/`backtest_agent` on DEFAULT, `critic` on HEAVY.)

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
