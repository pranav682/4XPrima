# critic_agent

## Purpose

**Adversarial.** Try to **kill** every candidate that survived `backtest_agent`, using the overfitting checklist, before it reaches a human. Default verdict is **kill** — a candidate is presumed overfit/spurious until the deterministic evidence forces otherwise.

> **Build status (Stage 3).** Implemented and reconciled to the real engine/evidence:
> - **Verdict is `{kill, survive_for_now}` — there is deliberately NO `approve`/`accept`/`deploy`/`trade` value (not even representable).** The critic NEVER authorizes anything; `survive_for_now` means only "not yet killed", explicitly NOT "validated". Only a human, downstream, can authorize live trading. (This supersedes the spec's earlier `accept | reject | needs_more_evidence`.)
> - **This is the stage that opens the OUT-OF-SAMPLE holdout** — but only in deterministic code. The harness (`core/agents/backtest_harness.py: run_robustness` / `run_candidate_oos`) supplies the token to `DataSplit.access_out_of_sample`; the LLM never sees or holds the token, only the resulting OOS evidence. OOS collapse vs in-sample is the canonical overfit kill.
> - Deterministic robustness evidence the critic interprets (never recomputes): in-sample + OOS runs, **cost-sensitivity** (1.5x/2x), **parameter-sensitivity** (neighbours within `parameter_ranges`), and **trade-concentration**. Same metrics-verbatim integrity gate as `backtest_agent`, now covering the OOS metrics too.
> - **Walk-forward instability** and **regime-dependence** checklist items are **deferred** until the harness computes them; the critic attacks the items it has deterministic evidence for.
> - Tier: **HEAVY** (the strongest reasoner) — the highest-stakes single judgement in the slow loop.

## Responsibilities

- Read the proposal/candidate, its `BacktestInterpretation`, and the relevant `MarketContextReport`.
- Run the `overfitting-checklist` (see skill) as a sequence of named tests:
  - Out-of-sample decay.
  - Walk-forward instability across folds.
  - Cost sensitivity (1.5× and 2× spread/commission).
  - Regime dependence (does it only work in one regime?).
  - Parameter sensitivity (small perturbations should not collapse the result).
  - Survivorship / look-ahead / data-snooping smells.
  - Trade-count sanity (too few trades → no statistical power; too many → fee bleed).
- For each test, emit a structured result with the **specific evidence** in the backtest report.
- Render a final `CriticVerdict` of `accept` (all gates pass), `reject` (one or more fail), or `needs_more_evidence` (a gate cannot be evaluated from the provided report).

## Inputs (with types) — as built

```python
@dataclass(frozen=True)
class CriticRequest:               # core.agents.critic_agent
    run_id: str
    robustness: tuple[RobustnessEvidence, ...]   # ALL deterministic evidence (incl. OOS)
    backtest_verdicts: BacktestVerdictSet | None = None  # prior triage, context only
    tier: ModelTier = ModelTier.HEAVY
```

`RobustnessEvidence` (per candidate, from the harness) carries the in-sample
`BacktestEvidence`, the token-gated out-of-sample `BacktestEvidence`, the
cost-stress points, the parameter-neighbour results, and the trade-concentration
stat. The critic reasons over these RAW numbers, not the prior agent's prose.

## Outputs (strict structured schema) — as built

```python
class CriticVerdictSet(BaseModel):     # core.models, frozen, extra=forbid
    run_id: str
    schema_version: str = "1.0"
    verdicts: tuple[CriticVerdict, ...]

class CriticVerdict(BaseModel):        # core.models, frozen, extra=forbid
    candidate_id: str
    in_sample_config_hash: str
    oos_config_hash: str | None
    in_sample_metrics: BacktestMetricsView         # copied VERBATIM
    out_of_sample_metrics: BacktestMetricsView | None  # copied verbatim if present
    verdict: CriticVerdictKind         # kill | survive_for_now  (NEVER approve)
    concerns: tuple[OverfittingConcern, ...]   # each mapped to a ChecklistItem
    assessment: str                    # ≤ 1000 chars, adversarial IS-vs-OOS read
    caveats: str                       # ≤ 500 chars
```

There is **NO approve/deploy/live field** anywhere, and the free text is scrubbed
for execution intent.

## Tools & Skills used

**Tools:** none at runtime. The deterministic backtester has already produced everything the critic needs; if information is missing, the answer is `needs_more_evidence`, **not** "run another backtest".

**Skills:**
- `overfitting-checklist` — the canonical list of tests and pass/fail bands.
- `backtesting-methodology` — to reason about walk-forward and OOS construction.
- `forex-cost-modeling` — for the cost sensitivity test.
- `prompt-caching-and-token-budget`.

## Model tier and why

**HEAVY tier** (`ModelTier.HEAVY` → the strongest reasoner). This is the highest-stakes single judgement in the slow loop — a false `survive_for_now` lets a curve-fit reach a human under a halo it didn't earn. The heavy tier's adversarial reasoning is the budget we explicitly pay for. The (long, stable) checklist prompt is cached aggressively; per-call inputs are small.

## System-prompt structure

```
[tools]                                          ← empty
[system]
  - identity: "you are an adversarial critic. Default to reject." ← stable
  - schema for CriticVerdict                     ← stable
  - overfitting-checklist skill (full text)      ← stable, large
  - backtesting-methodology skill                ← stable
  - forex-cost-modeling skill                    ← stable
  -------------- cache_control breakpoint ---------
  - none
[messages]
  - user: { subject, backtest_interpretation, market_context, thresholds }
```

Opus cache minimum is 4k tokens — the stable system block above is well above that, so caching is real.

## Token budget

- **Input target:** ≤ 18k (stable ~10–12k, volatile up to 6k).
- **Output target:** ≤ 3k (structured checks + verdict).
- Hard cap: 25k in / 6k out.

## Guardrails

- **Default verdict is `reject`.** The agent must positively cite passing evidence to flip to `accept`.
- **Must not** mark `accept` if any check fails — collapse to `reject` (or `needs_more_evidence` if a check is uncomputable).
- **Must not** ask for more backtests — `needs_more_evidence` is signal to the orchestrator that *deterministic* code (more data, longer OOS) is required.
- **Must not** restate the strategy's promise; the job is to attack, not summarise.
- **Must not** be swayed by the `interpretation` field of the backtest — it reasons over the raw metrics in `GateResult`s.

## Hand-offs

- `CriticVerdict` is consumed by:
  - `orchestrator_agent` (state machine: only `accept` advances to human review),
  - `reporting_agent` (the human sees both the proposal AND the critic verdict, side by side).
- Persisted under `data/runs/<run_id>/verdicts/<verdict_id>.json`.
