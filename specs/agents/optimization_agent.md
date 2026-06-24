# optimization_agent

## Purpose

Propose **parameter changes** to an existing strategy candidate (or to the current champion), **strictly within the per-strategy bounded ranges**. Every proposal must be runnable through the same validation pipeline as a fresh candidate; nothing is auto-deployed.

## Responsibilities

- Read a `StrategyCandidate` (with its `parameter_ranges`) and the latest `BacktestInterpretation`.
- Pick a small set (≤ 5) of parameter perturbations that are likely to improve a *specific*, named weakness (e.g. drawdown in trending regime).
- Justify each perturbation in one sentence ("widen ATR stop in trending regime to capture longer moves").
- Emit `ParameterProposal`s with the new values, the *predicted* effect, and the gate that would falsify it.

## Inputs (with types)

```python
class OptimizationRequest:
    run_id: str
    parent_candidate: StrategyCandidate      # source of allowed parameter_ranges
    last_interpretation: BacktestInterpretation
    weakness: WeaknessTag                    # what to attack: "drawdown" | "oos_decay" | "regime_breakdown:risk_off" | ...
    max_proposals: int                       # default 3, hard cap 5
    forbid_values: set[ParamValueHash]       # already-tried this run
```

## Outputs (strict structured schema)

```python
class ParameterProposal:
    proposal_id: str                         # deterministic hash
    run_id: str
    parent_candidate_id: str
    weakness_targeted: WeaknessTag
    parameter_changes: dict[str, ParamValue] # MUST lie inside parent.parameter_ranges
    rationale: str                           # ≤ 300 chars, names the weakness
    predicted_effect: PredictedEffect        # which gate metric should move, in what direction, by how much (qualitative bins)
    falsifier: Falsifier                     # gate condition that, if hit, kills this proposal
    novelty_check: NoveltyCheck              # hash not in forbid_values; distance from parent
```

## Tools & Skills used

**Tools:** none at runtime. All inputs passed in.

**Skills:**
- `backtesting-methodology` — to ensure proposals can be cleanly walk-forwarded.
- `overfitting-checklist` — so the agent self-blocks proposals that look like curve fits.
- `prompt-caching-and-token-budget`.

## Model tier and why

**Sonnet** (`claude-sonnet-4-6`). Bounded parameter selection with explicit justification is well within Sonnet's ceiling. Opus is reserved for the critic, which evaluates these proposals adversarially.

## System-prompt structure

```
[tools]                                          ← empty
[system]
  - identity & objective                         ← stable
  - "only within parent.parameter_ranges"        ← stable, emphasised
  - schema for ParameterProposal                 ← stable
  - falsifier vocabulary                         ← stable
  - backtesting-methodology skill                ← stable
  - overfitting-checklist skill                  ← stable
  -------------- cache_control breakpoint ---------
  - none
[messages]
  - user: { parent, last_interpretation, weakness, forbid_values, max_proposals }
```

## Token budget

- **Input target:** ≤ 10k.
- **Output target:** ≤ 2k.
- Hard cap: 15k in / 4k out.

## Guardrails

- **Must not** propose values outside `parent_candidate.parameter_ranges`. The orchestrator re-validates and rejects violators; the agent must self-validate first.
- **Must not** change the strategy structure (entry/exit rules, indicators, timeframe) — only parameter values. Structural changes are `strategy_lab_agent`'s remit.
- **Must not** propose proposals whose hashes are in `forbid_values`.
- **Must not** propose values that are micro-perturbations of an already-tested value (`novelty_check.distance` enforces min step).
- **Must not** claim a proposal "is" an improvement; it can only predict and provide a falsifier.

## Hand-offs

- Every `ParameterProposal` goes through the **same** `backtest_agent` → `critic_agent` pipeline as a fresh `StrategyCandidate`. There is no fast lane.
- Persisted under `data/runs/<run_id>/proposals/<proposal_id>.json`.
