---
name: risk-management-rules
description: Vocabulary and rules the risk manager enforces. Read-only reference for any agent that needs to summarise or reason about risk events; the risk manager itself is deterministic code (see specs/components/risk_manager.md).
---

# risk-management-rules

## When to use

- `reporting_agent` summarising fast-loop risk events for the human.
- `critic_agent` reasoning about whether a candidate's worst-case drawdown is acceptable.
- `orchestrator_agent` describing kill-switch events.
- Any agent that mentions risk concepts in output, so the language is consistent and auditable.

**The risk manager is deterministic code.** This file does not configure it; it documents its vocabulary.

## The rules (single source: `RiskConfig` in `specs/components/risk_manager.md`)

### Position sizing

```
size = (account_equity * max_risk_per_trade_pct) / (stop_distance_in_value)
```

- Stop distance is in account-currency value, not pips. Converting pips→value uses tick value and contract size for the instrument.
- If `stop_distance ≤ 0`, the request is rejected as `STOP_DISTANCE_NONPOSITIVE` — no defaulting.

### Caps (all rejections are recorded in the audit)

| Cap | Rejection reason | What it limits |
| --- | --- | --- |
| `max_risk_per_trade_pct` | `PER_TRADE_CAP` | The single-trade equity-at-risk if the stop hits. |
| `max_exposure_per_symbol_pct` | `PER_SYMBOL_CAP` | The combined exposure on one symbol (across simultaneous positions). |
| `max_portfolio_risk_pct` | `PORTFOLIO_CAP` | The combined open risk across the whole portfolio. |
| `max_correlation` + `max_correlated_exposure_pct` | `CORRELATION_CAP` | The combined exposure on correlated instruments. |
| `max_drawdown_pct` over `drawdown_window` | `DRAWDOWN_CAP` | New opens are denied when rolling drawdown exceeds threshold. |

### Kill switch

- A single flag, file-backed.
- When engaged: **every** new open is denied with reason `KILL_SWITCH`. Existing positions are managed by the flatten policy (`flatten_immediately` or `let_stops_work`).
- Engages automatically on: unhandled exception in fast loop, failure to reload `RiskConfig`, audit-log write failure.
- Cleared only by a deterministic operator command. No agent (and no slow-loop process) has the credentials to clear it.

### Reporting vocabulary (use these terms, in this order, when summarising)

1. **State** — `OK` | `THROTTLED` (drawdown cap) | `KILLED` (kill switch).
2. **Open risk** — sum of `(distance_to_stop × size × tick_value)` across positions, as a % of equity.
3. **Per-symbol exposure** — table of symbol → % equity exposed.
4. **Drawdown** — current rolling drawdown vs cap.
5. **Recent rejections** — count by `RejectionReason` in last 24h.
6. **Last config hash** — to identify which `RiskConfig` made the recent decisions.

## What this skill forbids

- Agents that read this **must not** say or imply "I will tighten the cap to X" or "we should temporarily raise…" — the risk manager is not slow-loop-configurable. Proposed `RiskConfig` changes are handled the same as strategy proposals: validation, critic, human approval, deterministic deploy.
- Reports must not paraphrase rejection reasons; quote the enum value.
- Numbers cited in reports must be pulled from the audit log, not estimated.

## Checklist (for an agent generating a risk-related summary)

- [ ] State, open risk, per-symbol exposure, drawdown, rejections, config hash — all present.
- [ ] Kill-switch state explicit even if `OK`.
- [ ] All numbers cite the audit log artefact.
- [ ] No language that implies the agent or human-via-chat can change the risk config; changes go via the approval gate.
