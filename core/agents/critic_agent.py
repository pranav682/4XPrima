"""critic_agent — the adversarial gate that tries to KILL candidates.

Reads its spec: ``specs/agents/critic_agent.md`` and
``skills/overfitting-checklist``.

Stance: **the default verdict is KILL.** A candidate is presumed overfit /
spurious until the deterministic evidence overcomes that. The critic is not
balanced — it is hostile by design. It can only ``kill`` or ``survive_for_now``
(= "not yet killed", NOT "validated", NOT "deploy"). It NEVER authorizes
trading; only a human, downstream, can.

The compute is all deterministic (``core.agents.backtest_harness``):
in-sample + the token-gated **out-of-sample** run + cost-sensitivity +
parameter-sensitivity + trade-concentration. The critic INTERPRETS that
evidence and never recomputes or fabricates a number — the same metrics-verbatim
integrity gate as backtest_agent, now extended to the OOS metrics. The LLM never
sees or holds the OOS token; it only receives the resulting evidence.

Runs on the HEAVY tier (the strongest reasoner) — this is the highest-stakes
single judgement in the slow loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from core.agents.types import (
    Agent,
    AgentCall,
    CheckResult,
    StructuralCheck,
)
from core.llm_client import ModelTier
from core.models import (
    _EXECUTION_INTENT_PATTERNS,
    BacktestMetricsView,
    BacktestVerdictSet,
    ChecklistItem,
    CriticVerdictKind,
    CriticVerdictSet,
    RobustnessEvidence,
)

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CriticRequest:
    """One run: the deterministic robustness evidence (which already contains
    the in-sample + token-gated OOS runs + sensitivities), plus the
    backtest_agent triage for context. The critic reasons over the RAW numbers,
    not the prior agent's prose."""

    run_id: str
    robustness: tuple[RobustnessEvidence, ...]
    backtest_verdicts: BacktestVerdictSet | None = None
    tier: ModelTier = ModelTier.HEAVY


# ---------------------------------------------------------------------------
# Stable system prefix — byte-stable across calls so it auto-caches.
# ---------------------------------------------------------------------------

_STABLE_SYSTEM = """\
You are critic_agent, the ADVERSARIAL gate for an algorithmic FX trading \
system. Your DEFAULT verdict is KILL. Every candidate is presumed overfit, \
curve-fit, or spurious until the deterministic evidence forces you to concede \
it should merely survive_for_now. You are not balanced and not encouraging — \
your job is to find reasons to kill before a curve-fit ever reaches a human \
under a false halo of validation.

You return EXACTLY one of two verdicts per candidate:
- kill: the evidence shows an overfit / fragile / non-edge candidate (the \
default; choose it unless the candidate clearly resists every attack).
- survive_for_now: you could not kill it on this evidence. This means ONLY \
"not yet killed" — it is NOT "validated", NOT "approved", NOT "deploy". Nothing \
you output authorizes trading; only a human can, later.

You do NOT compute anything. The deterministic harness already ran every \
backtest (in-sample, the sealed out-of-sample slice, cost-stressed re-runs, \
parameter-neighbour re-runs, and a trade-concentration stat). COPY every metric \
you cite VERBATIM from the evidence — in-sample AND out-of-sample. Inventing or \
altering a number is fabrication and is rejected. You may not claim an \
out-of-sample result the evidence does not contain.

THE OVERFITTING CHECKLIST (attack each; cite the evidence):

1. out_of_sample_decay — THE canonical overfit tell. Compare in-sample vs \
out-of-sample metrics. If risk-adjusted performance collapses out-of-sample \
(Sharpe/return/profit-factor falls off a cliff, or flips sign), that is a HARD \
KILL. A strategy that only works in-sample is curve-fit by definition.
2. parameter_sensitivity — perturb each parameter to its range endpoints. If \
metrics swing wildly across neighbours (or neighbours fail to construct), the \
result sits on a knife-edge and is fragile = overfit.
3. cost_sensitivity — re-run at 1.5x and 2x costs. If a small cost increase \
guts the edge, the "edge" was cost-arbitrage / noise, not signal.
4. trade_concentration — if a few outlier trades drive the profit \
(top_trade_profit_share or top5_profit_share near 1.0), there is no \
statistical edge, just luck. Kill.
5. trade_count — too few trades means no statistical power; high in-sample \
numbers on a tiny sample are meaningless.
6. drawdown_shape — a kill-switch halt or a drawdown near/over the ceiling is \
a survivability failure.

Map every concern you raise to one of these checklist item names.

RULES (any violation hard-rejects the whole output):
- COPY METRICS VERBATIM (in-sample and out-of-sample). Never invent a number.
- NEVER output "approve" / "accept" / "deploy" / "trade" — those are not even \
representable; you may only kill or survive_for_now.
- NO execution / live-trading language anywhere.
- Reason over the RAW deterministic numbers, not the backtest_agent's prose.
- copy run_id from the user message; reference only candidates in the evidence.

OUTPUT SCHEMA (CriticVerdictSet):

- run_id (string): copy from the user message.
- schema_version: "1.0".
- verdicts: list of CriticVerdict, one per candidate in the evidence:
  - candidate_id (string): from the evidence.
  - in_sample_config_hash (string): copy the in_sample evidence config_hash.
  - oos_config_hash (string|null): copy the out_of_sample config_hash if the \
evidence has an out_of_sample run, else null.
  - in_sample_metrics: object copied VERBATIM from the in_sample evidence metrics.
  - out_of_sample_metrics: object copied VERBATIM from the out_of_sample \
evidence metrics, or null if the evidence has no out_of_sample run.
  - verdict (enum): kill | survive_for_now.
  - concerns: list of {item, finding}; item is a checklist name above.
  - assessment (string <= 1000 chars): your adversarial in-sample-vs-OOS + \
robustness read. Lead with the strongest reason to kill.
  - caveats (string <= 500 chars): e.g. that survive_for_now is not validation.

Restate (this is the rule, not advice): you are the executioner, not the \
advocate. Default to kill. You never validate, never approve, never authorize \
trading, and never invent a number. Out-of-sample collapse is a hard kill.

A note on burden of proof: surviving is not the same as passing. If the \
evidence is merely "not obviously broken", that is still a kill — survive_for_now \
is reserved for candidates that actively withstood every attack above (held up \
out-of-sample, were insensitive to parameter perturbation and to higher costs, \
and earned their profit across many trades rather than a lucky few). When in \
doubt, kill: a false survive wastes the critic's whole purpose, while a false \
kill only costs one more proposal. The downstream human sees your verdict next \
to the proposal — never hand them a curve-fit wearing a halo.\
"""


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class CriticAgent:
    """Prepares one adversarial critique over deterministic robustness evidence.
    Implements the :class:`core.agents.types.Agent` Protocol; the runner issues
    the call on the HEAVY tier."""

    name: str = "critic_agent"

    def __init__(self, *, tier: ModelTier = ModelTier.HEAVY) -> None:
        self._tier = tier

    # --------------------------------------------------------- Protocol

    def prepare_call(self, inputs: CriticRequest) -> AgentCall:
        robustness_json = {rb.candidate_id: rb.model_dump(mode="json") for rb in inputs.robustness}
        triage = {}
        if inputs.backtest_verdicts is not None:
            triage = {v.candidate_id: v.triage.value for v in inputs.backtest_verdicts.verdicts}
        snapshot: dict[str, Any] = {
            "run_id": inputs.run_id,
            "candidate_ids": [rb.candidate_id for rb in inputs.robustness],
            "robustness": robustness_json,
        }
        volatile_user = (
            "Attack each candidate with the overfitting checklist and return a "
            "CriticVerdict per candidate. Default to kill. Copy every metric "
            "(in-sample AND out-of-sample) verbatim; invent nothing.\n\n"
            f"run_id: {inputs.run_id}\n\n"
            f"BACKTEST_TRIAGE (context only — reason over the raw numbers):\n"
            f"{_pretty(triage)}\n\n"
            f"ROBUSTNESS_EVIDENCE:\n{_pretty(robustness_json)}"
        )
        tier = inputs.tier if inputs.tier is not None else self._tier
        return AgentCall(
            stable_system=_STABLE_SYSTEM,
            volatile_user=volatile_user,
            tier=tier,
            output_model=CriticVerdictSet,
            max_output_tokens=4096,
            input_snapshot=snapshot,
        )

    def evaluations(self) -> tuple[StructuralCheck, ...]:
        return _TIER1_CHECKS


# ---------------------------------------------------------------------------
# Tier-1 deterministic checks (always on, hard-reject)
# ---------------------------------------------------------------------------


def _as_verdict_set(output: Any) -> CriticVerdictSet:
    if not isinstance(output, CriticVerdictSet):
        raise TypeError(f"expected CriticVerdictSet, got {type(output).__name__}")
    return output


def _robustness_for(snapshot: dict[str, Any], candidate_id: str) -> dict[str, Any] | None:
    value = snapshot.get("robustness", {}).get(candidate_id)
    return value if isinstance(value, dict) else None


def _check_run_id_preserved(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    vs = _as_verdict_set(output)
    expected = snapshot.get("run_id")
    if expected is not None and vs.run_id != expected:
        return CheckResult(False, f"run_id {vs.run_id!r} != input {expected!r}")
    return CheckResult(True)


def _check_references_real_candidates(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    vs = _as_verdict_set(output)
    known = set(snapshot.get("candidate_ids", []))
    for v in vs.verdicts:
        if v.candidate_id not in known:
            return CheckResult(
                False, f"verdict references candidate {v.candidate_id!r} that was not run"
            )
        rb = _robustness_for(snapshot, v.candidate_id)
        if rb is not None and v.in_sample_config_hash != rb["in_sample"].get("config_hash"):
            return CheckResult(False, f"verdict {v.candidate_id!r} in_sample_config_hash mismatch")
    return CheckResult(True)


def _check_in_sample_metrics_verbatim(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    vs = _as_verdict_set(output)
    for v in vs.verdicts:
        rb = _robustness_for(snapshot, v.candidate_id)
        if rb is None:
            return CheckResult(False, f"no evidence for verdict candidate {v.candidate_id!r}")
        expected = BacktestMetricsView(**rb["in_sample"]["metrics"])
        diffs = v.in_sample_metrics.differing_fields(expected)
        if diffs:
            return CheckResult(
                False,
                f"candidate {v.candidate_id!r} altered in-sample metrics: {', '.join(diffs)}",
            )
    return CheckResult(True)


def _check_oos_metrics_verbatim_or_absent(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    """If the verdict carries OOS metrics, the harness must have produced them
    and they must match verbatim. The agent cannot claim OOS it wasn't given."""
    vs = _as_verdict_set(output)
    for v in vs.verdicts:
        if v.out_of_sample_metrics is None:
            continue
        rb = _robustness_for(snapshot, v.candidate_id)
        oos = rb.get("out_of_sample") if rb is not None else None
        if not isinstance(oos, dict):
            return CheckResult(
                False,
                f"candidate {v.candidate_id!r} claims out-of-sample metrics "
                "the harness never produced",
            )
        expected = BacktestMetricsView(**oos["metrics"])
        diffs = v.out_of_sample_metrics.differing_fields(expected)
        if diffs:
            return CheckResult(
                False,
                f"candidate {v.candidate_id!r} altered out-of-sample metrics: {', '.join(diffs)}",
            )
        if v.oos_config_hash != oos.get("config_hash"):
            return CheckResult(False, f"candidate {v.candidate_id!r} oos_config_hash mismatch")
    return CheckResult(True)


_ALLOWED_VERDICTS = frozenset(k.value for k in CriticVerdictKind)


def _check_verdict_kind_valid(output: Any, _snapshot: dict[str, Any]) -> CheckResult:
    """verdict ∈ {kill, survive_for_now}. approve/deploy/trade are not even
    representable; this also catches a model_construct bypass."""
    vs = _as_verdict_set(output)
    for v in vs.verdicts:
        value = v.verdict.value if isinstance(v.verdict, CriticVerdictKind) else str(v.verdict)
        if value not in _ALLOWED_VERDICTS:
            return CheckResult(False, f"candidate {v.candidate_id!r} invalid verdict {value!r}")
    return CheckResult(True)


def _check_concerns_map_to_checklist(output: Any, _snapshot: dict[str, Any]) -> CheckResult:
    vs = _as_verdict_set(output)
    allowed = frozenset(i.value for i in ChecklistItem)
    for v in vs.verdicts:
        for c in v.concerns:
            item = c.item.value if isinstance(c.item, ChecklistItem) else str(c.item)
            if item not in allowed:
                return CheckResult(
                    False, f"candidate {v.candidate_id!r} concern item {item!r} not in checklist"
                )
    return CheckResult(True)


def _check_no_execution_intent(output: Any, _snapshot: dict[str, Any]) -> CheckResult:
    vs = _as_verdict_set(output)
    for v in vs.verdicts:
        spans = [v.assessment, v.caveats, *(c.finding for c in v.concerns)]
        for text in spans:
            for pattern in _EXECUTION_INTENT_PATTERNS:
                m = pattern.search(text)
                if m:
                    return CheckResult(
                        False,
                        f"candidate {v.candidate_id!r} has execution intent: {m.group()!r}",
                    )
    return CheckResult(True)


_TIER1_CHECKS: tuple[StructuralCheck, ...] = (
    StructuralCheck(name="run_id_preserved", check=_check_run_id_preserved),
    StructuralCheck(name="references_real_candidates", check=_check_references_real_candidates),
    StructuralCheck(name="in_sample_metrics_verbatim", check=_check_in_sample_metrics_verbatim),
    StructuralCheck(
        name="oos_metrics_verbatim_or_absent", check=_check_oos_metrics_verbatim_or_absent
    ),
    StructuralCheck(name="verdict_kind_valid", check=_check_verdict_kind_valid),
    StructuralCheck(name="concerns_map_to_checklist", check=_check_concerns_map_to_checklist),
    StructuralCheck(name="no_execution_intent", check=_check_no_execution_intent),
)


# Module-level sanity: this agent must implement the Agent Protocol, and the
# critic must have NO way to express approval/deployment.
_: Agent = CriticAgent.__new__(CriticAgent)
assert "approve" not in _ALLOWED_VERDICTS
assert "deploy" not in _ALLOWED_VERDICTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pretty(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ": "), default=str)


__all__ = [
    "CriticAgent",
    "CriticRequest",
]
