"""backtest_agent — turns deterministic backtest evidence into a judged verdict.

Reads its spec: ``specs/agents/backtest_agent.md``.

This agent closes the first loop: proposal → deterministic backtest →
interpreted verdict. It does NOT compute anything. The deterministic
:mod:`core.agents.backtest_harness` builds each strategy, runs the real Stage-2
engine over the IN-SAMPLE window, and produces a
:class:`core.models.BacktestEvidence` per candidate. This agent INTERPRETS that
evidence: an honest in-sample read, which fixed gates passed/failed, overfit
smells, and a triage recommendation for the critic stage.

HARD INTEGRITY BOUNDARY — the LLM must NEVER produce or alter a number:

- Every metric in a verdict is copied VERBATIM from the evidence. Tier-1
  rejects any mismatch (the core integrity check).
- In-sample ONLY. There is no out-of-sample metric in the evidence, the output
  model has no OOS field, and Tier-1 rejects an OOS-result claim in the prose.
- It proposes nothing for live trading; the free text is scrubbed for
  execution/deployment intent and Tier-1 re-checks it.

Runs on the DEFAULT tier (``gpt-5.4``).
"""

from __future__ import annotations

import json
import re
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
    BacktestEvidence,
    BacktestMetricsView,
    BacktestTriage,
    BacktestVerdictSet,
    StrategyProposal,
)

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BacktestAgentRequest:
    """One run: a proposal plus the deterministic evidence already computed by
    :mod:`core.agents.backtest_harness`. The agent does not run the harness —
    it only interprets its output."""

    run_id: str
    proposal: StrategyProposal
    evidence: tuple[BacktestEvidence, ...]
    tier: ModelTier = ModelTier.DEFAULT


# ---------------------------------------------------------------------------
# Stable system prefix — byte-stable across calls so OpenAI auto-caches it.
# ---------------------------------------------------------------------------

_STABLE_SYSTEM = """\
You are backtest_agent for an algorithmic FX trading system. You INTERPRET \
deterministic in-sample backtest evidence. You do NOT compute, optimize, or \
trade anything, and you NEVER produce or alter a number.

The deterministic harness has already built each strategy, run the real \
backtest engine over the IN-SAMPLE window, and handed you a BacktestEvidence \
per candidate (verbatim metrics, trade count, cost breakdown, config_hash, and \
fixed-gate pass/fail flags). Your job is to read that evidence and emit a \
BacktestVerdict per candidate: an honest assessment, the gates it passed or \
failed, flagged concerns, and a triage recommendation for the critic stage.

ABSOLUTE RULES (any violation hard-rejects the whole output):

- COPY METRICS VERBATIM. Every number in metrics and every gate you report \
must be copied EXACTLY from the candidate's evidence. Do not recompute, round, \
average, or "correct" anything. If you are tempted to produce a number that is \
not already in the evidence, stop — that is fabrication and it is rejected.
- IN-SAMPLE ONLY. The evidence is in-sample. There is NO out-of-sample / \
holdout result, and you must not invent one. Do not state an out-of-sample \
Sharpe/return/drawdown/profit-factor figure — there is none. You MAY (and \
should) caveat that results are in-sample only.
- NO PREDICTIVE CLAIMS. In-sample performance is NOT proof of future edge. \
Never claim a candidate "will" be profitable or is "validated". Describe what \
the in-sample run showed and how fragile it looks.
- NO TRADE / DEPLOY LANGUAGE. You do not approve, deploy, or signal live \
trading. No "go live", "deploy", "execute", "real money".
- Interpret the gates the harness already computed; do not invent new gates or \
flip a gate's pass/fail.
- Reference ONLY candidates that appear in the evidence. copy run_id from the \
user message.

WHAT TO ASSESS (honestly, per candidate):

- Risk-adjusted read: Sharpe / Sortino / max drawdown / profit factor — quote \
the evidence's numbers, say what they imply, note that high in-sample numbers \
are cheap.
- Overfit smells: too few trades (small samples lie), drawdown near the \
ceiling, profit driven by a handful of trades (low trade count + high return), \
exposure that is suspiciously high or low, a kill-switch halt.
- Triage: choose one of advance_to_critic | reject | needs_different_params.
  - advance_to_critic: gates broadly pass and the in-sample read is plausible \
enough to be worth the critic's adversarial pass.
  - reject: gates fail badly or the run halted / had almost no trades.
  - needs_different_params: promising shape but a parameter looks mis-set \
(e.g. drawdown just over ceiling, or trade count just under the floor).

OUTPUT SCHEMA (BacktestVerdictSet):

- run_id (string): copy from the user message.
- schema_version: "1.0".
- verdicts: list of BacktestVerdict, one per candidate in the evidence:
  - candidate_id (string): the evidence candidate_id.
  - config_hash (string): copy the evidence config_hash (the run's identity).
  - metrics: object copied VERBATIM from the evidence metrics (same fields, \
same numbers; sortino_ratio / profit_factor are null when the evidence shows \
null, meaning no downside / no losing trades).
  - gates: list of {name, passed, detail} copied from the evidence gates.
  - assessment (string <= 800 chars): the honest in-sample read.
  - concerns (list of short strings): overfit smells / fragility you see.
  - triage (enum): advance_to_critic | reject | needs_different_params.
  - caveats (string <= 500 chars): in-sample-only / not-predictive caveats.

OUTPUT EXAMPLE (shape, NOT content — copy the candidate's real numbers, not \
these placeholders):

{
  "run_id": "<from user message>",
  "schema_version": "1.0",
  "verdicts": [
    {
      "candidate_id": "<from evidence>",
      "config_hash": "<copied from evidence>",
      "metrics": {
        "total_return_pct": "<copied>", "annualised_return_pct": "<copied>",
        "sharpe_ratio": 0.0, "sortino_ratio": null,
        "max_drawdown_pct": "<copied>", "win_rate": 0.0,
        "profit_factor": null, "trade_count": 0,
        "avg_trade_pnl": "<copied>", "exposure_pct": 0.0
      },
      "gates": [
        {"name": "min_trade_count", "passed": true, "detail": "<copied>"},
        {"name": "max_drawdown", "passed": false, "detail": "<copied>"}
      ],
      "assessment": "In-sample Sharpe is modest and the run cleared the trade \
floor, but max drawdown sits near the ceiling; the edge looks thin and \
cost-sensitive. High in-sample numbers prove nothing here.",
      "concerns": ["drawdown near ceiling", "few trades for the timeframe"],
      "triage": "needs_different_params",
      "caveats": "In-sample only; no out-of-sample validation has run yet. \
Not evidence of a future edge."
    }
  ]
}

Restate (this is the rule, not advice): you interpret deterministic evidence. \
You never invent a number, never claim out-of-sample results, never predict \
the future, and never signal live trading. Predictive correctness is decided \
later by the out-of-sample slice and the critic — never by you.\
"""


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class BacktestAgent:
    """Prepares one interpretation call over deterministic evidence. Implements
    the :class:`core.agents.types.Agent` Protocol; the runner issues the call."""

    name: str = "backtest_agent"

    def __init__(self, *, tier: ModelTier = ModelTier.DEFAULT) -> None:
        self._tier = tier

    # --------------------------------------------------------- Protocol

    def prepare_call(self, inputs: BacktestAgentRequest) -> AgentCall:
        evidence_json = {ev.candidate_id: ev.model_dump(mode="json") for ev in inputs.evidence}
        rationale_by_id = {c.candidate_id: c.rationale for c in inputs.proposal.candidates}
        snapshot: dict[str, Any] = {
            "run_id": inputs.run_id,
            "candidate_ids": [ev.candidate_id for ev in inputs.evidence],
            "evidence": evidence_json,
        }
        volatile_user = (
            "Interpret the deterministic IN-SAMPLE backtest evidence below into "
            "a BacktestVerdict per candidate. Copy every metric and gate "
            "verbatim from the evidence; invent nothing.\n\n"
            f"run_id: {inputs.run_id}\n\n"
            f"PROPOSAL_RATIONALES:\n{_pretty(rationale_by_id)}\n\n"
            f"EVIDENCE:\n{_pretty(evidence_json)}"
        )
        tier = inputs.tier if inputs.tier is not None else self._tier
        return AgentCall(
            stable_system=_STABLE_SYSTEM,
            volatile_user=volatile_user,
            tier=tier,
            output_model=BacktestVerdictSet,
            max_output_tokens=4096,
            input_snapshot=snapshot,
        )

    def evaluations(self) -> tuple[StructuralCheck, ...]:
        return _TIER1_CHECKS


# ---------------------------------------------------------------------------
# Tier-1 deterministic checks (always on, hard-reject)
# ---------------------------------------------------------------------------


def _as_verdict_set(output: Any) -> BacktestVerdictSet:
    if not isinstance(output, BacktestVerdictSet):
        raise TypeError(f"expected BacktestVerdictSet, got {type(output).__name__}")
    return output


def _evidence_for(snapshot: dict[str, Any], candidate_id: str) -> dict[str, Any] | None:
    evidence = snapshot.get("evidence", {})
    value = evidence.get(candidate_id)
    return value if isinstance(value, dict) else None


def _check_run_id_preserved(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    vs = _as_verdict_set(output)
    expected = snapshot.get("run_id")
    if expected is not None and vs.run_id != expected:
        return CheckResult(False, f"run_id {vs.run_id!r} != input {expected!r}")
    return CheckResult(True)


def _check_references_run_candidates(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    vs = _as_verdict_set(output)
    run_ids = set(snapshot.get("candidate_ids", []))
    for v in vs.verdicts:
        if v.candidate_id not in run_ids:
            return CheckResult(
                False, f"verdict references candidate {v.candidate_id!r} that was not run"
            )
        ev = _evidence_for(snapshot, v.candidate_id)
        if ev is not None and v.config_hash != ev.get("config_hash"):
            return CheckResult(
                False,
                f"verdict {v.candidate_id!r} config_hash {v.config_hash!r} != "
                f"evidence {ev.get('config_hash')!r}",
            )
    return CheckResult(True)


def _check_metrics_verbatim(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    """THE integrity check: every metric in a verdict must equal the
    deterministic evidence value. Catches any LLM-fabricated/altered number."""
    vs = _as_verdict_set(output)
    for v in vs.verdicts:
        ev = _evidence_for(snapshot, v.candidate_id)
        if ev is None:
            return CheckResult(False, f"no evidence for verdict candidate {v.candidate_id!r}")
        expected = BacktestMetricsView(**ev["metrics"])
        diffs = v.metrics.differing_fields(expected)
        if diffs:
            return CheckResult(
                False,
                f"candidate {v.candidate_id!r} fabricated/altered metrics: {', '.join(diffs)}",
            )
    return CheckResult(True)


def _check_gates_verbatim(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    """Gate names + pass/fail must match the deterministic evidence (the LLM
    must not flip, add, or drop a gate)."""
    vs = _as_verdict_set(output)
    for v in vs.verdicts:
        ev = _evidence_for(snapshot, v.candidate_id)
        if ev is None:
            continue  # covered by references / metrics checks
        expected = {g["name"]: bool(g["passed"]) for g in ev.get("gates", [])}
        got = {g.name: g.passed for g in v.gates}
        if got != expected:
            return CheckResult(
                False, f"candidate {v.candidate_id!r} gate results do not match evidence"
            )
    return CheckResult(True)


_OOS_RESULT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:oos|out[- ]of[- ]sample|out of sample|held[- ]out|holdout)\b[^.]{0,50}"
        r"\b(?:sharpe|sortino|return|drawdown|profit\s*factor|win[\s-]?rate|pnl|p&l)\b"
        r"[^.]{0,25}[-+]?\d",
        re.IGNORECASE,
    ),
)


def _check_no_oos_results(output: Any, _snapshot: dict[str, Any]) -> CheckResult:
    """Reject any claimed out-of-sample metric in the prose. The evidence is
    in-sample only; an OOS number can only be fabricated. (Caveating that
    results are in-sample / OOS-not-yet-run is fine — it carries no figure.)"""
    vs = _as_verdict_set(output)
    for v in vs.verdicts:
        spans = [v.assessment, v.caveats, *v.concerns]
        for text in spans:
            for pattern in _OOS_RESULT_PATTERNS:
                m = pattern.search(text)
                if m:
                    return CheckResult(
                        False,
                        f"candidate {v.candidate_id!r} claims an out-of-sample "
                        f"result: {m.group()!r}",
                    )
    return CheckResult(True)


def _check_no_execution_intent(output: Any, _snapshot: dict[str, Any]) -> CheckResult:
    vs = _as_verdict_set(output)
    for v in vs.verdicts:
        spans = [v.assessment, v.caveats, *v.concerns]
        for text in spans:
            for pattern in _EXECUTION_INTENT_PATTERNS:
                m = pattern.search(text)
                if m:
                    return CheckResult(
                        False,
                        f"candidate {v.candidate_id!r} has execution intent: {m.group()!r}",
                    )
    return CheckResult(True)


_TRIAGE_VALUES = frozenset(t.value for t in BacktestTriage)


def _check_triage_valid(output: Any, _snapshot: dict[str, Any]) -> CheckResult:
    vs = _as_verdict_set(output)
    for v in vs.verdicts:
        value = v.triage.value if isinstance(v.triage, BacktestTriage) else str(v.triage)
        if value not in _TRIAGE_VALUES:
            return CheckResult(False, f"candidate {v.candidate_id!r} invalid triage {value!r}")
    return CheckResult(True)


_TIER1_CHECKS: tuple[StructuralCheck, ...] = (
    StructuralCheck(name="run_id_preserved", check=_check_run_id_preserved),
    StructuralCheck(name="references_run_candidates", check=_check_references_run_candidates),
    StructuralCheck(name="metrics_verbatim", check=_check_metrics_verbatim),
    StructuralCheck(name="gates_verbatim", check=_check_gates_verbatim),
    StructuralCheck(name="no_oos_results", check=_check_no_oos_results),
    StructuralCheck(name="no_execution_intent", check=_check_no_execution_intent),
    StructuralCheck(name="triage_valid", check=_check_triage_valid),
)


# Module-level sanity: this agent must implement the Agent Protocol.
_: Agent = BacktestAgent.__new__(BacktestAgent)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pretty(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ": "), default=str)


__all__ = [
    "BacktestAgent",
    "BacktestAgentRequest",
]
