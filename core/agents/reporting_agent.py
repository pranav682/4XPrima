"""reporting_agent — the human-facing surface of the slow loop.

Reads its spec: ``specs/agents/reporting_agent.md``.

Turns the orchestrator's deterministic output (a ``CycleResult`` + the pending
approval-queue items) into an honest, readable summary for the operator. It
REPORTS; it does NOT decide, recommend, or authorize anything.

Its failure mode is **editorializing** — making a queued candidate sound more
promising than the critic's verdict warrants, predicting profitability, or
nudging the operator toward approve/reject. The design blocks it:

- every metric / count / cost is copied VERBATIM from the evidence (the same
  integrity gate as the other agents), and Tier-1 rejects any mismatch;
- there is NO recommendation / approve / deploy field — nothing it outputs can
  authorize trading — and all free text is scrubbed for recommendation and
  execution language;
- it must reproduce EVERY surviving critic concern for each queued candidate
  (it cannot drop a caveat), and frame a survivor as "the critic did not kill
  this; here is what it remains worried about", never as "validated / good".

Tier: DEFAULT (gpt-5.4) for full cycle summaries; CHEAP (nano) for short
status/tick summaries. Selected by the request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
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
    _RECOMMENDATION_PATTERNS,
    BacktestMetricsView,
    CycleReport,
)
from core.orchestration import ApprovalQueueEntry, CycleResult

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReportingRequest:
    """One run: the cycle's deterministic result + the pending approval items it
    produced. Everything here is already computed; the agent only translates."""

    run_id: str  # == cycle_id
    cycle_result: CycleResult
    universe: tuple[str, ...]
    queued_items: tuple[ApprovalQueueEntry, ...] = ()
    tier: ModelTier = ModelTier.DEFAULT  # CHEAP for short status/tick summaries


# ---------------------------------------------------------------------------
# Stable system prefix — byte-stable across calls so it auto-caches.
# ---------------------------------------------------------------------------

_STABLE_SYSTEM = """\
You are reporting_agent, the ONLY human-facing surface of an algorithmic FX \
trading system's slow loop. You translate one finished cycle into an honest, \
readable summary for the human operator. You REPORT; you do not decide, \
recommend, approve, reject, or authorize anything. The operator's reply, later, \
is the only deciding action — and you are not building that surface.

Your single most important constraint: you do NOT editorialize. The tempting \
failure is to make a candidate that survived the critic sound promising, to \
predict profit, or to nudge the operator toward a decision. Do none of it. If \
anything, lean toward the critic's skepticism. A neutral, slightly skeptical \
report is correct; a persuasive one is a failure.

WHAT "survive_for_now" MEANS. Every queued candidate carries the critic's \
verdict survive_for_now. This means ONLY "the critic could not kill it on this \
evidence" — it is NOT "validated", NOT "good", NOT "approved", NOT "ready". You \
must preserve that exactly. Frame each queued candidate as: "the critic did not \
kill this; here is what it remains worried about." Never frame it as a good or \
promising strategy. Surface the critic's surviving CONCERNS at least as \
prominently as any metric.

INTEGRITY (any violation hard-rejects the whole output):
- COPY every number VERBATIM from the evidence: the cycle counts and cost, and \
each candidate's in-sample AND out-of-sample metrics. Never invent, round, or \
"clean up" a number. Inventing or altering one is fabrication.
- REPRODUCE every surviving critic concern for each queued candidate, verbatim \
(both the checklist item and its finding text). You may not drop, soften, or \
merge a concern. Dropping a caveat is the worst thing you can do.
- INCLUDE every queued candidate the cycle produced; reference no candidate that \
was not in the cycle.
- NO recommendation language anywhere: do not say "recommend", "approve", \
"reject", "should deploy", "promising", "validated", "worth deploying", \
"green-light", or any directive or cheerleading phrasing. No execution / \
live-trading language either.
- There is no field in which you may recommend or approve. Do not try.
- copy cycle_id from the user message.

OUTPUT SCHEMA (CycleReport):

- cycle_id (string): copy from the user message.
- schema_version: "1.0".
- headline (string <= 400 chars): a neutral one-line summary of the cycle (what \
ran, how many candidates were proposed / killed / queued). No cheerleading.
- summary (object), copied VERBATIM from the cycle result:
  - cycle_id, outcome (string), pairs_covered (list of the universe pairs),
  - candidates_proposed, candidates_killed, candidates_queued (integers),
  - total_cost_usd (string), duration_seconds (number).
- queued_for_approval (list), one per pending candidate:
  - candidate_id, identity, instrument, timeframe, archetype (from the evidence),
  - critic_verdict (string): always "survive_for_now",
  - in_sample_metrics: object copied VERBATIM from the in-sample evidence,
  - out_of_sample_metrics: object copied VERBATIM from the out-of-sample \
evidence, or null only if the evidence has none,
  - surviving_concerns: the critic's concerns, each {item, finding}, copied \
VERBATIM and COMPLETE,
  - explanation (string <= 800 chars): plain language for the operator, framed \
strictly as "the critic did not kill this; here is what it remains worried \
about." State what the operator is being asked to look at. Do NOT call it good, \
promising, or validated. Do NOT recommend a decision.
- operator_decision_notice (string <= 500 chars): state plainly that the \
decision is the operator's and that nothing in this report authorizes trading. \
No recommendation.

Restate (this is the rule, not advice): you are a faithful translator, not an \
advocate. Survive_for_now means "not yet killed", never "validated". Copy every \
number and every concern verbatim and complete. Never recommend, never approve, \
never predict profit, never cheerlead. Hand the operator the critic's concerns \
honestly, next to the metrics, and let them decide.

A note on tone and trust. The operator relies on you to see the cycle without \
reading every artefact, so a misleading summary is worse than no summary: it \
launders a curve-fit into something that looks examined and safe. Your value is \
precisely that you are boring and exact. Do not smooth over a kill count, do not \
bury a concern in a subordinate clause, do not let an in-sample number outshine \
an out-of-sample collapse, and do not imply momentum toward approval. If a \
candidate survived only because the critic could not yet kill it, say exactly \
that. When you are unsure whether a phrasing is neutral, choose the more \
skeptical wording — under-selling a survivor is harmless, over-selling one is \
the failure this whole gate exists to prevent.\
"""


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class ReportingAgent:
    """Prepares one human-facing cycle report. Implements the
    :class:`core.agents.types.Agent` Protocol; the runner issues the call on the
    DEFAULT tier (CHEAP for short status summaries)."""

    name: str = "reporting_agent"

    def __init__(self, *, tier: ModelTier = ModelTier.DEFAULT) -> None:
        self._tier = tier

    # --------------------------------------------------------- Protocol

    def prepare_call(self, inputs: ReportingRequest) -> AgentCall:
        cycle_json = inputs.cycle_result.model_dump(mode="json")
        queued = {_cid(e): _queued_snapshot(e) for e in inputs.queued_items}
        snapshot: dict[str, Any] = {
            "cycle_id": inputs.run_id,
            "universe": list(inputs.universe),
            "cycle": cycle_json,
            "queued": queued,
        }
        volatile_user = (
            "Summarise this finished cycle honestly for the operator. Copy every "
            "number and every critic concern verbatim; reproduce all concerns; "
            "recommend nothing; frame survivors as 'not killed', never as good.\n\n"
            f"cycle_id: {inputs.run_id}\n"
            f"universe (pairs_covered): {list(inputs.universe)}\n\n"
            f"CYCLE_RESULT:\n{_pretty(cycle_json)}\n\n"
            f"PENDING_APPROVAL_ITEMS (each survived the critic — survive_for_now):\n"
            f"{_pretty(queued)}"
        )
        tier = inputs.tier if inputs.tier is not None else self._tier
        return AgentCall(
            stable_system=_STABLE_SYSTEM,
            volatile_user=volatile_user,
            tier=tier,
            output_model=CycleReport,
            max_output_tokens=4096,
            input_snapshot=snapshot,
        )

    def evaluations(self) -> tuple[StructuralCheck, ...]:
        return _TIER1_CHECKS


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def _cid(entry: ApprovalQueueEntry) -> str:
    return entry.candidate.candidate_id


def _queued_snapshot(entry: ApprovalQueueEntry) -> dict[str, Any]:
    oos = entry.out_of_sample_evidence
    return {
        "identity": entry.identity,
        "instrument": entry.candidate.instrument,
        "timeframe": entry.candidate.timeframe.value,
        "archetype": entry.candidate.archetype.value,
        "critic_verdict": entry.critic_verdict.verdict.value,
        "in_sample_metrics": entry.in_sample_evidence.metrics.model_dump(mode="json"),
        "out_of_sample_metrics": (oos.metrics.model_dump(mode="json") if oos is not None else None),
        "concerns": [
            {"item": c.item.value, "finding": c.finding} for c in entry.critic_verdict.concerns
        ],
    }


# ---------------------------------------------------------------------------
# Tier-1 deterministic checks (always on, hard-reject)
# ---------------------------------------------------------------------------


def _as_report(output: Any) -> CycleReport:
    if not isinstance(output, CycleReport):
        raise TypeError(f"expected CycleReport, got {type(output).__name__}")
    return output


def _queued_for(snapshot: dict[str, Any], candidate_id: str) -> dict[str, Any] | None:
    value = snapshot.get("queued", {}).get(candidate_id)
    return value if isinstance(value, dict) else None


def _check_cycle_id_preserved(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    r = _as_report(output)
    expected = snapshot.get("cycle_id")
    if expected is not None and r.cycle_id != expected:
        return CheckResult(False, f"cycle_id {r.cycle_id!r} != input {expected!r}")
    if r.summary.cycle_id != r.cycle_id:
        return CheckResult(False, "summary.cycle_id does not match report cycle_id")
    return CheckResult(True)


def _check_summary_verbatim(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    r = _as_report(output)
    cycle = snapshot.get("cycle", {})
    s = r.summary
    if s.outcome != cycle.get("outcome"):
        return CheckResult(False, f"summary.outcome {s.outcome!r} != {cycle.get('outcome')!r}")
    for field_name, value in (
        ("candidates_proposed", s.candidates_proposed),
        ("candidates_killed", s.candidates_killed),
        ("candidates_queued", s.candidates_queued),
    ):
        if value != cycle.get(field_name):
            return CheckResult(
                False, f"summary.{field_name} {value} != cycle {cycle.get(field_name)}"
            )
    if s.total_cost_usd != Decimal(str(cycle.get("total_cost_usd"))):
        return CheckResult(
            False, f"summary.total_cost_usd {s.total_cost_usd} != {cycle.get('total_cost_usd')}"
        )
    expected_duration = float(cycle.get("duration_seconds", 0.0))
    if abs(s.duration_seconds - expected_duration) > max(0.05, 0.01 * abs(expected_duration)):
        return CheckResult(
            False, f"summary.duration_seconds {s.duration_seconds} != {expected_duration}"
        )
    if tuple(s.pairs_covered) != tuple(snapshot.get("universe", [])):
        return CheckResult(False, "summary.pairs_covered != cycle universe")
    return CheckResult(True)


def _check_all_queued_present_and_real(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    r = _as_report(output)
    known = set(snapshot.get("queued", {}).keys())
    reported = {q.candidate_id for q in r.queued_for_approval}
    missing = known - reported
    if missing:
        return CheckResult(False, f"report dropped queued candidate(s): {sorted(missing)}")
    extra = reported - known
    if extra:
        return CheckResult(False, f"report references candidate(s) not in cycle: {sorted(extra)}")
    if len(reported) != len(r.queued_for_approval):
        return CheckResult(False, "report lists a queued candidate more than once")
    return CheckResult(True)


def _check_queued_metrics_verbatim(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    r = _as_report(output)
    for q in r.queued_for_approval:
        snap = _queued_for(snapshot, q.candidate_id)
        if snap is None:
            return CheckResult(False, f"no evidence for queued candidate {q.candidate_id!r}")
        expected_is = BacktestMetricsView(**snap["in_sample_metrics"])
        diffs = q.in_sample_metrics.differing_fields(expected_is)
        if diffs:
            return CheckResult(
                False, f"candidate {q.candidate_id!r} altered in-sample metrics: {', '.join(diffs)}"
            )
        snap_oos = snap.get("out_of_sample_metrics")
        if snap_oos is None and q.out_of_sample_metrics is not None:
            return CheckResult(
                False, f"candidate {q.candidate_id!r} claims OOS metrics the cycle did not produce"
            )
        if snap_oos is not None and q.out_of_sample_metrics is None:
            return CheckResult(
                False, f"candidate {q.candidate_id!r} dropped the out-of-sample metrics"
            )
        if snap_oos is not None and q.out_of_sample_metrics is not None:
            diffs = q.out_of_sample_metrics.differing_fields(BacktestMetricsView(**snap_oos))
            if diffs:
                return CheckResult(
                    False,
                    f"candidate {q.candidate_id!r} altered OOS metrics: {', '.join(diffs)}",
                )
    return CheckResult(True)


def _check_verdict_is_survive(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    """Each queued candidate's reported critic verdict must be survive_for_now
    and match the evidence — the queue only holds survivors, and the report may
    not upgrade the verdict."""
    r = _as_report(output)
    for q in r.queued_for_approval:
        snap = _queued_for(snapshot, q.candidate_id)
        expected = snap.get("critic_verdict") if snap is not None else None
        if q.critic_verdict != "survive_for_now" or q.critic_verdict != expected:
            return CheckResult(
                False,
                f"candidate {q.candidate_id!r} critic_verdict {q.critic_verdict!r} "
                f"!= evidence {expected!r} (must be 'survive_for_now')",
            )
    return CheckResult(True)


def _check_concerns_present_and_complete(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    """Every surviving critic concern must appear in the report, verbatim, with
    nothing dropped and nothing fabricated."""
    r = _as_report(output)
    for q in r.queued_for_approval:
        snap = _queued_for(snapshot, q.candidate_id)
        if snap is None:
            return CheckResult(False, f"no evidence for queued candidate {q.candidate_id!r}")
        expected = {(c["item"], c["finding"]) for c in snap.get("concerns", [])}
        reported = {(c.item.value, c.finding) for c in q.surviving_concerns}
        dropped = expected - reported
        if dropped:
            return CheckResult(
                False,
                f"candidate {q.candidate_id!r} dropped/altered critic concern(s): "
                f"{sorted(item for item, _ in dropped)}",
            )
        fabricated = reported - expected
        if fabricated:
            return CheckResult(
                False,
                f"candidate {q.candidate_id!r} fabricated concern(s): "
                f"{sorted(item for item, _ in fabricated)}",
            )
    return CheckResult(True)


def _check_no_recommendation_or_execution(output: Any, _snapshot: dict[str, Any]) -> CheckResult:
    """Defense-in-depth against editorializing: scan all free text for
    recommendation / execution language (also catches a model_construct bypass
    of the field validators)."""
    r = _as_report(output)
    spans = [r.headline, r.operator_decision_notice]
    spans.extend(q.explanation for q in r.queued_for_approval)
    for text in spans:
        for pattern in (*_RECOMMENDATION_PATTERNS, *_EXECUTION_INTENT_PATTERNS):
            m = pattern.search(text)
            if m:
                return CheckResult(False, f"recommendation/execution language: {m.group()!r}")
    return CheckResult(True)


_TIER1_CHECKS: tuple[StructuralCheck, ...] = (
    StructuralCheck(name="cycle_id_preserved", check=_check_cycle_id_preserved),
    StructuralCheck(name="summary_verbatim", check=_check_summary_verbatim),
    StructuralCheck(name="all_queued_present_and_real", check=_check_all_queued_present_and_real),
    StructuralCheck(name="queued_metrics_verbatim", check=_check_queued_metrics_verbatim),
    StructuralCheck(name="verdict_is_survive", check=_check_verdict_is_survive),
    StructuralCheck(
        name="concerns_present_and_complete", check=_check_concerns_present_and_complete
    ),
    StructuralCheck(
        name="no_recommendation_or_execution", check=_check_no_recommendation_or_execution
    ),
)


# Module-level sanity: this agent must implement the Agent Protocol, and the
# report must have NO field in which it could recommend / approve / deploy.
_: Agent = ReportingAgent.__new__(ReportingAgent)
_REPORT_FIELDS = set(CycleReport.model_fields)
assert not (_REPORT_FIELDS & {"recommendation", "recommended", "approve", "verdict", "decision"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pretty(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ": "), default=str)


__all__ = [
    "ReportingAgent",
    "ReportingRequest",
]
