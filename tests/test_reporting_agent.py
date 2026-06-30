"""Tests for reporting_agent — fully hermetic (LLM mocked).

The reporting agent's failure mode is editorializing, so the tests focus there:
metrics/costs copied verbatim, every critic concern preserved, no recommendation
or execution language, survivors framed as "not killed" rather than validated,
and the structural absence of any approve/recommend field. Plus tier dispatch,
token budget, and an e2e through AgentRunner.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.agents.evaluation import EvaluationGate
from core.agents.reporting_agent import ReportingAgent, ReportingRequest
from core.agents.runner import AgentRunner
from core.agents.types import AgentBudget
from core.llm_client import AgentResponse, ModelTier, TokenUsage
from core.models import (
    BacktestEvidence,
    BacktestMetricsView,
    ChecklistItem,
    CriticVerdict,
    CriticVerdictKind,
    CycleReport,
    CycleReportSummary,
    EvidenceSegment,
    Granularity,
    OverfittingConcern,
    QueuedCandidateReport,
    StrategyArchetype,
    StrategyCandidate,
    StrategyParam,
)
from core.orchestration import ApprovalQueueEntry
from core.orchestration.orchestrator import CycleOutcome, CycleResult

BASE = datetime(2024, 1, 1, tzinfo=UTC)

CheckFn = Callable[[Any, dict[str, Any]], Any]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mv(total: str = "0.05", sharpe: float = 0.5, pf: float | None = 1.4) -> BacktestMetricsView:
    return BacktestMetricsView(
        total_return_pct=Decimal(total),
        annualised_return_pct=Decimal(total),
        sharpe_ratio=sharpe,
        sortino_ratio=0.7,
        max_drawdown_pct=Decimal("0.08"),
        win_rate=0.55,
        profit_factor=pf,
        trade_count=40,
        avg_trade_pnl=Decimal("1.2"),
        exposure_pct=0.3,
    )


def _evidence(
    cid: str, segment: EvidenceSegment, metrics: BacktestMetricsView, cfg: str
) -> BacktestEvidence:
    return BacktestEvidence(
        candidate_id=cid,
        config_hash=cfg,
        pair="USDJPY",
        segment=segment,
        window_start=BASE,
        window_end=BASE + timedelta(days=30),
        bars_total=100,
        bars_processed=100,
        halted_due_to_kill_switch=False,
        halt_reason=None,
        n_signals_proposed=10,
        n_signals_accepted=8,
        n_signals_rejected=2,
        starting_balance=Decimal("100000"),
        ending_equity=Decimal("105000"),
        cost_total=Decimal("5"),
        metrics=metrics,
        gates=(),
        gates_all_passed=True,
    )


def _verdict(
    cid: str,
    concerns: tuple[OverfittingConcern, ...],
    is_m: BacktestMetricsView,
    oos_m: BacktestMetricsView,
) -> CriticVerdict:
    return CriticVerdict(
        candidate_id=cid,
        in_sample_config_hash="is-h",
        oos_config_hash="oos-h",
        in_sample_metrics=is_m,
        out_of_sample_metrics=oos_m,
        verdict=CriticVerdictKind.SURVIVE_FOR_NOW,
        concerns=concerns,
        assessment="Survived the attacks but the out-of-sample sample is thin.",
        caveats="survive_for_now is not validation",
    )


def _candidate(cid: str) -> StrategyCandidate:
    return StrategyCandidate(
        candidate_id=cid,
        run_id="cyc-1",
        archetype=StrategyArchetype.MA_CROSSOVER,
        instrument="USDJPY",
        timeframe=Granularity.H1,
        parameters=(
            StrategyParam(name="fast_period", value=Decimal("5")),
            StrategyParam(name="slow_period", value=Decimal("15")),
            StrategyParam(name="size", value=Decimal("1000")),
            StrategyParam(name="stop_distance_pips", value=Decimal("80")),
        ),
        parameter_ranges=(),
        rationale="fixture",
    )


def _entry(
    cid: str = "c1", concerns: tuple[OverfittingConcern, ...] | None = None
) -> ApprovalQueueEntry:
    is_m = _mv(total="0.30", sharpe=2.1, pf=2.4)
    oos_m = _mv(total="0.02", sharpe=0.3, pf=1.1)
    if concerns is None:
        concerns = (
            OverfittingConcern(
                item=ChecklistItem.OUT_OF_SAMPLE_DECAY,
                finding="Sharpe falls 2.1 -> 0.3 out-of-sample.",
            ),
            OverfittingConcern(
                item=ChecklistItem.TRADE_COUNT, finding="Only 6 out-of-sample trades."
            ),
        )
    return ApprovalQueueEntry(
        entry_id=f"cyc-1:{cid}",
        cycle_id="cyc-1",
        identity=f"idy-{cid}",
        candidate=_candidate(cid),
        in_sample_evidence=_evidence(cid, EvidenceSegment.IN_SAMPLE, is_m, "is-h"),
        out_of_sample_evidence=_evidence(cid, EvidenceSegment.OUT_OF_SAMPLE, oos_m, "oos-h"),
        critic_verdict=_verdict(cid, concerns, is_m, oos_m),
        created_at=BASE,
    )


def _cycle(proposed: int = 3, killed: int = 2, queued: int = 1) -> CycleResult:
    return CycleResult(
        cycle_id="cyc-1",
        outcome=CycleOutcome.COMPLETED,
        started_at=BASE,
        ended_at=BASE + timedelta(seconds=12),
        duration_seconds=12.5,
        total_cost_usd=Decimal("0.17"),
        stage_costs_usd={"market_context": "0.03", "critic": "0.10"},
        candidates_proposed=proposed,
        candidates_killed=killed,
        candidates_queued=queued,
        queued_identities=("idy-c1",),
        abort_reason=None,
    )


def _request(
    entries: tuple[ApprovalQueueEntry, ...] | None = None, tier: ModelTier = ModelTier.DEFAULT
) -> ReportingRequest:
    entries = entries if entries is not None else (_entry(),)
    return ReportingRequest(
        run_id="cyc-1",
        cycle_result=_cycle(queued=len(entries)),
        universe=("USDJPY", "EURUSD"),
        queued_items=entries,
        tier=tier,
    )


def _verbatim_report(req: ReportingRequest) -> CycleReport:
    """The honest, fully-verbatim report a perfect run would produce."""
    cycle = req.cycle_result
    queued = []
    for e in req.queued_items:
        oos = e.out_of_sample_evidence
        queued.append(
            QueuedCandidateReport(
                candidate_id=e.candidate.candidate_id,
                identity=e.identity,
                instrument=e.candidate.instrument,
                timeframe=e.candidate.timeframe,
                archetype=e.candidate.archetype,
                critic_verdict="survive_for_now",
                in_sample_metrics=e.in_sample_evidence.metrics,
                out_of_sample_metrics=oos.metrics if oos is not None else None,
                surviving_concerns=e.critic_verdict.concerns,
                explanation="The critic did not kill this; it remains worried about the concerns.",
            )
        )
    return CycleReport(
        cycle_id=cycle.cycle_id,
        headline="Cycle complete: 3 proposed, 2 killed, 1 queued for the operator to review.",
        summary=CycleReportSummary(
            cycle_id=cycle.cycle_id,
            outcome=cycle.outcome.value,
            pairs_covered=req.universe,
            candidates_proposed=cycle.candidates_proposed,
            candidates_killed=cycle.candidates_killed,
            candidates_queued=cycle.candidates_queued,
            total_cost_usd=cycle.total_cost_usd,
            duration_seconds=cycle.duration_seconds,
        ),
        queued_for_approval=tuple(queued),
        operator_decision_notice="The decision is the operator's; nothing here authorizes trading.",
    )


def _checks() -> dict[str, CheckFn]:
    agent = ReportingAgent.__new__(ReportingAgent)
    return {c.name: c.check for c in agent.evaluations()}


def _snapshot(req: ReportingRequest) -> dict[str, Any]:
    return ReportingAgent().prepare_call(req).input_snapshot


def _run_all_checks(report: CycleReport, snapshot: dict[str, Any]) -> dict[str, Any]:
    return {name: check(report, snapshot) for name, check in _checks().items()}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_verbatim_report_passes_every_tier1_check() -> None:
    req = _request()
    report = _verbatim_report(req)
    results = _run_all_checks(report, _snapshot(req))
    assert all(r.passed for r in results.values()), {
        k: v.reason for k, v in results.items() if not v.passed
    }


def test_summary_counts_and_cost_are_verbatim() -> None:
    req = _request()
    report = _verbatim_report(req)
    s = report.summary
    assert (s.candidates_proposed, s.candidates_killed, s.candidates_queued) == (3, 2, 1)
    assert s.total_cost_usd == Decimal("0.17")
    assert tuple(s.pairs_covered) == ("USDJPY", "EURUSD")


# ---------------------------------------------------------------------------
# Tier-1 catches fabrication / dropped concerns / editorializing
# ---------------------------------------------------------------------------


def test_tier1_catches_fabricated_metric() -> None:
    req = _request()
    report = _verbatim_report(req)
    q = report.queued_for_approval[0]
    tampered = q.model_copy(update={"in_sample_metrics": _mv(total="0.99", sharpe=9.9)})
    report = report.model_copy(update={"queued_for_approval": (tampered,)})
    res = _run_all_checks(report, _snapshot(req))
    assert not res["queued_metrics_verbatim"].passed


def test_tier1_catches_altered_oos_metric() -> None:
    req = _request()
    report = _verbatim_report(req)
    q = report.queued_for_approval[0]
    tampered = q.model_copy(update={"out_of_sample_metrics": _mv(total="0.40", sharpe=3.0)})
    report = report.model_copy(update={"queued_for_approval": (tampered,)})
    res = _run_all_checks(report, _snapshot(req))
    assert not res["queued_metrics_verbatim"].passed


def test_tier1_catches_dropped_critic_concern() -> None:
    req = _request()
    report = _verbatim_report(req)
    q = report.queued_for_approval[0]
    # Keep only the first of the two surviving concerns.
    dropped = q.model_copy(update={"surviving_concerns": (q.surviving_concerns[0],)})
    report = report.model_copy(update={"queued_for_approval": (dropped,)})
    res = _run_all_checks(report, _snapshot(req))
    assert not res["concerns_present_and_complete"].passed


def test_tier1_catches_fabricated_concern() -> None:
    req = _request()
    report = _verbatim_report(req)
    q = report.queued_for_approval[0]
    extra = (
        *q.surviving_concerns,
        OverfittingConcern(item=ChecklistItem.COST_SENSITIVITY, finding="made up"),
    )
    fabricated = q.model_copy(update={"surviving_concerns": extra})
    report = report.model_copy(update={"queued_for_approval": (fabricated,)})
    res = _run_all_checks(report, _snapshot(req))
    assert not res["concerns_present_and_complete"].passed


def test_tier1_catches_dropped_queued_candidate() -> None:
    req = _request(entries=(_entry("c1"), _entry("c2")))
    report = _verbatim_report(req)
    # Drop the second candidate from the report.
    report = report.model_copy(update={"queued_for_approval": (report.queued_for_approval[0],)})
    res = _run_all_checks(report, _snapshot(req))
    assert not res["all_queued_present_and_real"].passed


def test_recommendation_language_rejected_at_construction() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QueuedCandidateReport(
            candidate_id="c1",
            identity="idy-c1",
            instrument="USDJPY",
            timeframe=Granularity.H1,
            archetype=StrategyArchetype.MA_CROSSOVER,
            critic_verdict="survive_for_now",
            in_sample_metrics=_mv(),
            out_of_sample_metrics=None,
            surviving_concerns=(),
            explanation="I recommend approving this promising strategy.",
        )


def test_tier1_catches_injected_recommendation_bypassing_validator() -> None:
    req = _request()
    report = _verbatim_report(req)
    q = report.queued_for_approval[0]
    # model_construct bypasses the field validator → Tier-1 must still catch it.
    bad = q.model_construct(
        **{**q.__dict__, "explanation": "You should approve this; it looks promising."}
    )
    report = report.model_copy(update={"queued_for_approval": (bad,)})
    res = _run_all_checks(report, _snapshot(req))
    assert not res["no_recommendation_or_execution"].passed


def test_tier1_catches_execution_language_in_notice() -> None:
    req = _request()
    report = _verbatim_report(req)
    bad = report.model_construct(
        **{**report.__dict__, "operator_decision_notice": "Deploy this to the live account now."}
    )
    res = _run_all_checks(bad, _snapshot(req))
    assert not res["no_recommendation_or_execution"].passed


def test_tier1_catches_upgraded_verdict() -> None:
    req = _request()
    report = _verbatim_report(req)
    q = report.queued_for_approval[0]
    upgraded = q.model_construct(**{**q.__dict__, "critic_verdict": "approved"})
    report = report.model_copy(update={"queued_for_approval": (upgraded,)})
    res = _run_all_checks(report, _snapshot(req))
    assert not res["verdict_is_survive"].passed


# ---------------------------------------------------------------------------
# Framing + structural non-recommendation
# ---------------------------------------------------------------------------


def test_survivor_reported_with_concerns_and_not_as_validated() -> None:
    req = _request()
    report = _verbatim_report(req)
    q = report.queued_for_approval[0]
    assert q.critic_verdict == "survive_for_now"
    # both critic concerns are present
    assert {c.item for c in q.surviving_concerns} == {
        ChecklistItem.OUT_OF_SAMPLE_DECAY,
        ChecklistItem.TRADE_COUNT,
    }
    # the report never claims validation
    assert "validated" not in q.explanation.lower()
    assert "good strategy" not in q.explanation.lower()


def test_report_has_no_recommendation_or_approve_field() -> None:
    for model in (CycleReport, QueuedCandidateReport, CycleReportSummary):
        fields = set(model.model_fields)
        assert not (
            fields
            & {"recommendation", "recommended", "approve", "verdict", "decision", "should_deploy"}
        )


# ---------------------------------------------------------------------------
# Tier dispatch + token budget
# ---------------------------------------------------------------------------


def test_default_and_cheap_tier_dispatch_assemble_cleanly() -> None:
    default_call = ReportingAgent().prepare_call(_request(tier=ModelTier.DEFAULT))
    cheap_call = ReportingAgent().prepare_call(_request(tier=ModelTier.CHEAP))
    assert default_call.tier == ModelTier.DEFAULT
    assert cheap_call.tier == ModelTier.CHEAP
    # CHEAP-tier call assembles with the same stable prefix + output model.
    assert cheap_call.output_model is CycleReport
    assert cheap_call.stable_system == default_call.stable_system
    assert cheap_call.input_snapshot["cycle_id"] == "cyc-1"


def test_token_budget_volatile_user_is_bounded() -> None:
    call = ReportingAgent().prepare_call(
        _request(entries=(_entry("c1"), _entry("c2"), _entry("c3")))
    )
    assert len(call.volatile_user) < 16000


# ---------------------------------------------------------------------------
# e2e via AgentRunner
# ---------------------------------------------------------------------------


def test_e2e_via_agent_runner() -> None:
    req = _request()
    canned = _verbatim_report(req)
    llm = MagicMock()
    llm.generate_structured.return_value = (
        canned,
        AgentResponse(
            agent_name="reporting_agent",
            run_id="cyc-1:attempt-0",
            tier=ModelTier.DEFAULT,
            model="gpt-5.4",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=3000, cached_tokens=2000, completion_tokens=400),
            extra_metadata={"attempts": "1"},
        ),
    )
    runner = AgentRunner(llm_provider=llm, evaluation_gate=EvaluationGate(), budget=AgentBudget())
    result = runner.run(ReportingAgent(), req, run_id="cyc-1")
    assert result.succeeded
    assert result.eval_verdict is not None and result.eval_verdict.tier1_passed
    assert isinstance(result.output, CycleReport)


# ---------------------------------------------------------------------------
# Optional live smoke
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (os.getenv("RUN_LIVE_TESTS") == "1" and os.getenv("OPENAI_API_KEY")),
    reason="RUN_LIVE_TESTS=1 and OPENAI_API_KEY must be set",
)
def test_live_smoke_integrity_holds() -> None:
    from core.config import OpenAISettings
    from core.llm_client import OpenAIProvider

    req = _request()
    runner = AgentRunner(
        llm_provider=OpenAIProvider(OpenAISettings()),  # type: ignore[call-arg]
        evaluation_gate=EvaluationGate(),
        budget=AgentBudget(),
    )
    result = runner.run(ReportingAgent(), req, run_id="cyc-live")
    assert result.succeeded, result.failure
    assert result.eval_verdict is not None and result.eval_verdict.tier1_passed
    report = result.output
    assert isinstance(report, CycleReport)
    # integrity + no-recommendation + concerns-present all enforced by Tier-1,
    # which passed; double-check concerns survived into the report.
    q = report.queued_for_approval[0]
    assert len(q.surviving_concerns) == 2
