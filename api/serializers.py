"""Verbatim serializers: core models -> JSON-safe dicts.

Every value is produced by ``model_dump(mode="json")``, so ``Decimal`` is a
string and ``datetime`` is ISO-8601 — numbers pass through exactly as persisted,
never re-floated or recomputed. Composite responses only *assemble* existing
serialized values; they compute nothing.
"""

from __future__ import annotations

from typing import Any

from core.analysis.pair_screener import ScreeningReport
from core.models import BacktestArtifact, CycleReport
from core.orchestration import ApprovalQueueEntry, CycleResult, RegistryEntry

# Curve points are not persisted in BacktestEvidence (only summary metrics are),
# and the API must not re-run the backtest to re-derive them. So we say so,
# honestly, instead of fabricating a line.
EQUITY_CURVE_NOTICE = (
    "Equity-curve points are not persisted in BacktestEvidence — only summary "
    "metrics are stored. The API does not re-run the backtest (it must not "
    "re-derive numbers), so no per-bar curve is available for this slice."
)

_CYCLE_SUMMARY_FIELDS = (
    "cycle_id",
    "outcome",
    "started_at",
    "ended_at",
    "duration_seconds",
    "total_cost_usd",
    "candidates_proposed",
    "candidates_killed",
    "candidates_queued",
    "abort_reason",
)


def cycle_summary(cycle: CycleResult) -> dict[str, Any]:
    full = cycle.model_dump(mode="json")
    return {key: full[key] for key in _CYCLE_SUMMARY_FIELDS}


def cycle_detail(cycle: CycleResult) -> dict[str, Any]:
    return cycle.model_dump(mode="json")


def registry_entry(entry: RegistryEntry) -> dict[str, Any]:
    return entry.model_dump(mode="json")


def approval_item(entry: ApprovalQueueEntry, report: CycleReport | None) -> dict[str, Any]:
    """The queued entry plus the reporting-agent's framing for this candidate,
    if a CycleReport was generated. The critic's verdict + concerns already live
    on the entry; the frontend shows them whether or not a report exists."""
    data = entry.model_dump(mode="json")
    explanation: str | None = None
    if report is not None:
        for queued in report.queued_for_approval:
            if queued.candidate_id == entry.candidate.candidate_id:
                explanation = queued.explanation
                break
    data["report_explanation"] = explanation
    return data


def backtest_detail(
    config_hash: str,
    entry: RegistryEntry,
    *,
    in_sample_artifact: BacktestArtifact | None = None,
    out_of_sample_artifact: BacktestArtifact | None = None,
) -> dict[str, Any]:
    ins = entry.in_sample_evidence
    oos = entry.out_of_sample_evidence
    verdict = entry.critic_verdict
    have_curve = in_sample_artifact is not None or out_of_sample_artifact is not None
    return {
        "config_hash": config_hash,
        "identity": entry.identity,
        "state": entry.state.value,
        "candidate": entry.candidate.model_dump(mode="json"),
        "in_sample": ins.model_dump(mode="json") if ins is not None else None,
        "out_of_sample": oos.model_dump(mode="json") if oos is not None else None,
        "critic_verdict": verdict.model_dump(mode="json") if verdict is not None else None,
        # Rich artifacts (equity curve + annotations), verbatim, when persisted.
        "in_sample_artifact": (
            in_sample_artifact.model_dump(mode="json") if in_sample_artifact is not None else None
        ),
        "out_of_sample_artifact": (
            out_of_sample_artifact.model_dump(mode="json")
            if out_of_sample_artifact is not None
            else None
        ),
        "equity_curve_available": have_curve,
        "equity_curve_notice": "" if have_curve else EQUITY_CURVE_NOTICE,
    }


EMPTY_UNIVERSE: dict[str, Any] = {
    "available": False,
    "as_of": None,
    "granularity": None,
    "lookback_count": None,
    "candidate_pairs": [],
    "admitted": [],
    "dropped": [],
    "correlation": {"pairs": [], "matrix": []},
    "profiles": [],
}


def universe_view(report: ScreeningReport) -> dict[str, Any]:
    """The screener's STRUCTURAL decisions — admitted shortlist + dropped pairs
    with reasons + the return-correlation matrix + per-pair structural profiles.

    Structural fields ONLY: the screener never ranks by historical return, so
    nothing here is a profitability score (a test asserts no return field leaks)."""
    d = report.model_dump(mode="json")
    return {
        "available": True,
        "as_of": d["as_of"],
        "granularity": d["granularity"],
        "lookback_count": d["lookback_count"],
        "candidate_pairs": d["candidate_pairs"],
        # shortlist = admitted (each carries the structural selection reason);
        # excluded = dropped (each carries the structural drop reason).
        "admitted": d["shortlist"],
        "dropped": d["excluded"],
        "correlation": d["correlation"],
        "profiles": d["profiles"],
    }


__all__ = [
    "EMPTY_UNIVERSE",
    "EQUITY_CURVE_NOTICE",
    "approval_item",
    "backtest_detail",
    "cycle_detail",
    "cycle_summary",
    "registry_entry",
    "universe_view",
]
