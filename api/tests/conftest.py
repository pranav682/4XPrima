"""Fixtures: a temp ``data/orchestration`` seeded with real core models, plus a
TestClient over the read-only API. No fabricated shapes — every artefact is a
genuine frozen core model dumped to its persisted JSON form."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.config import ApiSettings
from api.main import create_app
from core.analysis.pair_screener import (
    CorrelationMatrix,
    ExclusionEntry,
    ScreeningReport,
    ShortlistEntry,
)
from core.models import (
    BacktestArtifact,
    BacktestEvidence,
    BacktestMetricsView,
    ChecklistItem,
    CriticVerdict,
    CriticVerdictKind,
    CycleReport,
    CycleReportSummary,
    EquityCurvePoint,
    EvidenceSegment,
    Granularity,
    OverfittingConcern,
    QueuedCandidateReport,
    StrategyArchetype,
    StrategyCandidate,
    StrategyParam,
)
from core.orchestration import (
    ApprovalQueueEntry,
    BacktestArtifactStore,
    RegistryEntry,
    RegistryState,
)
from core.orchestration.orchestrator import CycleOutcome, CycleResult

BASE = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
CYCLE_ID = "cycle-abc123"
SURVIVOR_HASH = "is-survivor-hash"
SURVIVOR_OOS_HASH = "oos-survivor-hash"


def _mv(total: str, sharpe: float, pf: float | None, trades: int) -> BacktestMetricsView:
    return BacktestMetricsView(
        total_return_pct=Decimal(total),
        annualised_return_pct=Decimal(total),
        sharpe_ratio=sharpe,
        sortino_ratio=0.7,
        max_drawdown_pct=Decimal("0.08"),
        win_rate=0.55,
        profit_factor=pf,
        trade_count=trades,
        avg_trade_pnl=Decimal("1.2"),
        exposure_pct=0.3,
    )


def _evidence(
    cid: str, seg: EvidenceSegment, metrics: BacktestMetricsView, cfg: str
) -> BacktestEvidence:
    return BacktestEvidence(
        candidate_id=cid,
        config_hash=cfg,
        pair="USDJPY",
        segment=seg,
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


def _candidate(cid: str, instrument: str = "USDJPY") -> StrategyCandidate:
    return StrategyCandidate(
        candidate_id=cid,
        run_id=CYCLE_ID,
        archetype=StrategyArchetype.MA_CROSSOVER,
        instrument=instrument,
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


_CONCERNS = (
    OverfittingConcern(
        item=ChecklistItem.OUT_OF_SAMPLE_DECAY, finding="Sharpe falls 2.1 -> 0.3 out-of-sample."
    ),
    OverfittingConcern(item=ChecklistItem.TRADE_COUNT, finding="Only 6 out-of-sample trades."),
)


def _survivor_entry() -> RegistryEntry:
    is_m = _mv("0.30", 2.1, 2.4, 40)
    oos_m = _mv("0.02", 0.3, 1.1, 6)
    verdict = CriticVerdict(
        candidate_id="cand-survivor",
        in_sample_config_hash=SURVIVOR_HASH,
        oos_config_hash=SURVIVOR_OOS_HASH,
        in_sample_metrics=is_m,
        out_of_sample_metrics=oos_m,
        verdict=CriticVerdictKind.SURVIVE_FOR_NOW,
        concerns=_CONCERNS,
        assessment="Survived but the OOS sample is thin.",
        caveats="survive_for_now is not validation",
    )
    return RegistryEntry(
        identity="idy-survivor",
        candidate=_candidate("cand-survivor"),
        state=RegistryState.QUEUED_FOR_APPROVAL,
        run_id=CYCLE_ID,
        in_sample_evidence=_evidence(
            "cand-survivor", EvidenceSegment.IN_SAMPLE, is_m, SURVIVOR_HASH
        ),
        out_of_sample_evidence=_evidence(
            "cand-survivor", EvidenceSegment.OUT_OF_SAMPLE, oos_m, SURVIVOR_OOS_HASH
        ),
        critic_verdict=verdict,
        created_at=BASE,
        updated_at=BASE + timedelta(minutes=5),
    )


def _killed_entry() -> RegistryEntry:
    is_m = _mv("-0.15", -1.4, 0.4, 40)
    oos_m = _mv("-0.20", -1.9, 0.3, 5)
    verdict = CriticVerdict(
        candidate_id="cand-killed",
        in_sample_config_hash="is-killed-hash",
        oos_config_hash="oos-killed-hash",
        in_sample_metrics=is_m,
        out_of_sample_metrics=oos_m,
        verdict=CriticVerdictKind.KILL,
        concerns=(
            OverfittingConcern(item=ChecklistItem.OUT_OF_SAMPLE_DECAY, finding="OOS collapse."),
        ),
        assessment="Hard kill: negative in-sample, worse out-of-sample.",
        caveats="kill is the default",
    )
    return RegistryEntry(
        identity="idy-killed",
        candidate=_candidate("cand-killed", instrument="EURUSD"),
        state=RegistryState.KILLED,
        run_id=CYCLE_ID,
        in_sample_evidence=_evidence(
            "cand-killed", EvidenceSegment.IN_SAMPLE, is_m, "is-killed-hash"
        ),
        out_of_sample_evidence=_evidence(
            "cand-killed", EvidenceSegment.OUT_OF_SAMPLE, oos_m, "oos-killed-hash"
        ),
        critic_verdict=verdict,
        created_at=BASE,
        updated_at=BASE + timedelta(minutes=4),
    )


def _queue_entry() -> ApprovalQueueEntry:
    survivor = _survivor_entry()
    assert survivor.in_sample_evidence is not None
    assert survivor.critic_verdict is not None
    return ApprovalQueueEntry(
        entry_id=f"{CYCLE_ID}:idy-survivor",
        cycle_id=CYCLE_ID,
        identity="idy-survivor",
        candidate=survivor.candidate,
        in_sample_evidence=survivor.in_sample_evidence,
        out_of_sample_evidence=survivor.out_of_sample_evidence,
        critic_verdict=survivor.critic_verdict,
        created_at=BASE + timedelta(minutes=5),
    )


def _cycle() -> CycleResult:
    return CycleResult(
        cycle_id=CYCLE_ID,
        outcome=CycleOutcome.COMPLETED,
        started_at=BASE,
        ended_at=BASE + timedelta(seconds=42),
        duration_seconds=42.5,
        total_cost_usd=Decimal("0.1734"),
        stage_costs_usd={"market_context": "0.03", "critic": "0.10"},
        candidates_proposed=3,
        candidates_killed=2,
        candidates_queued=1,
        queued_identities=("idy-survivor",),
        abort_reason=None,
    )


def _older_cycle() -> CycleResult:
    return CycleResult(
        cycle_id="cycle-older",
        outcome=CycleOutcome.ABORTED_BUDGET,
        started_at=BASE - timedelta(days=1),
        ended_at=BASE - timedelta(days=1) + timedelta(seconds=10),
        duration_seconds=10.0,
        total_cost_usd=Decimal("0.05"),
        stage_costs_usd={"market_context": "0.03"},
        candidates_proposed=0,
        candidates_killed=0,
        candidates_queued=0,
        queued_identities=(),
        abort_reason="per-cycle budget would be exceeded before stage 'strategy_lab'",
    )


def _report() -> CycleReport:
    survivor = _survivor_entry()
    assert survivor.in_sample_evidence is not None and survivor.out_of_sample_evidence is not None
    assert survivor.critic_verdict is not None
    return CycleReport(
        cycle_id=CYCLE_ID,
        headline="Cycle complete: 3 proposed, 2 killed, 1 queued for the operator to review.",
        summary=CycleReportSummary(
            cycle_id=CYCLE_ID,
            outcome="completed",
            pairs_covered=("USDJPY", "EURUSD"),
            candidates_proposed=3,
            candidates_killed=2,
            candidates_queued=1,
            total_cost_usd=Decimal("0.1734"),
            duration_seconds=42.5,
        ),
        queued_for_approval=(
            QueuedCandidateReport(
                candidate_id="cand-survivor",
                identity="idy-survivor",
                instrument="USDJPY",
                timeframe=Granularity.H1,
                archetype=StrategyArchetype.MA_CROSSOVER,
                critic_verdict="survive_for_now",
                in_sample_metrics=survivor.in_sample_evidence.metrics,
                out_of_sample_metrics=survivor.out_of_sample_evidence.metrics,
                surviving_concerns=survivor.critic_verdict.concerns,
                explanation="The critic did not kill this; it remains worried about OOS decay.",
            ),
        ),
        operator_decision_notice="The decision is the operator's; nothing here authorizes trading.",
    )


def _artifact(config_hash: str, segment: EvidenceSegment, net: str) -> BacktestArtifact:
    start = Decimal("100000")
    end = start + Decimal(net)
    curve = tuple(
        EquityCurvePoint(
            bar_index=i,
            time=BASE + timedelta(hours=i),
            equity=start + (Decimal(net) * Decimal(i) / Decimal(4)),
            drawdown_pct=Decimal("0.00"),
        )
        for i in range(5)
    )
    return BacktestArtifact(
        config_hash=config_hash,
        candidate_id="cand-survivor",
        pair="USDJPY",
        segment=segment,
        window_start=BASE,
        window_end=BASE + timedelta(hours=4),
        starting_balance=start,
        ending_balance=end,
        ending_equity=end,
        peak_equity=end if end > start else start,
        net_pnl=Decimal(net),
        return_pct=Decimal(net) / start,
        max_drawdown_pct=Decimal("0.02"),
        trade_count=40 if segment == EvidenceSegment.IN_SAMPLE else 6,
        cost_total=Decimal("214.50"),
        bars_processed=5,
        halted_due_to_kill_switch=False,
        equity_curve=curve,
    )


def _write_store(path: Path, entries: dict[str, Any]) -> None:
    path.write_text(json.dumps({"schema_version": "1.0", "entries": entries}, default=str))


def _seed(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "cycles").mkdir(exist_ok=True)
    (data_dir / "reports").mkdir(exist_ok=True)

    reg = {
        e.identity: json.loads(e.model_dump_json()) for e in (_survivor_entry(), _killed_entry())
    }
    _write_store(data_dir / "registry.json", reg)

    queue_entries = [json.loads(_queue_entry().model_dump_json())]
    (data_dir / "approval_queue.json").write_text(
        json.dumps({"schema_version": "1.0", "entries": queue_entries}, default=str)
    )

    for cycle in (_cycle(), _older_cycle()):
        (data_dir / "cycles" / f"{cycle.cycle_id}.json").write_text(cycle.model_dump_json())
    (data_dir / "reports" / f"{CYCLE_ID}.json").write_text(_report().model_dump_json())

    # Equity-curve artifacts for the SURVIVOR only (the killed candidate has
    # none, so the "honestly unavailable" path stays testable).
    artifacts = BacktestArtifactStore(data_dir / "backtests")
    artifacts.save(_artifact(SURVIVOR_HASH, EvidenceSegment.IN_SAMPLE, "8200"))
    artifacts.save(_artifact(SURVIVOR_OOS_HASH, EvidenceSegment.OUT_OF_SAMPLE, "360"))

    (data_dir / "universe.json").write_text(_screening_report().model_dump_json())


def _screening_report() -> ScreeningReport:
    return ScreeningReport(
        as_of=BASE,
        granularity=Granularity.D,
        lookback_count=500,
        candidate_pairs=("EURUSD", "GBPUSD", "USDCHF", "NZDUSD"),
        profiles=(),
        correlation=CorrelationMatrix(pairs=("EURUSD", "GBPUSD"), matrix=((1.0, 0.2), (0.2, 1.0))),
        shortlist=(
            ShortlistEntry(
                pair="EURUSD",
                selection_rank=1,
                cost_to_move=Decimal("0.02"),
                max_correlation_with_selected=0.0,
                reason="lowest cost-to-move among eligible (spread/ATR=0.02)",
            ),
            ShortlistEntry(
                pair="GBPUSD",
                selection_rank=2,
                cost_to_move=Decimal("0.03"),
                max_correlation_with_selected=0.2,
                reason="cost-to-move spread/ATR=0.03; |corr| 0.20 with selected <= 0.80",
            ),
        ),
        excluded=(
            ExclusionEntry(pair="USDCHF", reason="cost-to-move spread/ATR 3.50 > 0.25"),
            ExclusionEntry(pair="NZDUSD", reason="|correlation| 0.92 with EURUSD exceeds 0.80"),
        ),
    )


@pytest.fixture
def seeded_client(tmp_path: Path) -> Iterator[TestClient]:
    _seed(tmp_path / "data")
    app = create_app(ApiSettings(data_dir=tmp_path / "data"))
    with TestClient(app) as client:
        yield client


@pytest.fixture
def empty_client(tmp_path: Path) -> Iterator[TestClient]:
    """Day-one state: the data dir does not exist at all."""
    app = create_app(ApiSettings(data_dir=tmp_path / "does-not-exist"))
    with TestClient(app) as client:
        yield client
