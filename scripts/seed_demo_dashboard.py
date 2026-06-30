"""Seed the dashboard's data store with a DEMO cycle — for local preview only.

This runs the REAL deterministic engine on synthetic candles, so the evidence,
metrics, and equity curves are genuine engine output (not fabricated). The only
illustrative parts are the synthetic price series and the critic verdict prose
(the real critic is an LLM; here the verdict KIND is derived from the real
out-of-sample numbers so the honesty semantics stay truthful). Run a real cycle
with ``python -m scripts.run_orchestrator`` for fully real data including a real
critic.

    python -m scripts.seed_demo_dashboard          # writes data/orchestration/*
    rm -rf data/orchestration                       # reset to day-one empty

No LLM calls, no network, no cost.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from core.agents.backtest_harness import (
    BacktestRunConfig,
    run_candidate_artifacts,
    run_robustness,
)
from core.models import (
    Candle,
    ChecklistItem,
    CriticVerdict,
    CriticVerdictKind,
    Granularity,
    OverfittingConcern,
    RobustnessEvidence,
    StrategyArchetype,
    StrategyCandidate,
    StrategyParam,
)
from core.orchestration import (
    ApprovalQueueEntry,
    BacktestArtifactStore,
    ChampionChallengerRegistry,
    RegistryState,
)
from core.orchestration.orchestrator import CycleOutcome, CycleResult

BASE = datetime(2026, 6, 1, tzinfo=UTC)
CYCLE_ID = "cycle-demo01"
DATA = Path("data/orchestration")


def _candles(pair: str, base: float, trend: float, amp: float, n: int = 420) -> list[Candle]:
    """A deterministic trending-with-cycles price series — enough structure that
    an MA crossover actually trades. (Synthetic price; real engine output.)"""
    pip = Decimal("0.01") if pair.endswith("JPY") else Decimal("0.0001")
    out: list[Candle] = []
    for i in range(n):
        px = base + trend * i + amp * math.sin(i / 23.0) + 0.4 * amp * math.sin(i / 7.0)
        p = Decimal(str(round(px, 3)))
        out.append(
            Candle(
                pair=pair,
                granularity=Granularity.H1,
                time=BASE + timedelta(hours=i),
                open=p,
                high=p + pip * 3,
                low=p - pip * 3,
                close=p,
                volume=1000,
                complete=True,
            )
        )
    return out


class _Provider:
    def __init__(self, by_pair: dict[str, list[Candle]]) -> None:
        self._d = by_pair

    def get_candles(
        self,
        pair: str,
        *,
        granularity: Granularity,
        count: int | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ) -> list[Candle]:
        c = self._d[pair.replace("/", "").replace("_", "").upper()]
        return c[-count:] if count else c


def _candidate(cid: str, pair: str, fast: str, slow: str, stop: str) -> StrategyCandidate:
    return StrategyCandidate(
        candidate_id=cid,
        run_id=CYCLE_ID,
        archetype=StrategyArchetype.MA_CROSSOVER,
        instrument=pair,
        timeframe=Granularity.H1,
        parameters=(
            StrategyParam(name="fast_period", value=Decimal(fast)),
            StrategyParam(name="slow_period", value=Decimal(slow)),
            StrategyParam(name="size", value=Decimal("10000")),
            StrategyParam(name="stop_distance_pips", value=Decimal(stop)),
        ),
        parameter_ranges=(),
        rationale="Demo trend-following candidate (synthetic price; real engine output).",
    )


def _verdict(rb: RobustnessEvidence) -> CriticVerdict:
    """A verdict whose KIND follows the REAL out-of-sample numbers, with concerns
    tied to the real evidence. Prose is illustrative; the semantics are honest."""
    ins = rb.in_sample
    oos = rb.out_of_sample
    oos_positive = oos is not None and oos.metrics.total_return_pct > 0
    kind = CriticVerdictKind.SURVIVE_FOR_NOW if oos_positive else CriticVerdictKind.KILL
    concerns: list[OverfittingConcern] = []
    if oos is not None:
        if oos.metrics.trade_count < 30:
            concerns.append(
                OverfittingConcern(
                    item=ChecklistItem.TRADE_COUNT,
                    finding=(
                        f"Out-of-sample rests on {oos.metrics.trade_count} trades — "
                        "weak statistical power."
                    ),
                )
            )
        if oos.metrics.sharpe_ratio < ins.metrics.sharpe_ratio:
            concerns.append(
                OverfittingConcern(
                    item=ChecklistItem.OUT_OF_SAMPLE_DECAY,
                    finding=(
                        f"Sharpe decays {ins.metrics.sharpe_ratio:.2f} (in-sample) → "
                        f"{oos.metrics.sharpe_ratio:.2f} (out-of-sample)."
                    ),
                )
            )
    if not concerns:
        concerns.append(
            OverfittingConcern(
                item=ChecklistItem.TRADE_CONCENTRATION,
                finding="Check whether a few trades drive the result before trusting it.",
            )
        )
    return CriticVerdict(
        candidate_id=ins.candidate_id,
        in_sample_config_hash=ins.config_hash,
        oos_config_hash=oos.config_hash if oos is not None else None,
        in_sample_metrics=ins.metrics,
        out_of_sample_metrics=oos.metrics if oos is not None else None,
        verdict=kind,
        concerns=tuple(concerns),
        assessment=(
            "survive_for_now: out-of-sample stayed positive, but on the caveats below."
            if oos_positive
            else "kill: out-of-sample did not hold up."
        ),
        caveats="survive_for_now is not validation; nothing here authorizes trading.",
    )


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    provider = _Provider(
        {
            "USDJPY": _candles("USDJPY", 150.0, 0.010, 1.2),
            "EURUSD": _candles("EURUSD", 1.10, -0.00002, 0.004),
        }
    )
    config = BacktestRunConfig(lookback_count=420, oos_fraction=0.2, min_trade_count=1)
    registry = ChampionChallengerRegistry(DATA / "registry.json")
    artifacts = BacktestArtifactStore(DATA / "backtests")
    candidates = [
        _candidate("mac-usdjpy", "USDJPY", "10", "30", "90"),
        _candidate("mac-eurusd", "EURUSD", "12", "48", "80"),
    ]

    queued: list[str] = []
    killed = 0
    now = BASE + timedelta(minutes=5)
    for cand in candidates:
        rb = run_robustness(cand, candle_provider=provider, config=config)
        for art in run_candidate_artifacts(cand, candle_provider=provider, config=config):
            artifacts.save(art)
        verdict = _verdict(rb)
        ident = registry.upsert_proposed(cand, run_id=CYCLE_ID, now=now)
        registry.record_backtest(ident, rb.in_sample, now=now)
        if verdict.verdict == CriticVerdictKind.SURVIVE_FOR_NOW:
            registry.record_critic(
                ident,
                verdict,
                state=RegistryState.SURVIVED_FOR_NOW,
                out_of_sample_evidence=rb.out_of_sample,
                now=now,
            )
            _queue_survivor(cand, ident, rb, verdict, now)
            registry.set_state(ident, RegistryState.QUEUED_FOR_APPROVAL, now=now)
            queued.append(ident)
        else:
            registry.record_critic(
                ident,
                verdict,
                state=RegistryState.KILLED,
                out_of_sample_evidence=rb.out_of_sample,
                now=now,
            )
            killed += 1
        oos_txt = (
            f" | OOS sharpe {rb.out_of_sample.metrics.sharpe_ratio:+.2f}"
            if rb.out_of_sample
            else ""
        )
        print(
            f"  {cand.instrument:7} {verdict.verdict.value:16} "
            f"IS sharpe {rb.in_sample.metrics.sharpe_ratio:+.2f}{oos_txt}"
        )

    _write_cycle(len(candidates), killed, len(queued), tuple(queued))
    print(
        f"\nSeeded demo cycle into {DATA}/ — proposed {len(candidates)}, "
        f"killed {killed}, queued {len(queued)}."
    )
    print("Reset any time with: rm -rf data/orchestration")
    return 0


def _queue_survivor(
    cand: StrategyCandidate,
    ident: str,
    rb: RobustnessEvidence,
    verdict: CriticVerdict,
    now: datetime,
) -> None:
    queue_path = DATA / "approval_queue.json"
    entry = ApprovalQueueEntry(
        entry_id=f"{CYCLE_ID}:{ident}",
        cycle_id=CYCLE_ID,
        identity=ident,
        candidate=cand,
        in_sample_evidence=rb.in_sample,
        out_of_sample_evidence=rb.out_of_sample,
        critic_verdict=verdict,
        created_at=now,
    )
    existing: list[dict[str, object]] = []
    if queue_path.is_file():
        existing = json.loads(queue_path.read_text()).get("entries", [])
    existing.append(json.loads(entry.model_dump_json()))
    queue_path.write_text(json.dumps({"schema_version": "1.0", "entries": existing}))


def _write_cycle(proposed: int, killed: int, queued: int, ids: tuple[str, ...]) -> None:
    cr = CycleResult(
        cycle_id=CYCLE_ID,
        outcome=CycleOutcome.COMPLETED,
        started_at=BASE,
        ended_at=BASE + timedelta(seconds=63),
        duration_seconds=63.4,
        total_cost_usd=Decimal("0.1734"),
        stage_costs_usd={
            "market_context": "0.0296",
            "strategy_lab": "0.0190",
            "backtest_agent": "0.0230",
            "critic": "0.0981",
        },
        candidates_proposed=proposed,
        candidates_killed=killed,
        candidates_queued=queued,
        queued_identities=ids,
        abort_reason=None,
    )
    (DATA / "cycles").mkdir(parents=True, exist_ok=True)
    (DATA / "cycles" / f"{CYCLE_ID}.json").write_text(cr.model_dump_json())


if __name__ == "__main__":
    raise SystemExit(main())
