"""Seed the dashboard's data store with a DEMO cycle — for local preview only.

This runs the REAL deterministic engine + the REAL pair screener on synthetic
candles, so evidence, metrics, equity curves, and the screener's structural
decisions are all genuine output (not fabricated). The only illustrative parts
are the synthetic price series and the demo critic verdict prose (the real critic
is an LLM; here the verdict KIND follows the real out-of-sample numbers). Run a
real cycle with ``python -m scripts.run_orchestrator`` for fully real data.

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
from core.analysis.pair_screener import PairScreener, ScreenConfig
from core.models import (
    Candle,
    ChecklistItem,
    CriticVerdict,
    CriticVerdictKind,
    Granularity,
    OverfittingConcern,
    Quote,
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

# The 5-major default trading universe (what gets backtested).
MAJORS: tuple[tuple[str, str, float, float, float, str, str], ...] = (
    # (candidate_id, pair, base_price, per-bar trend, cycle amplitude, fast, slow)
    ("mac-usdjpy", "USDJPY", 150.0, 0.010, 1.20, "10", "30"),
    ("mac-eurusd", "EURUSD", 1.100, -0.00002, 0.0040, "12", "48"),
    ("mac-gbpusd", "GBPUSD", 1.270, 0.00003, 0.0060, "10", "40"),
    ("mac-usdcad", "USDCAD", 1.360, -0.00001, 0.0050, "14", "42"),
    ("mac-audusd", "AUDUSD", 0.660, 0.00002, 0.0035, "12", "36"),
)


# ---------------------------------------------------------------------------
# Synthetic candles (H1 for backtests, D for screening) — real engine consumes
# ---------------------------------------------------------------------------


def _h1_candles(pair: str, base: float, trend: float, amp: float, n: int = 420) -> list[Candle]:
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


class _H1Provider:
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


def _daily_candles(pair: str, base: float, returns: list[float], n: int) -> list[Candle]:
    """Daily candles from a per-bar return series (screening cadence)."""
    pip = Decimal("0.01") if pair.endswith("JPY") else Decimal("0.0001")
    out: list[Candle] = []
    px = base
    for i in range(min(n, len(returns))):
        px = px * (1.0 + returns[i])
        # Skip weekends so the screener's coverage/gap logic is clean.
        day = BASE + timedelta(days=i)
        if day.weekday() >= 5:
            continue
        p = Decimal(str(round(px, 5)))
        out.append(
            Candle(
                pair=pair,
                granularity=Granularity.D,
                time=day.replace(hour=21),
                open=p,
                high=p + pip * 20,
                low=p - pip * 20,
                close=p,
                volume=1000,
                complete=True,
            )
        )
    return out


def _returns(phase: float, scale: float = 0.004, n: int = 620) -> list[float]:
    return [
        scale * math.sin(i / 13.0 + phase) + 0.5 * scale * math.sin(i / 5.0 + phase * 2)
        for i in range(n)
    ]


class _DailyProvider:
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
        c = self._d.get(pair.replace("/", "").replace("_", "").upper(), [])
        return c[-count:] if count else c


class _Prices:
    """Live-quote stub for cost-to-move (spread). Wide spread -> dropped for cost."""

    def __init__(self, spreads: dict[str, float], last: dict[str, float]) -> None:
        self._spreads = spreads
        self._last = last

    def get_quote(self, pair: str) -> Quote:
        key = pair.replace("/", "").replace("_", "").upper()
        mid = Decimal(str(self._last[key]))
        half = Decimal(str(self._spreads[key])) / 2
        return Quote(pair=key, bid=mid - half, ask=mid + half, timestamp=BASE)


# ---------------------------------------------------------------------------
# Demo critic verdict (KIND follows the real OOS numbers)
# ---------------------------------------------------------------------------


def _verdict(rb: RobustnessEvidence) -> CriticVerdict:
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
                        f"Sharpe decays {ins.metrics.sharpe_ratio:.2f} (in-sample) -> "
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
            "survive_for_now: out-of-sample stayed positive, on the caveats below."
            if oos_positive
            else "kill: out-of-sample did not hold up."
        ),
        caveats="survive_for_now is not validation; nothing here authorizes trading.",
    )


def _candidate(cid: str, pair: str, fast: str, slow: str) -> StrategyCandidate:
    stop = "90" if pair.endswith("JPY") else "80"
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


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def _seed_trading() -> None:
    provider = _H1Provider(
        {pair: _h1_candles(pair, base, trend, amp) for _, pair, base, trend, amp, _, _ in MAJORS}
    )
    config = BacktestRunConfig(lookback_count=420, oos_fraction=0.2, min_trade_count=1)
    registry = ChampionChallengerRegistry(DATA / "registry.json")
    artifacts = BacktestArtifactStore(DATA / "backtests")

    queued: list[str] = []
    killed = 0
    now = BASE + timedelta(minutes=5)
    for cid, pair, _base, _trend, _amp, fast, slow in MAJORS:
        cand = _candidate(cid, pair, fast, slow)
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
        oos_s = rb.out_of_sample.metrics.sharpe_ratio if rb.out_of_sample else None
        oos_txt = f" | OOS {oos_s:+.2f}" if oos_s is not None else ""
        is_s = rb.in_sample.metrics.sharpe_ratio
        print(f"  {pair:7} {verdict.verdict.value:16} IS {is_s:+.2f}{oos_txt}")

    _write_cycle(len(MAJORS), killed, len(queued), tuple(queued))


def _seed_universe() -> None:
    """Run the REAL pair screener over a broader candidate set so the Universe
    view shows admits AND drops (correlation / cost / coverage)."""
    # ~740 calendar days -> ~528 weekday candles, comfortably over the 500 the
    # screener expects (so admitted pairs read full coverage).
    n = 740
    aud = _returns(phase=0.0, scale=0.004, n=n)
    candles: dict[str, list[Candle]] = {
        "EURUSD": _daily_candles("EURUSD", 1.10, _returns(0.7, 0.0038, n), n),
        "USDJPY": _daily_candles("USDJPY", 150.0, _returns(2.1, 0.0045, n), n),
        "GBPUSD": _daily_candles("GBPUSD", 1.27, _returns(3.9, 0.0042, n), n),
        "USDCAD": _daily_candles("USDCAD", 1.36, _returns(5.2, 0.0035, n), n),
        "AUDUSD": _daily_candles("AUDUSD", 0.66, aud, n),
        # highly correlated with AUDUSD -> dropped for correlation
        "NZDUSD": _daily_candles(
            "NZDUSD", 0.60, [a * 0.98 + 0.0004 * math.sin(i / 3.0) for i, a in enumerate(aud)], n
        ),
        # normal series but a wide spread (below) -> dropped for cost-to-move
        "USDCHF": _daily_candles("USDCHF", 0.90, _returns(1.3, 0.0030, n), n),
        # too little history -> dropped for coverage
        "EURGBP": _daily_candles("EURGBP", 0.86, _returns(4.4, 0.0033, n), 60),
    }
    last = {p: float(c[-1].close) for p, c in candles.items() if c}
    spreads = {
        # AUDUSD + NZDUSD are the two cheapest, so NZDUSD is evaluated right after
        # AUDUSD is selected and gets dropped for its ~0.98 correlation with it.
        "AUDUSD": 0.00008,
        "NZDUSD": 0.00010,
        "EURUSD": 0.00012,
        "USDJPY": 0.015,
        "GBPUSD": 0.00016,
        "USDCAD": 0.00018,
        "USDCHF": 0.02000,  # WIDE -> cost drop
        "EURGBP": 0.00015,
    }
    screener = PairScreener(_DailyProvider(candles), price_provider=_Prices(spreads, last))
    report = screener.screen(
        ScreenConfig(
            pairs=tuple(candles),
            granularity=Granularity.D,
            lookback_count=500,
            shortlist_size=5,
        ),
        as_of=BASE,
    )
    (DATA / "universe.json").write_text(report.model_dump_json())
    print(
        f"  screened {len(report.candidate_pairs)} candidates -> "
        f"{len(report.shortlist)} admitted, {len(report.excluded)} dropped"
    )


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
        ended_at=BASE + timedelta(seconds=104),
        duration_seconds=104.2,
        total_cost_usd=Decimal("0.2380"),
        stage_costs_usd={
            "market_context": "0.0296",
            "strategy_lab": "0.0250",
            "backtest_agent": "0.0350",
            "critic": "0.1484",
        },
        candidates_proposed=proposed,
        candidates_killed=killed,
        candidates_queued=queued,
        queued_identities=ids,
        abort_reason=None,
    )
    (DATA / "cycles").mkdir(parents=True, exist_ok=True)
    (DATA / "cycles" / f"{CYCLE_ID}.json").write_text(cr.model_dump_json())


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    print("Trading universe (5 majors, real engine):")
    _seed_trading()
    print("Pair screen (real screener):")
    _seed_universe()
    print(f"\nSeeded demo into {DATA}/ — reset any time with: rm -rf data/orchestration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
