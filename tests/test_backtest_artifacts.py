"""Tests for the dashboard equity-curve artifacts: the harness builder, the
store round-trip, and the orchestrator persisting them (without touching the
slim LLM-facing evidence)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from core.agents.backtest_harness import (
    BacktestRunConfig,
    run_candidate_artifacts,
)
from core.models import (
    Candle,
    EvidenceSegment,
    Granularity,
    StrategyArchetype,
    StrategyCandidate,
    StrategyParam,
)
from core.orchestration import BacktestArtifactStore

BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _candles(pair: str, start: str, n: int = 120) -> list[Candle]:
    out: list[Candle] = []
    px = Decimal(start)
    step = Decimal("0.20") if pair.endswith("JPY") else Decimal("0.0020")
    for i in range(n):
        px = px + (step if (i // 6) % 2 == 0 else -step)
        out.append(
            Candle(
                pair=pair,
                granularity=Granularity.H1,
                time=BASE + timedelta(hours=i),
                open=px,
                high=px + step / 4,
                low=px - step / 4,
                close=px,
                volume=1000,
                complete=True,
            )
        )
    return out


class _Provider:
    def __init__(self, candles: list[Candle]) -> None:
        self._c = candles

    def get_candles(
        self,
        pair: str,
        *,
        granularity: Granularity,
        count: int | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ) -> list[Candle]:
        return self._c[-count:] if count else self._c


def _candidate() -> StrategyCandidate:
    return StrategyCandidate(
        candidate_id="mac-jpy",
        run_id="run-1",
        archetype=StrategyArchetype.MA_CROSSOVER,
        instrument="USDJPY",
        timeframe=Granularity.H1,
        parameters=(
            StrategyParam(name="fast_period", value=Decimal("5")),
            StrategyParam(name="slow_period", value=Decimal("15")),
            StrategyParam(name="size", value=Decimal("10000")),
            StrategyParam(name="stop_distance_pips", value=Decimal("80")),
        ),
        parameter_ranges=(),
        rationale="test",
    )


def _config() -> BacktestRunConfig:
    return BacktestRunConfig(lookback_count=120, oos_fraction=0.2, min_trade_count=1)


def test_run_candidate_artifacts_in_sample_and_oos_with_curves() -> None:
    arts = run_candidate_artifacts(
        _candidate(), candle_provider=_Provider(_candles("USDJPY", "150.00")), config=_config()
    )
    segments = [a.segment for a in arts]
    assert EvidenceSegment.IN_SAMPLE in segments
    assert EvidenceSegment.OUT_OF_SAMPLE in segments
    for a in arts:
        assert len(a.equity_curve) > 0
        # net_pnl is exactly ending_equity - starting_balance (captured, verbatim)
        assert a.net_pnl == a.ending_equity - a.starting_balance
        # downsample keeps the curve bounded
        assert len(a.equity_curve) <= 400


def test_run_candidate_artifacts_is_deterministic() -> None:
    p = _config()
    a1 = run_candidate_artifacts(
        _candidate(), candle_provider=_Provider(_candles("USDJPY", "150.00")), config=p
    )
    a2 = run_candidate_artifacts(
        _candidate(), candle_provider=_Provider(_candles("USDJPY", "150.00")), config=p
    )
    assert [x.config_hash for x in a1] == [x.config_hash for x in a2]
    assert [x.net_pnl for x in a1] == [x.net_pnl for x in a2]


def test_artifact_store_round_trip(tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    arts = run_candidate_artifacts(
        _candidate(), candle_provider=_Provider(_candles("USDJPY", "150.00")), config=_config()
    )
    store = BacktestArtifactStore(tmp_path / "backtests")
    for a in arts:
        store.save(a)
    first = arts[0]
    loaded = store.get(first.config_hash)
    assert loaded is not None
    assert loaded.config_hash == first.config_hash
    assert loaded.equity_curve == first.equity_curve
    assert store.get("nope") is None
    assert len(store.all_artifacts()) == len(arts)
