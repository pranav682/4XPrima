"""Tests for the deterministic orchestrator — fully hermetic.

The four worker agents + AgentRunner are mocked (no real LLM); the deterministic
harness runs the REAL Stage-2 engine on stub candles. Coverage:

- run_cycle sequences the agents in order, passing each typed output to the next.
- per-cycle budget aborts BEFORE the next agent call and records partial state.
- registry keyed by strategy identity; killed candidates not queued, survivors are.
- SAFETY WALLS: no promote/set-live/mutate-config path; risk states not writable.
- fail-closed: a worker failure halts the cycle, promotes nothing, leaves state clean.
- approval queue entries are pending; nothing auto-approved.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.agents.backtest_harness import BacktestRunConfig
from core.agents.types import AgentMetrics, AgentRunFailure, AgentRunResult, EvalVerdict
from core.llm_client import ModelTier
from core.models import (
    BacktestMetricsView,
    BacktestTriage,
    BacktestVerdict,
    BacktestVerdictSet,
    Candle,
    CriticVerdict,
    CriticVerdictKind,
    CriticVerdictSet,
    Granularity,
    MarketContextReport,
    ParamRange,
    StrategyArchetype,
    StrategyCandidate,
    StrategyParam,
    StrategyProposal,
)
from core.orchestration import (
    ApprovalQueue,
    ChampionChallengerRegistry,
    CycleOutcome,
    Orchestrator,
    OrchestratorConfig,
    RegistryState,
    strategy_identity,
)

BASE = datetime(2024, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Candles / provider
# ---------------------------------------------------------------------------


def _candles(pair: str, start: str, n: int = 80) -> list[Candle]:
    out: list[Candle] = []
    px = Decimal(start)
    step = Decimal("0.20") if pair.endswith("JPY") else Decimal("0.0020")
    for i in range(n):
        px = px + (step if (i // 7) % 2 == 0 else -step)
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


class StubCandleProvider:
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
        canon = pair.replace("/", "").replace("_", "").upper()
        c = self._d.get(canon, [])
        return c[-count:] if count else c


def _provider() -> StubCandleProvider:
    return StubCandleProvider(
        {"USDJPY": _candles("USDJPY", "150.00"), "EURUSD": _candles("EURUSD", "1.1000")}
    )


# ---------------------------------------------------------------------------
# Candidate / agent-output builders
# ---------------------------------------------------------------------------


def _candidate(cid: str, instrument: str, stop: str) -> StrategyCandidate:
    def P(n: str, v: str) -> StrategyParam:
        return StrategyParam(name=n, value=Decimal(v))

    def R(n: str, lo: str, hi: str) -> ParamRange:
        return ParamRange(name=n, low=Decimal(lo), high=Decimal(hi))

    return StrategyCandidate(
        candidate_id=cid,
        run_id="cyc-1",
        archetype=StrategyArchetype.MA_CROSSOVER,
        instrument=instrument,
        timeframe=Granularity.H1,
        parameters=(
            P("fast_period", "5"),
            P("slow_period", "15"),
            P("size", "1000"),
            P("stop_distance_pips", stop),
        ),
        parameter_ranges=(
            R("fast_period", "3", "10"),
            R("slow_period", "12", "25"),
            R("size", "500", "2000"),
            R("stop_distance_pips", "40", "120"),
        ),
        rationale="test candidate",
    )


def _proposal(candidates: list[StrategyCandidate]) -> StrategyProposal:
    return StrategyProposal(run_id="cyc-1", as_of=BASE, candidates=tuple(candidates))


def _metrics_view() -> BacktestMetricsView:
    return BacktestMetricsView(
        total_return_pct=Decimal("0.05"),
        annualised_return_pct=Decimal("0.10"),
        sharpe_ratio=0.5,
        sortino_ratio=0.7,
        max_drawdown_pct=Decimal("0.08"),
        win_rate=0.55,
        profit_factor=1.4,
        trade_count=40,
        avg_trade_pnl=Decimal("1.2"),
        exposure_pct=0.3,
    )


def _bt_set(triage_by_id: dict[str, BacktestTriage]) -> BacktestVerdictSet:
    return BacktestVerdictSet(
        run_id="cyc-1",
        verdicts=tuple(
            BacktestVerdict(
                candidate_id=cid,
                config_hash="h",
                metrics=_metrics_view(),
                gates=(),
                assessment="in-sample read",
                triage=triage,
                caveats="in-sample only",
            )
            for cid, triage in triage_by_id.items()
        ),
    )


def _critic_set(verdict_by_id: dict[str, CriticVerdictKind]) -> CriticVerdictSet:
    return CriticVerdictSet(
        run_id="cyc-1",
        verdicts=tuple(
            CriticVerdict(
                candidate_id=cid,
                in_sample_config_hash="h",
                in_sample_metrics=_metrics_view(),
                verdict=kind,
                assessment="adversarial read",
                caveats="not validation",
            )
            for cid, kind in verdict_by_id.items()
        ),
    )


def _metrics(cost: str = "0.02") -> AgentMetrics:
    return AgentMetrics(
        agent_name="x",
        run_id="r",
        tier=ModelTier.DEFAULT,
        model="m",
        prompt_tokens=100,
        cached_tokens=0,
        completion_tokens=50,
        estimated_cost_usd=Decimal(cost),
        cache_hit_ratio=0.0,
        latency_seconds=0.01,
        attempts=1,
    )


def _ok(output: Any, cost: str = "0.02") -> AgentRunResult:
    return AgentRunResult(
        output=output,
        failure=None,
        metrics=_metrics(cost),
        eval_verdict=EvalVerdict(tier1_passed=True),
    )


def _fail(reason: str = "boom", cost: str = "0.02") -> AgentRunResult:
    return AgentRunResult(
        output=None,
        failure=AgentRunFailure(code="fatal_llm_error", reason=reason),
        metrics=_metrics(cost),
        eval_verdict=None,
    )


def _config(max_usd: str = "5.00", per_call: str = "1.00") -> OrchestratorConfig:
    return OrchestratorConfig(
        max_cost_per_cycle_usd=Decimal(max_usd),
        per_call_cost_ceiling_usd=Decimal(per_call),
        n_candidates=2,
        backtest_config=BacktestRunConfig(lookback_count=80, oos_fraction=0.2, min_trade_count=1),
    )


def _make_orch(
    tmp_path: Any, runner: MagicMock, config: OrchestratorConfig | None = None
) -> tuple[Orchestrator, ChampionChallengerRegistry, ApprovalQueue]:
    registry = ChampionChallengerRegistry(tmp_path / "registry.json")
    queue = ApprovalQueue(tmp_path / "queue.json")
    orch = Orchestrator(
        market_context_agent=MagicMock(name="ctx_agent"),
        strategy_lab_agent=MagicMock(name="lab_agent"),
        backtest_agent=MagicMock(name="bt_agent"),
        critic_agent=MagicMock(name="critic_agent"),
        runner=runner,
        candle_provider=_provider(),
        registry=registry,
        approval_queue=queue,
        config=config or _config(),
    )
    return orch, registry, queue


# ---------------------------------------------------------------------------
# Sequencing
# ---------------------------------------------------------------------------


def test_run_cycle_sequences_agents_and_passes_typed_outputs(tmp_path: Any) -> None:
    a = _candidate("mac-jpy", "USDJPY", "80")
    b = _candidate("mac-eur", "EURUSD", "50")
    ctx = MarketContextReport(run_id="cyc-1", as_of=BASE)
    proposal = _proposal([a, b])
    runner = MagicMock()
    runner.run.side_effect = [
        _ok(ctx),
        _ok(proposal),
        _ok(
            _bt_set(
                {
                    "mac-jpy": BacktestTriage.ADVANCE_TO_CRITIC,
                    "mac-eur": BacktestTriage.ADVANCE_TO_CRITIC,
                }
            )
        ),
        _ok(_critic_set({"mac-jpy": CriticVerdictKind.KILL, "mac-eur": CriticVerdictKind.KILL})),
    ]
    orch, _registry, _queue = _make_orch(tmp_path, runner)
    result = orch.run_cycle(("USDJPY", "EURUSD"), cycle_id="cyc-1")

    assert result.outcome == CycleOutcome.COMPLETED
    assert runner.run.call_count == 4
    agents = [c.args[0]._mock_name for c in runner.run.call_args_list]
    assert agents == ["ctx_agent", "lab_agent", "bt_agent", "critic_agent"]
    # Typed output threaded into the next agent's request.
    lab_req = runner.run.call_args_list[1].args[1]
    assert lab_req.market_context is ctx
    bt_req = runner.run.call_args_list[2].args[1]
    assert bt_req.proposal is proposal
    crit_req = runner.run.call_args_list[3].args[1]
    assert {rb.candidate_id for rb in crit_req.robustness} == {"mac-jpy", "mac-eur"}


def test_no_candidates_completes_without_backtest_or_critic(tmp_path: Any) -> None:
    runner = MagicMock()
    runner.run.side_effect = [
        _ok(MarketContextReport(run_id="cyc-1", as_of=BASE)),
        _ok(_proposal([])),
    ]
    orch, _r, queue = _make_orch(tmp_path, runner)
    result = orch.run_cycle(("USDJPY",), cycle_id="cyc-1")
    assert result.outcome == CycleOutcome.COMPLETED
    assert result.candidates_proposed == 0
    assert runner.run.call_count == 2  # ctx + lab only
    assert queue.pending() == ()


# ---------------------------------------------------------------------------
# Budget: abort before the next agent call
# ---------------------------------------------------------------------------


def test_budget_aborts_before_next_agent_call(tmp_path: Any) -> None:
    a = _candidate("mac-jpy", "USDJPY", "80")
    b = _candidate("mac-eur", "EURUSD", "50")
    runner = MagicMock()
    # Each call costs 0.02; cap 0.07, per-call ceiling 0.02. After ctx+lab+bt
    # (running 0.06) the pre-check before the critic (0.06+0.02 > 0.07) aborts.
    runner.run.side_effect = [
        _ok(MarketContextReport(run_id="cyc-1", as_of=BASE), "0.02"),
        _ok(_proposal([a, b]), "0.02"),
        _ok(
            _bt_set(
                {
                    "mac-jpy": BacktestTriage.ADVANCE_TO_CRITIC,
                    "mac-eur": BacktestTriage.ADVANCE_TO_CRITIC,
                }
            ),
            "0.02",
        ),
        _ok(
            _critic_set({"mac-jpy": CriticVerdictKind.SURVIVE_FOR_NOW}), "0.02"
        ),  # must NOT be used
    ]
    orch, registry, queue = _make_orch(tmp_path, runner, _config(max_usd="0.07", per_call="0.02"))
    result = orch.run_cycle(("USDJPY", "EURUSD"), cycle_id="cyc-1")

    assert result.outcome == CycleOutcome.ABORTED_BUDGET
    assert runner.run.call_count == 3  # critic was NOT called
    assert "critic" in (result.abort_reason or "")
    # Partial state: candidates backtested, nothing queued / promoted.
    assert queue.pending() == ()
    assert all(e.state != RegistryState.QUEUED_FOR_APPROVAL for e in registry.all_entries())


# ---------------------------------------------------------------------------
# Registry identity + routing
# ---------------------------------------------------------------------------


def test_strategy_identity_is_stable_and_dedupes(tmp_path: Any) -> None:
    a1 = _candidate("id-A", "USDJPY", "80")
    a2 = _candidate("id-B", "USDJPY", "80")  # same spec, different candidate_id
    assert strategy_identity(a1) == strategy_identity(a2)
    different = _candidate("id-C", "USDJPY", "90")  # different stop
    assert strategy_identity(a1) != strategy_identity(different)

    registry = ChampionChallengerRegistry(tmp_path / "r.json")
    registry.upsert_proposed(a1, run_id="c1")
    registry.upsert_proposed(a2, run_id="c2")  # same identity → one entry
    assert len(registry.all_entries()) == 1


def test_killed_not_queued_survivor_queued_end_to_end(tmp_path: Any) -> None:
    a = _candidate("mac-jpy", "USDJPY", "80")  # advances + survives → queued
    b = _candidate("mac-eur", "EURUSD", "50")  # backtest-rejected → killed
    runner = MagicMock()
    runner.run.side_effect = [
        _ok(MarketContextReport(run_id="cyc-1", as_of=BASE)),
        _ok(_proposal([a, b])),
        _ok(
            _bt_set({"mac-jpy": BacktestTriage.ADVANCE_TO_CRITIC, "mac-eur": BacktestTriage.REJECT})
        ),
        _ok(_critic_set({"mac-jpy": CriticVerdictKind.SURVIVE_FOR_NOW})),
    ]
    orch, registry, queue = _make_orch(tmp_path, runner)
    result = orch.run_cycle(("USDJPY", "EURUSD"), cycle_id="cyc-1")

    assert result.outcome == CycleOutcome.COMPLETED
    assert result.candidates_proposed == 2
    assert result.candidates_killed == 1
    assert result.candidates_queued == 1

    by_id = {strategy_identity(a): a, strategy_identity(b): b}
    states = {e.identity: e.state for e in registry.all_entries()}
    assert states[strategy_identity(a)] == RegistryState.QUEUED_FOR_APPROVAL
    assert states[strategy_identity(b)] == RegistryState.KILLED
    pending = queue.pending()
    assert len(pending) == 1
    assert pending[0].identity == strategy_identity(a)
    assert pending[0].candidate.candidate_id == "mac-jpy"
    assert pending[0].status == "pending"
    assert pending[0].out_of_sample_evidence is not None  # critic opened OOS
    assert by_id  # silence unused


def test_critic_kill_records_killed_not_queued(tmp_path: Any) -> None:
    a = _candidate("mac-jpy", "USDJPY", "80")
    runner = MagicMock()
    runner.run.side_effect = [
        _ok(MarketContextReport(run_id="cyc-1", as_of=BASE)),
        _ok(_proposal([a])),
        _ok(_bt_set({"mac-jpy": BacktestTriage.ADVANCE_TO_CRITIC})),
        _ok(_critic_set({"mac-jpy": CriticVerdictKind.KILL})),
    ]
    orch, registry, queue = _make_orch(tmp_path, runner)
    result = orch.run_cycle(("USDJPY",), cycle_id="cyc-1")
    assert result.candidates_killed == 1
    assert result.candidates_queued == 0
    assert queue.pending() == ()
    assert registry.get(strategy_identity(a)).state == RegistryState.KILLED  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Fail-closed
# ---------------------------------------------------------------------------


def test_worker_failure_halts_cycle_promotes_nothing(tmp_path: Any) -> None:
    a = _candidate("mac-jpy", "USDJPY", "80")
    runner = MagicMock()
    runner.run.side_effect = [
        _ok(MarketContextReport(run_id="cyc-1", as_of=BASE)),
        _ok(_proposal([a])),
        _fail(reason="schema validation failed"),  # backtest_agent fails
    ]
    orch, registry, queue = _make_orch(tmp_path, runner)
    result = orch.run_cycle(("USDJPY",), cycle_id="cyc-1")
    assert result.outcome == CycleOutcome.ABORTED_FAILURE
    assert "backtest_agent" in (result.abort_reason or "")
    assert queue.pending() == ()  # queue uncorrupted, nothing queued
    # No candidate reached a risk-authorizing or queued state.
    for e in registry.all_entries():
        assert e.state in (RegistryState.PROPOSED, RegistryState.BACKTESTED)


# ---------------------------------------------------------------------------
# Safety walls
# ---------------------------------------------------------------------------


def test_orchestrator_exposes_no_promotion_or_config_mutation_path() -> None:
    public = {m for m in dir(Orchestrator) if not m.startswith("_")}
    assert public == {"run_cycle"}, public

    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "core" / "orchestration"
    src = (base / "orchestrator.py").read_text()
    # Code-pattern tokens (not prose) — the docstring legitimately *describes*
    # the walls ("it cannot promote…"), so we look for actual calls/symbols.
    for forbidden in (
        "RegistryState.CHAMPION",
        "RegistryState.APPROVED",
        "RegistryState.LIVE",
        ".trip(",
        ".reset(",
        "RiskManager",
        ".promote(",
        "promote_to_champion",
        "set_champion",
        "set_live(",
        "paper_live",
        "go_live",
    ):
        assert forbidden not in src, f"orchestrator references {forbidden!r}"


def test_registry_rejects_risk_authorizing_states_at_runtime(tmp_path: Any) -> None:
    registry = ChampionChallengerRegistry(tmp_path / "r.json")
    a = _candidate("x", "USDJPY", "80")
    ident = registry.upsert_proposed(a, run_id="c1")
    # The mypy wall is types; this is the belt-and-braces runtime guard.
    for state in (RegistryState.CHAMPION, RegistryState.APPROVED, RegistryState.LIVE):
        with pytest.raises(ValueError):
            registry.set_state(ident, state)  # type: ignore[arg-type]
    # And nobody got promoted.
    assert registry.current_champion() is None


def test_registry_has_no_champion_promotion_method() -> None:
    public = {m for m in dir(ChampionChallengerRegistry) if not m.startswith("_")}
    for forbidden in ("promote", "promote_to_champion", "set_champion", "approve", "set_live"):
        assert forbidden not in public


def test_approval_queue_has_no_approve_or_reject_method() -> None:
    public = {m for m in dir(ApprovalQueue) if not m.startswith("_")}
    assert "append" in public
    for forbidden in ("approve", "reject", "decide", "remove", "pop"):
        assert forbidden not in public
