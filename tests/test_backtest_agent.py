"""Tests for backtest_agent + its deterministic harness — fully hermetic.

The harness runs the REAL Stage-2 engine over fixture candles; the agent (whose
LLM is mocked) only interprets. Coverage:

- harness: a StrategyProposal -> build_strategy -> real engine -> BacktestEvidence
  with metrics that match a direct engine run (verbatim), and determinism.
- agent: interprets evidence into a valid BacktestVerdictSet.
- Tier-1 integrity catches: a FABRICATED metric, an OOS-result claim, an
  execution field + execution intent, an out-of-enum triage, and a reference to
  a non-run candidate.
- in-sample-only: neither the agent nor the harness touches the OOS token.
- end-to-end via AgentRunner; token-budget regression.
- optional live smoke gated on RUN_LIVE_TESTS=1 + OPENAI_API_KEY.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from core.agents.backtest_agent import (
    _STABLE_SYSTEM,
    BacktestAgent,
    BacktestAgentRequest,
)
from core.agents.backtest_harness import (
    BacktestRunConfig,
    metrics_view,
    run_candidate,
    run_proposal,
)
from core.agents.evaluation import EvaluationGate
from core.agents.runner import AgentRunner
from core.agents.types import AgentBudget, AgentCall
from core.backtest import BacktestEngine, DataSplit
from core.llm_client import AgentResponse, LlmFatalError, ModelTier, TokenUsage
from core.models import (
    BacktestEvidence,
    BacktestTriage,
    BacktestVerdict,
    BacktestVerdictSet,
    Candle,
    Granularity,
    ParamRange,
    StrategyArchetype,
    StrategyCandidate,
    StrategyParam,
    StrategyProposal,
)
from core.strategy import build_strategy

# NOTE: deliberately do NOT call structlog.configure() here — structlog config
# is global, so forcing a level would pollute other test modules (e.g. the
# runner's log-assertion tests). pytest captures the engine's stdout logs.

BASE = datetime(2024, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _candles(pair: str = "USDJPY", n: int = 400) -> list[Candle]:
    """Oscillating JPY-priced candles that produce crossovers for fast 5 / slow 15."""
    out: list[Candle] = []
    px = Decimal("150.00")
    for i in range(n):
        px = px + (Decimal("0.20") if (i // 7) % 2 == 0 else Decimal("-0.20"))
        out.append(
            Candle(
                pair=pair,
                granularity=Granularity.H1,
                time=BASE + timedelta(hours=i),
                open=px,
                high=px + Decimal("0.05"),
                low=px - Decimal("0.05"),
                close=px,
                volume=1000,
                complete=True,
            )
        )
    return out


class StubCandleProvider:
    def __init__(self, candles_by_pair: dict[str, list[Candle]]) -> None:
        self._d = candles_by_pair

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
        candles = self._d.get(canon, [])
        return candles[-count:] if count else candles


def _candidate(candidate_id: str = "mac-1", run_id: str = "bt-1") -> StrategyCandidate:
    def P(n: str, v: str) -> StrategyParam:
        return StrategyParam(name=n, value=Decimal(v))

    def R(n: str, lo: str, hi: str) -> ParamRange:
        return ParamRange(name=n, low=Decimal(lo), high=Decimal(hi))

    return StrategyCandidate(
        candidate_id=candidate_id,
        run_id=run_id,
        archetype=StrategyArchetype.MA_CROSSOVER,
        instrument="USDJPY",
        timeframe=Granularity.H1,
        parameters=(
            P("fast_period", "5"),
            P("slow_period", "15"),
            P("size", "1000"),
            P("stop_distance_pips", "80"),
        ),
        parameter_ranges=(
            R("fast_period", "3", "20"),
            R("slow_period", "10", "40"),
            R("size", "500", "2000"),
            R("stop_distance_pips", "40", "120"),
        ),
        rationale="USDJPY event-driven; a crossover tests post-event follow-through.",
    )


def _proposal(run_id: str = "bt-1") -> StrategyProposal:
    return StrategyProposal(run_id=run_id, as_of=BASE, candidates=(_candidate(run_id=run_id),))


def _config() -> BacktestRunConfig:
    return BacktestRunConfig(lookback_count=400, oos_fraction=0.2, min_trade_count=5)


def _provider() -> StubCandleProvider:
    return StubCandleProvider({"USDJPY": _candles()})


def _evidence(run_id: str = "bt-1") -> tuple[BacktestEvidence, ...]:
    ev, skipped = run_proposal(_proposal(run_id), candle_provider=_provider(), config=_config())
    assert not skipped, skipped
    return ev


def _verbatim_verdict_set(
    run_id: str, evidence: tuple[BacktestEvidence, ...]
) -> BacktestVerdictSet:
    """Build a verdict set that copies every metric + gate verbatim — i.e. what
    a well-behaved LLM should produce. Used as the mocked LLM output."""
    return BacktestVerdictSet(
        run_id=run_id,
        verdicts=tuple(
            BacktestVerdict(
                candidate_id=ev.candidate_id,
                config_hash=ev.config_hash,
                metrics=ev.metrics,
                gates=ev.gates,
                assessment=(
                    "In-sample read is weak: the run breached the drawdown ceiling "
                    "and the edge looks thin. High in-sample numbers prove nothing."
                ),
                concerns=("drawdown near/over ceiling", "few trades for the timeframe"),
                triage=BacktestTriage.REJECT,
                caveats="In-sample only; no out-of-sample validation has run yet.",
            )
            for ev in evidence
        ),
    )


def _agent_snapshot(run_id: str, evidence: tuple[BacktestEvidence, ...]) -> dict[str, Any]:
    call = BacktestAgent().prepare_call(
        BacktestAgentRequest(run_id=run_id, proposal=_proposal(run_id), evidence=evidence)
    )
    return call.input_snapshot


def _checks() -> dict[str, Any]:
    agent = BacktestAgent.__new__(BacktestAgent)
    return {c.name: c.check for c in agent.evaluations()}


# ---------------------------------------------------------------------------
# Harness: real engine, verbatim metrics, determinism
# ---------------------------------------------------------------------------


def test_harness_produces_evidence_matching_a_direct_engine_run() -> None:
    config = _config()
    candidate = _candidate()
    candles = _candles()
    evidence = run_candidate(candidate, candle_provider=_provider(), config=config)

    # Run the engine directly on the same in-sample slice and compare verbatim.
    in_sample = DataSplit(candles, oos_fraction=config.oos_fraction).in_sample
    direct = BacktestEngine(
        bars=list(in_sample),
        strategy=build_strategy(candidate),
        risk_config=config.risk_config,
        cost_model=config.cost_model,
        starting_balance=config.starting_balance,
    ).run()

    assert evidence.config_hash == direct.config_hash
    assert evidence.pair == "USDJPY"
    assert evidence.bars_total == len(in_sample)
    assert evidence.metrics == metrics_view(direct.metrics)
    # The harness ran the in-sample slice only (80% of 400).
    assert evidence.bars_total == 320


def test_harness_is_deterministic() -> None:
    ev1 = _evidence()
    ev2 = _evidence()
    assert ev1 == ev2
    assert ev1[0].config_hash == ev2[0].config_hash


def test_harness_gates_are_computed_and_flagged() -> None:
    evidence = _evidence()[0]
    gate_names = {g.name for g in evidence.gates}
    assert {
        "min_trade_count",
        "max_drawdown",
        "not_halted",
        "in_sample_profit_factor",
    } == gate_names
    assert isinstance(evidence.gates_all_passed, bool)


def test_harness_skips_candidate_with_no_data() -> None:
    proposal = StrategyProposal(
        run_id="bt-1",
        as_of=BASE,
        candidates=(_candidate(candidate_id="eur-1").model_copy(update={"instrument": "EURGBP"}),),
    )
    evidence, skipped = run_proposal(proposal, candle_provider=_provider(), config=_config())
    assert evidence == ()
    assert len(skipped) == 1
    assert skipped[0][0] == "eur-1"


# ---------------------------------------------------------------------------
# Agent: prepare_call + valid interpretation
# ---------------------------------------------------------------------------


def test_agent_name() -> None:
    assert BacktestAgent.name == "backtest_agent"


def test_prepare_call_shape() -> None:
    evidence = _evidence()
    call = BacktestAgent().prepare_call(
        BacktestAgentRequest(run_id="bt-1", proposal=_proposal(), evidence=evidence)
    )
    assert isinstance(call, AgentCall)
    assert call.tier == ModelTier.DEFAULT
    assert call.output_model is BacktestVerdictSet
    assert "backtest_agent" in call.stable_system
    assert "VERBATIM" in call.stable_system
    assert len(call.stable_system) >= 4096
    assert "EVIDENCE" in call.volatile_user
    assert call.input_snapshot["run_id"] == "bt-1"
    assert evidence[0].candidate_id in call.input_snapshot["evidence"]


def test_verbatim_verdict_passes_every_tier1_check() -> None:
    evidence = _evidence()
    snap = _agent_snapshot("bt-1", evidence)
    verdicts = _verbatim_verdict_set("bt-1", evidence)
    for name, fn in _checks().items():
        res = fn(verdicts, snap)
        assert res.passed, f"{name}: {res.reason}"


# ---------------------------------------------------------------------------
# Tier-1 integrity catches
# ---------------------------------------------------------------------------


def test_tier1_catches_fabricated_metric() -> None:
    evidence = _evidence()
    snap = _agent_snapshot("bt-1", evidence)
    good = _verbatim_verdict_set("bt-1", evidence)
    # Alter ONE metric — the core fabrication the integrity check exists for.
    tampered_metrics = good.verdicts[0].metrics.model_copy(
        update={"total_return_pct": Decimal("9.99")}
    )
    bad_verdict = good.verdicts[0].model_copy(update={"metrics": tampered_metrics})
    bad = good.model_copy(update={"verdicts": (bad_verdict,)})
    res = _checks()["metrics_verbatim"](bad, snap)
    assert not res.passed
    assert "total_return_pct" in res.reason


def test_tier1_catches_flipped_gate() -> None:
    evidence = _evidence()
    snap = _agent_snapshot("bt-1", evidence)
    good = _verbatim_verdict_set("bt-1", evidence)
    flipped = tuple(g.model_copy(update={"passed": not g.passed}) for g in good.verdicts[0].gates)
    bad_verdict = good.verdicts[0].model_copy(update={"gates": flipped})
    bad = good.model_copy(update={"verdicts": (bad_verdict,)})
    res = _checks()["gates_verbatim"](bad, snap)
    assert not res.passed


def test_tier1_catches_oos_result_claim() -> None:
    evidence = _evidence()
    snap = _agent_snapshot("bt-1", evidence)
    good = _verbatim_verdict_set("bt-1", evidence)
    # Plant a fabricated out-of-sample number in the prose.
    bad_verdict = good.verdicts[0].model_copy(
        update={"assessment": "Out-of-sample Sharpe was 2.1, so it validates well."}
    )
    bad = good.model_copy(update={"verdicts": (bad_verdict,)})
    res = _checks()["no_oos_results"](bad, snap)
    assert not res.passed
    assert "out-of-sample" in res.reason.lower()


def test_caveat_mentioning_oos_without_a_figure_is_allowed() -> None:
    evidence = _evidence()
    snap = _agent_snapshot("bt-1", evidence)
    good = _verbatim_verdict_set("bt-1", evidence)  # caveat already mentions OOS
    assert _checks()["no_oos_results"](good, snap).passed


def test_model_rejects_execution_language_in_assessment() -> None:
    with pytest.raises(ValidationError):
        BacktestVerdict(
            candidate_id="c",
            config_hash="h",
            metrics=_evidence()[0].metrics,
            gates=(),
            assessment="Looks good — deploy this live to production.",
            triage=BacktestTriage.ADVANCE_TO_CRITIC,
        )


def test_tier1_catches_execution_intent_via_model_construct() -> None:
    evidence = _evidence()
    snap = _agent_snapshot("bt-1", evidence)
    good = _verbatim_verdict_set("bt-1", evidence)
    bad_verdict = good.verdicts[0].model_construct(
        **{**good.verdicts[0].__dict__, "assessment": "Strong — go live with real money."}
    )
    bad = good.model_copy(update={"verdicts": (bad_verdict,)})
    res = _checks()["no_execution_intent"](bad, snap)
    assert not res.passed


def test_model_rejects_planted_oos_and_execution_fields() -> None:
    base = _verbatim_verdict_set("bt-1", _evidence()).verdicts[0]
    fields = {
        "candidate_id": base.candidate_id,
        "config_hash": base.config_hash,
        "metrics": base.metrics,
        "gates": base.gates,
        "assessment": "fine",
        "triage": BacktestTriage.REJECT,
    }
    with pytest.raises(ValidationError):
        BacktestVerdict(**fields, oos_sharpe=2.1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        BacktestVerdict(**fields, deploy_live=True)  # type: ignore[call-arg]


def test_tier1_catches_out_of_enum_triage() -> None:
    evidence = _evidence()
    snap = _agent_snapshot("bt-1", evidence)
    good = _verbatim_verdict_set("bt-1", evidence)
    bad_verdict = good.verdicts[0].model_construct(
        **{**good.verdicts[0].__dict__, "triage": "ship_it"}
    )
    bad = good.model_copy(update={"verdicts": (bad_verdict,)})
    res = _checks()["triage_valid"](bad, snap)
    assert not res.passed
    assert "ship_it" in res.reason


def test_tier1_catches_reference_to_non_run_candidate() -> None:
    evidence = _evidence()
    snap = _agent_snapshot("bt-1", evidence)
    good = _verbatim_verdict_set("bt-1", evidence)
    bad_verdict = good.verdicts[0].model_copy(update={"candidate_id": "never-ran"})
    bad = good.model_copy(update={"verdicts": (bad_verdict,)})
    res = _checks()["references_run_candidates"](bad, snap)
    assert not res.passed
    assert "never-ran" in res.reason


# ---------------------------------------------------------------------------
# In-sample only: the OOS holdout token is never touched
# ---------------------------------------------------------------------------


def test_backtest_agent_never_touches_oos_token() -> None:
    # backtest_agent is IN-SAMPLE only — it must never reference the OOS token
    # or accessor. (The harness DOES open the OOS slice, but only for the critic
    # stage, in deterministic code — see test_critic_agent.)
    src = (Path(__file__).resolve().parents[1] / "core" / "agents" / "backtest_agent.py").read_text()
    assert "I_AM_DONE_TUNING" not in src
    assert "access_out_of_sample" not in src


# ---------------------------------------------------------------------------
# End-to-end + token budget
# ---------------------------------------------------------------------------


def test_stable_prefix_meets_cache_threshold() -> None:
    assert len(_STABLE_SYSTEM) >= 4096
    assert len(_STABLE_SYSTEM) <= 25_000 * 4


def test_end_to_end_via_runner() -> None:
    evidence = _evidence()
    canned = _verbatim_verdict_set("bt-1", evidence)
    llm = MagicMock()
    llm.generate_structured.return_value = (
        canned,
        AgentResponse(
            agent_name="backtest_agent",
            run_id="bt-1:attempt-0",
            tier=ModelTier.DEFAULT,
            model="gpt-5.4",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=3000, cached_tokens=1800, completion_tokens=400),
            extra_metadata={"attempts": "1"},
        ),
    )
    runner = AgentRunner(llm_provider=llm, evaluation_gate=EvaluationGate(), budget=AgentBudget())
    result = runner.run(
        BacktestAgent(),
        BacktestAgentRequest(run_id="bt-1", proposal=_proposal(), evidence=evidence),
        run_id="bt-1",
    )
    assert result.succeeded
    assert result.output is canned
    assert result.eval_verdict is not None
    assert result.eval_verdict.tier1_passed


def test_runner_surfaces_fatal_llm_error() -> None:
    llm = MagicMock()
    llm.generate_structured.side_effect = LlmFatalError("schema validation failed")
    runner = AgentRunner(llm_provider=llm, evaluation_gate=EvaluationGate(), budget=AgentBudget())
    result = runner.run(
        BacktestAgent(),
        BacktestAgentRequest(run_id="bt-1", proposal=_proposal(), evidence=_evidence()),
        run_id="bt-1",
    )
    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code == "fatal_llm_error"


# ---------------------------------------------------------------------------
# Optional live smoke (skipped in CI)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1" or not os.environ.get("OPENAI_API_KEY"),
    reason="RUN_LIVE_TESTS=1 and OPENAI_API_KEY must be set",
)
def test_live_smoke_one_real_interpretation() -> None:
    from core.config import OpenAISettings
    from core.llm_client import OpenAIProvider

    evidence = _evidence("bt-live-1")
    provider = OpenAIProvider(OpenAISettings())  # type: ignore[call-arg]
    runner = AgentRunner(
        llm_provider=provider, evaluation_gate=EvaluationGate(), budget=AgentBudget()
    )
    result = runner.run(
        BacktestAgent(),
        BacktestAgentRequest(
            run_id="bt-live-1", proposal=_proposal("bt-live-1"), evidence=evidence
        ),
        run_id="bt-live-1",
    )
    assert result.succeeded, result.failure
    out = result.output
    assert isinstance(out, BacktestVerdictSet)
    # The metrics-verbatim + no-OOS Tier-1 checks held on a real call.
    assert result.eval_verdict is not None and result.eval_verdict.tier1_passed
