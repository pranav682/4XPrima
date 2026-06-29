"""Tests for critic_agent + the OOS / robustness harness — fully hermetic.

The harness runs the REAL Stage-2 engine (including a REAL token-gated OOS run);
the critic's LLM is mocked. Coverage:

- OOS harness: a candidate is run on the sealed slice ONLY through the
  token-gated path; the LLM layer never receives the token, only the evidence.
- robustness: cost / parameter / trade-concentration evidence computed correctly.
- a candidate that collapses out-of-sample → the critic kills it (e2e).
- Tier-1 catches: an altered in-sample metric, a fabricated OOS claim, an
  altered OOS metric, an "approve"-like verdict, an execution field/intent, and
  a non-real candidate.
- the critic cannot even REPRESENT approve / deploy / trade.
- token-budget regression; e2e via AgentRunner.
- optional live smoke gated on RUN_LIVE_TESTS=1 + OPENAI_API_KEY.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from core.agents.backtest_harness import BacktestRunConfig, run_robustness
from core.agents.critic_agent import (
    _STABLE_SYSTEM,
    CriticAgent,
    CriticRequest,
)
from core.agents.evaluation import EvaluationGate
from core.agents.runner import AgentRunner
from core.agents.types import AgentBudget, AgentCall
from core.llm_client import AgentResponse, LlmFatalError, ModelTier, TokenUsage
from core.models import (
    BacktestEvidence,
    BacktestMetricsView,
    Candle,
    ChecklistItem,
    CriticVerdict,
    CriticVerdictKind,
    CriticVerdictSet,
    EvidenceSegment,
    Granularity,
    OverfittingConcern,
    ParamRange,
    RobustnessEvidence,
    StrategyArchetype,
    StrategyCandidate,
    StrategyParam,
)

BASE = datetime(2024, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Real-harness builders (genuine engine + token-gated OOS)
# ---------------------------------------------------------------------------


def _candles(n: int = 500) -> list[Candle]:
    out: list[Candle] = []
    px = Decimal("150.00")
    for i in range(n):
        px = px + (Decimal("0.25") if (i // 9) % 2 == 0 else Decimal("-0.25"))
        out.append(
            Candle(
                pair="USDJPY",
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
    def __init__(self, candles: list[Candle]) -> None:
        self._candles = candles

    def get_candles(
        self,
        pair: str,
        *,
        granularity: Granularity,
        count: int | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ) -> list[Candle]:
        return self._candles[-count:] if count else self._candles


def _candidate(candidate_id: str = "mac-1") -> StrategyCandidate:
    def P(n: str, v: str) -> StrategyParam:
        return StrategyParam(name=n, value=Decimal(v))

    def R(n: str, lo: str, hi: str) -> ParamRange:
        return ParamRange(name=n, low=Decimal(lo), high=Decimal(hi))

    return StrategyCandidate(
        candidate_id=candidate_id,
        run_id="c1",
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
        rationale="trend test",
    )


def _real_robustness() -> RobustnessEvidence:
    return run_robustness(
        _candidate(),
        candle_provider=StubCandleProvider(_candles()),
        config=BacktestRunConfig(lookback_count=500, oos_fraction=0.2, min_trade_count=5),
    )


# ---------------------------------------------------------------------------
# Constructed builders (clean, hand-built metric splits)
# ---------------------------------------------------------------------------


def _metrics(
    total: str = "0.20",
    sharpe: float = 1.5,
    pf: float | None = 1.8,
    dd: str = "0.10",
    trades: int = 50,
) -> BacktestMetricsView:
    return BacktestMetricsView(
        total_return_pct=Decimal(total),
        annualised_return_pct=Decimal("0.30"),
        sharpe_ratio=sharpe,
        sortino_ratio=2.0,
        max_drawdown_pct=Decimal(dd),
        win_rate=0.6,
        profit_factor=pf,
        trade_count=trades,
        avg_trade_pnl=Decimal("2.0"),
        exposure_pct=0.3,
    )


def _evidence(
    cid: str, segment: EvidenceSegment, metrics: BacktestMetricsView, chash: str
) -> BacktestEvidence:
    return BacktestEvidence(
        candidate_id=cid,
        config_hash=chash,
        pair="USDJPY",
        segment=segment,
        window_start=BASE,
        window_end=BASE + timedelta(hours=100),
        bars_total=100,
        bars_processed=100,
        halted_due_to_kill_switch=False,
        halt_reason=None,
        n_signals_proposed=50,
        n_signals_accepted=50,
        n_signals_rejected=0,
        starting_balance=Decimal("100000"),
        ending_equity=Decimal("120000"),
        cost_total=Decimal("10"),
        metrics=metrics,
        gates=(),
        gates_all_passed=True,
    )


def _robustness(
    cid: str = "mac-1",
    *,
    is_metrics: BacktestMetricsView | None = None,
    oos_metrics: BacktestMetricsView | None = None,
    include_oos: bool = True,
) -> RobustnessEvidence:
    is_ev = _evidence(cid, EvidenceSegment.IN_SAMPLE, is_metrics or _metrics(), "ishash")
    oos_ev = None
    if include_oos:
        oos_default = _metrics(total="-0.15", sharpe=-2.0, pf=0.5)  # collapse
        oos_ev = _evidence(
            cid, EvidenceSegment.OUT_OF_SAMPLE, oos_metrics or oos_default, "ooshash"
        )
    return RobustnessEvidence(candidate_id=cid, in_sample=is_ev, out_of_sample=oos_ev)


def _verbatim_verdict_set(
    run_id: str,
    robustness: tuple[RobustnessEvidence, ...],
    kind: CriticVerdictKind = CriticVerdictKind.KILL,
) -> CriticVerdictSet:
    return CriticVerdictSet(
        run_id=run_id,
        verdicts=tuple(
            CriticVerdict(
                candidate_id=rb.candidate_id,
                in_sample_config_hash=rb.in_sample.config_hash,
                oos_config_hash=rb.out_of_sample.config_hash if rb.out_of_sample else None,
                in_sample_metrics=rb.in_sample.metrics,
                out_of_sample_metrics=rb.out_of_sample.metrics if rb.out_of_sample else None,
                verdict=kind,
                concerns=(
                    OverfittingConcern(
                        item=ChecklistItem.OUT_OF_SAMPLE_DECAY,
                        finding="Out-of-sample Sharpe collapses vs in-sample.",
                    ),
                ),
                assessment="Out-of-sample performance collapses vs in-sample; classic curve-fit.",
                caveats="survive_for_now would not have meant validated.",
            )
            for rb in robustness
        ),
    )


def _snapshot(run_id: str, robustness: tuple[RobustnessEvidence, ...]) -> dict[str, Any]:
    return (
        CriticAgent()
        .prepare_call(CriticRequest(run_id=run_id, robustness=robustness))
        .input_snapshot
    )


def _checks() -> dict[str, Any]:
    agent = CriticAgent.__new__(CriticAgent)
    return {c.name: c.check for c in agent.evaluations()}


# ---------------------------------------------------------------------------
# OOS harness is real + token-gated; the token never reaches the LLM
# ---------------------------------------------------------------------------


def test_oos_run_is_real_token_gated_and_deterministic() -> None:
    rb = _real_robustness()
    assert rb.out_of_sample is not None
    assert rb.out_of_sample.segment == EvidenceSegment.OUT_OF_SAMPLE
    assert rb.in_sample.segment == EvidenceSegment.IN_SAMPLE
    # Different windows → different config_hash and different metrics.
    assert rb.in_sample.config_hash != rb.out_of_sample.config_hash
    assert rb.in_sample.metrics != rb.out_of_sample.metrics
    assert rb == _real_robustness()  # deterministic


def test_oos_token_never_reaches_the_llm_layer() -> None:
    call = CriticAgent().prepare_call(CriticRequest(run_id="c1", robustness=(_real_robustness(),)))
    blob = call.volatile_user + json.dumps(call.input_snapshot, default=str)
    assert "I_AM_DONE_TUNING" not in blob
    assert "access_out_of_sample" not in blob
    # But the OOS EVIDENCE did reach the LLM (it's the point).
    assert "out_of_sample" in json.dumps(call.input_snapshot, default=str)


def test_critic_source_never_references_oos_token() -> None:
    src = (Path(__file__).resolve().parents[1] / "core" / "agents" / "critic_agent.py").read_text()
    assert "I_AM_DONE_TUNING" not in src
    assert "access_out_of_sample" not in src


def test_robustness_evidence_components_are_computed() -> None:
    rb = _real_robustness()
    # cost-sensitivity at the configured multipliers
    assert tuple(c.cost_multiplier for c in rb.cost_stress) == (Decimal("1.5"), Decimal("2.0"))
    # parameter sensitivity: 4 params x 2 endpoints, with at least one fragile/
    # non-constructible neighbour (fast=20 >= slow=15 base).
    assert len(rb.param_sensitivity) == 8
    assert any(not n.constructible for n in rb.param_sensitivity)
    # trade-concentration shares are in [0, 1]
    tc = rb.trade_concentration
    assert tc is not None
    assert 0.0 <= tc.top_trade_profit_share <= 1.0
    assert 0.0 <= tc.top5_profit_share <= 1.0


# ---------------------------------------------------------------------------
# The critic can only kill or survive_for_now — never approve
# ---------------------------------------------------------------------------


def test_critic_cannot_represent_approval() -> None:
    values = {k.value for k in CriticVerdictKind}
    assert values == {"kill", "survive_for_now"}
    for forbidden in ("approve", "accept", "deploy", "trade", "advance"):
        assert forbidden not in values
    with pytest.raises(ValueError):
        CriticVerdictKind("approve")


def test_verbatim_verdict_passes_every_tier1_check() -> None:
    robustness = (_robustness(),)
    snap = _snapshot("c1", robustness)
    verdicts = _verbatim_verdict_set("c1", robustness)
    for name, fn in _checks().items():
        res = fn(verdicts, snap)
        assert res.passed, f"{name}: {res.reason}"


def test_kill_on_oos_collapse_end_to_end() -> None:
    # In-sample looks great; out-of-sample collapses → the canonical kill.
    robustness = (
        _robustness(
            is_metrics=_metrics(total="0.35", sharpe=2.4, pf=2.6),
            oos_metrics=_metrics(total="-0.22", sharpe=-1.9, pf=0.4),
        ),
    )
    canned = _verbatim_verdict_set("c1", robustness, kind=CriticVerdictKind.KILL)
    llm = MagicMock()
    llm.generate_structured.return_value = (
        canned,
        AgentResponse(
            agent_name="critic_agent",
            run_id="c1:attempt-0",
            tier=ModelTier.HEAVY,
            model="gpt-5.5",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=4000, cached_tokens=2500, completion_tokens=500),
            extra_metadata={"attempts": "1"},
        ),
    )
    runner = AgentRunner(llm_provider=llm, evaluation_gate=EvaluationGate(), budget=AgentBudget())
    result = runner.run(
        CriticAgent(), CriticRequest(run_id="c1", robustness=robustness), run_id="c1"
    )
    assert result.succeeded
    assert result.eval_verdict is not None and result.eval_verdict.tier1_passed
    out = result.output
    assert isinstance(out, CriticVerdictSet)
    assert out.verdicts[0].verdict == CriticVerdictKind.KILL
    assert result.metrics.tier == ModelTier.HEAVY


# ---------------------------------------------------------------------------
# Tier-1 integrity catches
# ---------------------------------------------------------------------------


def test_tier1_catches_altered_in_sample_metric() -> None:
    robustness = (_robustness(),)
    snap = _snapshot("c1", robustness)
    good = _verbatim_verdict_set("c1", robustness)
    tampered = good.verdicts[0].in_sample_metrics.model_copy(update={"sharpe_ratio": 9.9})
    bad = good.model_copy(
        update={"verdicts": (good.verdicts[0].model_copy(update={"in_sample_metrics": tampered}),)}
    )
    res = _checks()["in_sample_metrics_verbatim"](bad, snap)
    assert not res.passed
    assert "sharpe_ratio" in res.reason


def test_tier1_catches_fabricated_oos_claim() -> None:
    # Robustness WITHOUT an OOS run, but the verdict claims OOS metrics.
    robustness = (_robustness(include_oos=False),)
    snap = _snapshot("c1", robustness)
    fabricated = CriticVerdict(
        candidate_id="mac-1",
        in_sample_config_hash="ishash",
        oos_config_hash="made-up",
        in_sample_metrics=robustness[0].in_sample.metrics,
        out_of_sample_metrics=_metrics(total="0.5", sharpe=3.0),
        verdict=CriticVerdictKind.SURVIVE_FOR_NOW,
        concerns=(),
        assessment="Held up out-of-sample.",
    )
    bad = CriticVerdictSet(run_id="c1", verdicts=(fabricated,))
    res = _checks()["oos_metrics_verbatim_or_absent"](bad, snap)
    assert not res.passed
    assert "never produced" in res.reason


def test_tier1_catches_altered_oos_metric() -> None:
    robustness = (_robustness(),)  # has OOS
    snap = _snapshot("c1", robustness)
    good = _verbatim_verdict_set("c1", robustness)
    tampered = good.verdicts[0].out_of_sample_metrics.model_copy(
        update={"total_return_pct": Decimal("0.99")}
    )
    bad = good.model_copy(
        update={
            "verdicts": (good.verdicts[0].model_copy(update={"out_of_sample_metrics": tampered}),)
        }
    )
    res = _checks()["oos_metrics_verbatim_or_absent"](bad, snap)
    assert not res.passed
    assert "total_return_pct" in res.reason


def test_tier1_catches_approve_like_verdict_via_model_construct() -> None:
    robustness = (_robustness(),)
    snap = _snapshot("c1", robustness)
    good = _verbatim_verdict_set("c1", robustness)
    bad_verdict = good.verdicts[0].model_construct(
        **{**good.verdicts[0].__dict__, "verdict": "approve"}
    )
    bad = good.model_copy(update={"verdicts": (bad_verdict,)})
    res = _checks()["verdict_kind_valid"](bad, snap)
    assert not res.passed
    assert "approve" in res.reason


def test_model_rejects_execution_language_in_assessment() -> None:
    with pytest.raises(ValidationError):
        CriticVerdict(
            candidate_id="c",
            in_sample_config_hash="h",
            in_sample_metrics=_metrics(),
            verdict=CriticVerdictKind.SURVIVE_FOR_NOW,
            assessment="Survives — deploy this live.",
        )


def test_model_rejects_planted_execution_field() -> None:
    with pytest.raises(ValidationError):
        CriticVerdict(
            candidate_id="c",
            in_sample_config_hash="h",
            in_sample_metrics=_metrics(),
            verdict=CriticVerdictKind.KILL,
            assessment="dead",
            deploy_live=True,  # type: ignore[call-arg]
        )


def test_tier1_catches_execution_intent_via_model_construct() -> None:
    robustness = (_robustness(),)
    snap = _snapshot("c1", robustness)
    good = _verbatim_verdict_set("c1", robustness)
    bad_verdict = good.verdicts[0].model_construct(
        **{**good.verdicts[0].__dict__, "assessment": "Strong — go live with real money."}
    )
    bad = good.model_copy(update={"verdicts": (bad_verdict,)})
    res = _checks()["no_execution_intent"](bad, snap)
    assert not res.passed


def test_tier1_catches_reference_to_non_run_candidate() -> None:
    robustness = (_robustness(),)
    snap = _snapshot("c1", robustness)
    good = _verbatim_verdict_set("c1", robustness)
    bad_verdict = good.verdicts[0].model_copy(update={"candidate_id": "ghost"})
    bad = good.model_copy(update={"verdicts": (bad_verdict,)})
    res = _checks()["references_real_candidates"](bad, snap)
    assert not res.passed
    assert "ghost" in res.reason


# ---------------------------------------------------------------------------
# Token budget + fatal path
# ---------------------------------------------------------------------------


def test_stable_prefix_meets_cache_threshold() -> None:
    assert len(_STABLE_SYSTEM) >= 4096
    assert len(_STABLE_SYSTEM) <= 25_000 * 4


def test_prepare_call_uses_heavy_tier() -> None:
    call = CriticAgent().prepare_call(CriticRequest(run_id="c1", robustness=(_robustness(),)))
    assert isinstance(call, AgentCall)
    assert call.tier == ModelTier.HEAVY
    assert call.output_model is CriticVerdictSet
    assert "ADVERSARIAL" in call.stable_system


def test_runner_surfaces_fatal_llm_error() -> None:
    llm = MagicMock()
    llm.generate_structured.side_effect = LlmFatalError("schema validation failed")
    runner = AgentRunner(llm_provider=llm, evaluation_gate=EvaluationGate(), budget=AgentBudget())
    result = runner.run(
        CriticAgent(), CriticRequest(run_id="c1", robustness=(_robustness(),)), run_id="c1"
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
def test_live_smoke_one_real_heavy_critique() -> None:
    from core.config import OpenAISettings
    from core.llm_client import OpenAIProvider

    robustness = (_real_robustness(),)
    provider = OpenAIProvider(OpenAISettings())  # type: ignore[call-arg]
    runner = AgentRunner(
        llm_provider=provider, evaluation_gate=EvaluationGate(), budget=AgentBudget()
    )
    result = runner.run(
        CriticAgent(), CriticRequest(run_id="c-live-1", robustness=robustness), run_id="c-live-1"
    )
    assert result.succeeded, result.failure
    out = result.output
    assert isinstance(out, CriticVerdictSet)
    # Integrity + no-approve held on a real HEAVY-tier call.
    assert result.eval_verdict is not None and result.eval_verdict.tier1_passed
    assert all(
        v.verdict in (CriticVerdictKind.KILL, CriticVerdictKind.SURVIVE_FOR_NOW)
        for v in out.verdicts
    )
