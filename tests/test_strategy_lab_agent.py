"""Tests for strategy_lab_agent — fully hermetic (mock llm_client, no network).

Coverage:
- prepare_call() shape: DEFAULT tier, StrategyProposal schema, cache-sized
  stable prefix carrying the archetype catalog; volatile carries the universe.
- a valid candidate parses AND is constructible into a real Strategy.
- Tier-1 catches: unknown archetype, out-of-range parameter, inverted range,
  off-universe pair, a non-constructible candidate (fast >= slow), and a
  planted execution/deploy intent — plus extra="forbid" rejecting a planted
  execution FIELD.
- malformed LLM output handled (schema reject + fatal-provider path).
- token-budget regression (stable prefix >= cache threshold).
- end-to-end through AgentRunner with a mocked provider + committed fixture.
- optional live smoke gated on RUN_LIVE_TESTS=1 + OPENAI_API_KEY.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from core.agents.evaluation import EvaluationGate
from core.agents.runner import AgentRunner
from core.agents.strategy_lab_agent import (
    _STABLE_SYSTEM,
    DEFAULT_TIMEFRAMES,
    DEFAULT_UNIVERSE,
    StrategyLabAgent,
    StrategyLabRequest,
)
from core.agents.types import AgentBudget, AgentCall
from core.llm_client import AgentResponse, LlmFatalError, ModelTier, TokenUsage
from core.models import (
    Granularity,
    MarketContextReport,
    ParamRange,
    RegimeAssessment,
    RiskState,
    StrategyArchetype,
    StrategyCandidate,
    StrategyParam,
    StrategyProposal,
    TrendState,
    VolState,
)
from core.strategy import MovingAverageCrossover, build_strategy

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 6, 28, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _context(run_id: str = "lab-fixture-001") -> MarketContextReport:
    return MarketContextReport(
        run_id=run_id,
        as_of=NOW,
        regimes=(
            RegimeAssessment(
                pair="USDJPY",
                risk_state=RiskState.RISK_ON,
                trend_state=TrendState.TRENDING_UP,
                vol_state=VolState.ELEVATED,
                confidence=Decimal("0.6"),
                rationale="firmer US yields",
            ),
        ),
    )


def _params(fast: str = "12", slow: str = "48", stop_pips: str = "70") -> tuple[StrategyParam, ...]:
    # Realistic stops are in PIPS now (70 pips on a JPY pair, not 0.004 price).
    return (
        StrategyParam(name="fast_period", value=Decimal(fast)),
        StrategyParam(name="slow_period", value=Decimal(slow)),
        StrategyParam(name="size", value=Decimal("1000")),
        StrategyParam(name="stop_distance_pips", value=Decimal(stop_pips)),
    )


def _ranges(
    fast: tuple[str, str] = ("5", "30"),
    slow: tuple[str, str] = ("30", "120"),
    stop_pips: tuple[str, str] = ("40", "120"),
) -> tuple[ParamRange, ...]:
    return (
        ParamRange(name="fast_period", low=Decimal(fast[0]), high=Decimal(fast[1])),
        ParamRange(name="slow_period", low=Decimal(slow[0]), high=Decimal(slow[1])),
        ParamRange(name="size", low=Decimal("500"), high=Decimal("2000")),
        ParamRange(
            name="stop_distance_pips", low=Decimal(stop_pips[0]), high=Decimal(stop_pips[1])
        ),
    )


def _candidate(**overrides: Any) -> StrategyCandidate:
    base: dict[str, Any] = {
        "candidate_id": "c1",
        "run_id": "lab-fixture-001",
        "archetype": StrategyArchetype.MA_CROSSOVER,
        "instrument": "USDJPY",
        "timeframe": Granularity.H1,
        "parameters": _params(),
        "parameter_ranges": _ranges(),
        "rationale": "USDJPY trending_up; a crossover rides the continuation.",
    }
    base.update(overrides)
    return StrategyCandidate(**base)


def _proposal(candidates: tuple[StrategyCandidate, ...] | None = None) -> StrategyProposal:
    return StrategyProposal(
        run_id="lab-fixture-001",
        as_of=NOW,
        candidates=candidates if candidates is not None else (_candidate(),),
    )


def _fixture_proposal() -> StrategyProposal:
    data = json.loads((FIXTURES / "strategy_proposal_sample.json").read_text())
    return StrategyProposal(**data)


def _checks() -> dict[str, Any]:
    agent = StrategyLabAgent.__new__(StrategyLabAgent)
    return {c.name: c.check for c in agent.evaluations()}


def _snapshot() -> dict[str, Any]:
    return {
        "run_id": "lab-fixture-001",
        "allowed_universe": list(DEFAULT_UNIVERSE),
        "allowed_timeframes": [t.value for t in DEFAULT_TIMEFRAMES],
        "max_candidates": 3,
    }


# ---------------------------------------------------------------------------
# prepare_call shape
# ---------------------------------------------------------------------------


def test_agent_name() -> None:
    assert StrategyLabAgent.name == "strategy_lab_agent"


def test_prepare_call_shape() -> None:
    agent = StrategyLabAgent()
    call = agent.prepare_call(
        StrategyLabRequest(run_id="lab-fixture-001", market_context=_context())
    )
    assert isinstance(call, AgentCall)
    assert call.tier == ModelTier.DEFAULT
    assert call.output_model is StrategyProposal
    # Stable prefix carries the registry catalog and crosses the cache threshold.
    assert "strategy_lab_agent" in call.stable_system
    assert "ma_crossover" in call.stable_system
    assert len(call.stable_system) >= 4096  # ~1024 tokens at chars/4
    # Volatile carries run_id + the allowed universe.
    assert "lab-fixture-001" in call.volatile_user
    assert "USDJPY" in call.volatile_user
    # Snapshot has what the Tier-1 checks need.
    assert call.input_snapshot["run_id"] == "lab-fixture-001"
    assert "USDJPY" in call.input_snapshot["allowed_universe"]
    assert call.input_snapshot["max_candidates"] == 3


def test_request_tier_override_promotes_to_heavy() -> None:
    agent = StrategyLabAgent()
    call = agent.prepare_call(
        StrategyLabRequest(run_id="r", market_context=_context(), tier=ModelTier.HEAVY)
    )
    assert call.tier == ModelTier.HEAVY


def test_evaluations_returns_required_checks() -> None:
    names = set(_checks())
    assert {
        "run_id_preserved",
        "candidate_count",
        "archetypes_in_registry",
        "pairs_in_universe",
        "timeframes_allowed",
        "params_and_ranges_sane",
        "candidates_constructible",
        "no_execution_intent",
    } <= names


# ---------------------------------------------------------------------------
# Valid candidate parses + is constructible
# ---------------------------------------------------------------------------


def test_valid_candidate_is_constructible_into_real_strategy() -> None:
    candidate = _candidate()
    strategy = build_strategy(candidate)
    assert isinstance(strategy, MovingAverageCrossover)
    assert strategy.params()["pair"] == "USDJPY"
    # And it passes every Tier-1 check.
    checks = _checks()
    snap = _snapshot()
    proposal = _proposal((candidate,))
    for name, fn in checks.items():
        res = fn(proposal, snap)
        assert res.passed, f"{name}: {res.reason}"


# ---------------------------------------------------------------------------
# Pip-relative stop bound — regression for the absolute-price-units bug that
# locked JPY pairs out of every proposal (surfaced by a live run). Uses
# REALISTIC per-pair stops, not conveniently-small ones (that masking is why
# the original hermetic suite missed it).
# ---------------------------------------------------------------------------


def _all_tier1_pass(candidate: StrategyCandidate) -> None:
    snap = _snapshot()
    for name, fn in _checks().items():
        res = fn(_proposal((candidate,)), snap)
        assert res.passed, f"{name}: {res.reason}"


def test_realistic_usdjpy_80pip_stop_passes_and_constructs() -> None:
    # 80 pips on USDJPY = 0.80 in price units — would have FAILED the old
    # [0.0001, 0.5] absolute cap. Now it passes and constructs.
    candidate = _candidate(
        instrument="USDJPY",
        parameters=_params(stop_pips="80"),
        parameter_ranges=_ranges(stop_pips=("40", "120")),
    )
    _all_tier1_pass(candidate)
    strategy = build_strategy(candidate)
    assert isinstance(strategy, MovingAverageCrossover)
    assert strategy.params()["stop_distance"] == "0.80"  # 80 * 0.01 pip size


def test_realistic_eurusd_50pip_stop_passes_and_constructs() -> None:
    candidate = _candidate(
        instrument="EURUSD",
        parameters=_params(stop_pips="50"),
        parameter_ranges=_ranges(stop_pips=("30", "80")),
    )
    _all_tier1_pass(candidate)
    strategy = build_strategy(candidate)
    assert isinstance(strategy, MovingAverageCrossover)
    assert strategy.params()["stop_distance"] == "0.0050"  # 50 * 0.0001 pip size


@pytest.mark.parametrize("pair", ["USDJPY", "EURUSD"])
def test_absurd_1000pip_stop_rejects_on_both_pair_types(pair: str) -> None:
    # 1000 pips is genuinely absurd on either pair type — it must still reject,
    # so the pip relaxation didn't just disable the bound.
    bad = _candidate(
        instrument=pair,
        parameters=_params(stop_pips="1000"),
        parameter_ranges=_ranges(stop_pips=("500", "1500")),
    )
    res = _checks()["params_and_ranges_sane"](_proposal((bad,)), _snapshot())
    assert not res.passed
    assert "stop_distance_pips" in res.reason


# ---------------------------------------------------------------------------
# Tier-1 catches each bad case
# ---------------------------------------------------------------------------


def test_tier1_catches_unknown_archetype() -> None:
    good = _candidate()
    bad = StrategyCandidate.model_construct(**{**good.__dict__, "archetype": "wat_crossover"})
    res = _checks()["archetypes_in_registry"](_proposal((bad,)), _snapshot())
    assert not res.passed
    assert "not in registry" in res.reason


def test_tier1_catches_out_of_range_parameter() -> None:
    # fast_period 500 exceeds the archetype max (200) and its own range.
    bad = _candidate(parameters=_params(fast="500"))
    res = _checks()["params_and_ranges_sane"](_proposal((bad,)), _snapshot())
    assert not res.passed
    assert "fast_period" in res.reason


def test_tier1_catches_inverted_range() -> None:
    # ParamRange's validator rejects low >= high at construction; bypass it with
    # model_construct so the Tier-1 deterministic check is what catches it.
    inverted = ParamRange.model_construct(name="fast_period", low=Decimal("30"), high=Decimal("5"))
    ranges = (inverted, *_ranges()[1:])
    bad = StrategyCandidate.model_construct(**{**_candidate().__dict__, "parameter_ranges": ranges})
    res = _checks()["params_and_ranges_sane"](_proposal((bad,)), _snapshot())
    assert not res.passed
    assert "high" in res.reason


def test_tier1_catches_off_universe_pair() -> None:
    bad = _candidate(instrument="NZDUSD")  # not in DEFAULT_UNIVERSE
    res = _checks()["pairs_in_universe"](_proposal((bad,)), _snapshot())
    assert not res.passed
    assert "NZDUSD" in res.reason


def test_tier1_catches_off_universe_timeframe() -> None:
    bad = _candidate(timeframe=Granularity.M5)  # not in DEFAULT_TIMEFRAMES
    res = _checks()["timeframes_allowed"](_proposal((bad,)), _snapshot())
    assert not res.passed


def test_tier1_catches_non_constructible_candidate() -> None:
    # fast >= slow: individually in-range/in-limits, but the constructor rejects
    # it. params_and_ranges passes; constructibility is the catch.
    bad = _candidate(
        parameters=_params(fast="40", slow="25"),
        parameter_ranges=_ranges(fast=("5", "100"), slow=("3", "100")),
    )
    snap = _snapshot()
    assert _checks()["params_and_ranges_sane"](_proposal((bad,)), snap).passed
    res = _checks()["candidates_constructible"](_proposal((bad,)), snap)
    assert not res.passed
    assert "not constructible" in res.reason


def test_tier1_catches_execution_intent_via_model_construct() -> None:
    # The model validator rejects execution language; model_construct bypasses
    # it, and the Tier-1 check catches the escape.
    bad = StrategyCandidate.model_construct(
        **{**_candidate().__dict__, "rationale": "Deploy this live to production now."}
    )
    res = _checks()["no_execution_intent"](_proposal((bad,)), _snapshot())
    assert not res.passed
    assert "execution intent" in res.reason


def test_tier1_catches_too_many_candidates() -> None:
    many = tuple(_candidate(candidate_id=f"c{i}") for i in range(5))
    res = _checks()["candidate_count"](_proposal(many), _snapshot())
    assert not res.passed


# ---------------------------------------------------------------------------
# Malformed output handled
# ---------------------------------------------------------------------------


def test_model_rejects_planted_execution_field() -> None:
    # extra="forbid" means a planted execution FIELD can't even parse.
    with pytest.raises(ValidationError):
        StrategyCandidate(  # type: ignore[call-arg]
            candidate_id="c",
            run_id="r",
            archetype=StrategyArchetype.MA_CROSSOVER,
            instrument="USDJPY",
            timeframe=Granularity.H1,
            parameters=_params(),
            parameter_ranges=_ranges(),
            rationale="ok",
            execute_live=True,
        )


def test_model_rejects_execution_language_in_rationale() -> None:
    with pytest.raises(ValidationError):
        _candidate(rationale="go live with this on real money")


def test_runner_surfaces_fatal_llm_error() -> None:
    agent = StrategyLabAgent()
    llm = MagicMock()
    llm.generate_structured.side_effect = LlmFatalError("schema validation failed")
    runner = AgentRunner(llm_provider=llm, evaluation_gate=EvaluationGate(), budget=AgentBudget())
    result = runner.run(
        agent,
        StrategyLabRequest(run_id="r", market_context=_context()),
        run_id="r",
    )
    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code == "fatal_llm_error"


# ---------------------------------------------------------------------------
# Token-budget regression
# ---------------------------------------------------------------------------


def test_stable_prefix_meets_cache_threshold() -> None:
    # >= ~1024 tokens (chars/4) so OpenAI auto-caches it; comfortably under the
    # 16k input ceiling on its own.
    assert len(_STABLE_SYSTEM) >= 4096
    assert len(_STABLE_SYSTEM) <= 16_000 * 4


# ---------------------------------------------------------------------------
# End-to-end through AgentRunner with a mocked provider + committed fixture
# ---------------------------------------------------------------------------


def test_end_to_end_via_runner_with_fixture_proposal() -> None:
    agent = StrategyLabAgent()
    canned = _fixture_proposal()

    llm = MagicMock()
    llm.generate_structured.return_value = (
        canned,
        AgentResponse(
            agent_name="strategy_lab_agent",
            run_id="lab-fixture-001:attempt-0",
            tier=ModelTier.DEFAULT,
            model="gpt-5.4",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=2000, cached_tokens=1200, completion_tokens=300),
            extra_metadata={"attempts": "1"},
        ),
    )
    runner = AgentRunner(llm_provider=llm, evaluation_gate=EvaluationGate(), budget=AgentBudget())
    result = runner.run(
        agent,
        StrategyLabRequest(run_id="lab-fixture-001", market_context=_context()),
        run_id="lab-fixture-001",
    )

    assert result.succeeded
    assert result.output is canned
    assert result.eval_verdict is not None
    assert result.eval_verdict.tier1_passed
    assert result.metrics.tier == ModelTier.DEFAULT
    # Every candidate in the accepted proposal is constructible.
    proposal = result.output
    assert isinstance(proposal, StrategyProposal)
    for c in proposal.candidates:
        assert isinstance(build_strategy(c), MovingAverageCrossover)


# ---------------------------------------------------------------------------
# Optional live smoke (skipped in CI)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1" or not os.environ.get("OPENAI_API_KEY"),
    reason="RUN_LIVE_TESTS=1 and OPENAI_API_KEY must be set",
)
def test_live_smoke_one_real_proposal() -> None:
    from core.config import OpenAISettings
    from core.llm_client import OpenAIProvider

    agent = StrategyLabAgent()
    provider = OpenAIProvider(OpenAISettings())  # type: ignore[call-arg]
    runner = AgentRunner(
        llm_provider=provider, evaluation_gate=EvaluationGate(), budget=AgentBudget()
    )
    result = runner.run(
        agent,
        StrategyLabRequest(run_id="lab-live-001", market_context=_context("lab-live-001")),
        run_id="lab-live-001",
    )
    assert result.succeeded, result.failure
    proposal = result.output
    assert isinstance(proposal, StrategyProposal)
    assert 1 <= len(proposal.candidates) <= 3
    for c in proposal.candidates:
        assert isinstance(build_strategy(c), MovingAverageCrossover)
