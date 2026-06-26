"""Tests for EvaluationGate — fully hermetic.

Covers:
- Tier-1 catches planted policy failures (trade-call, missing field,
  out-of-range) on the market_context_agent checks. Hard reject.
- Tier-2 OFF by default → no provider call (assert it).
- Tier-2 ON → judge invoked, verdict + reasons plumbed through, soft
  signals (flag/fail) returned without auto-rejecting.
- Tier-2 SAMPLED with explicit RNG.
- Tier-2 'fail' is still a SOFT signal — tier1_passed stays True.
- Judge unavailable (exception) → 'flag' returned with error in reasons.
- Misconfiguration: tier2_mode != "off" without provider raises at
  construction.
"""

from __future__ import annotations

import json
import random
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.agents.evaluation import EvaluationGate, JudgeVerdict
from core.agents.market_context_agent import MarketContextAgent
from core.agents.types import (
    AgentCall,
    EvalVerdict,
    StructuralCheck,
)
from core.llm_client import AgentResponse, ModelTier, TokenUsage
from core.models import MarketContextReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fixture_report() -> MarketContextReport:
    from pathlib import Path

    data = json.loads(
        (Path(__file__).parent / "fixtures" / "market_context_report_sample.json").read_text()
    )
    return MarketContextReport(**data)


def _agent_call_for(report: MarketContextReport) -> AgentCall:
    """Build a minimal AgentCall whose input_snapshot matches the report."""

    return AgentCall(
        stable_system="STABLE SYS PREFIX",
        volatile_user="VOLATILE",
        tier=ModelTier.DEFAULT,
        output_model=MarketContextReport,
        max_output_tokens=512,
        input_snapshot={"run_id": report.run_id, "watchlist": ["EURUSD"]},
    )


class _FakeAgent:
    """Stand-in Agent that exposes per-test Tier-1 checks."""

    def __init__(self, checks: tuple[StructuralCheck, ...]) -> None:
        self.name = "fake"
        self._checks = checks

    def prepare_call(self, _inputs: Any) -> AgentCall:  # not used here
        raise NotImplementedError

    def evaluations(self) -> tuple[StructuralCheck, ...]:
        return self._checks


def _market_context_agent() -> MarketContextAgent:
    return MarketContextAgent.__new__(MarketContextAgent)


def _construct_report_with(good: MarketContextReport, **overrides) -> MarketContextReport:
    base = {
        "run_id": good.run_id,
        "as_of": good.as_of,
        "schema_version": good.schema_version,
        "regimes": good.regimes,
        "key_scheduled_events": good.key_scheduled_events,
        "notable_surprises": good.notable_surprises,
        "sentiment": good.sentiment,
        "risk_flags": good.risk_flags,
        "notes": good.notes,
        "confidence": good.confidence,
    }
    base.update(overrides)
    return MarketContextReport.model_construct(**base)


# ---------------------------------------------------------------------------
# Tier-1 catches policy failures (using market_context_agent's checks)
# ---------------------------------------------------------------------------


def test_tier1_catches_planted_trade_call_hard_reject() -> None:
    good = _fixture_report()
    bad = _construct_report_with(good, notes="Go long EURUSD here.")
    gate = EvaluationGate()  # tier2 off, no provider needed
    verdict = gate.evaluate(
        agent=_market_context_agent(), call=_agent_call_for(good), output=bad
    )
    assert isinstance(verdict, EvalVerdict)
    assert not verdict.tier1_passed
    # The failing check name is in the failure string.
    assert any("no_trade_calls" in f for f in verdict.tier1_failures)
    assert not verdict.tier2_ran


def test_tier1_catches_out_of_range_confidence() -> None:
    from decimal import Decimal

    good = _fixture_report()
    bad = _construct_report_with(good, confidence=Decimal("1.5"))
    gate = EvaluationGate()
    verdict = gate.evaluate(
        agent=_market_context_agent(), call=_agent_call_for(good), output=bad
    )
    assert not verdict.tier1_passed
    assert any("confidences_bounded" in f for f in verdict.tier1_failures)


def test_tier1_catches_missing_run_id() -> None:
    good = _fixture_report()
    bad = _construct_report_with(good, run_id="")
    gate = EvaluationGate()
    verdict = gate.evaluate(
        agent=_market_context_agent(), call=_agent_call_for(good), output=bad
    )
    assert not verdict.tier1_passed
    assert any("required_fields_present" in f for f in verdict.tier1_failures)


def test_tier1_catches_run_id_mismatch_with_input_snapshot() -> None:
    good = _fixture_report()
    call = _agent_call_for(good)
    # The snapshot says run_id=ctx-fixture-001; planting a different report id.
    bad_call_snapshot = AgentCall(
        stable_system=call.stable_system,
        volatile_user=call.volatile_user,
        tier=call.tier,
        output_model=call.output_model,
        max_output_tokens=call.max_output_tokens,
        input_snapshot={"run_id": "DIFFERENT"},
    )
    gate = EvaluationGate()
    verdict = gate.evaluate(
        agent=_market_context_agent(), call=bad_call_snapshot, output=good
    )
    assert not verdict.tier1_passed
    assert any("run_id_preserved" in f for f in verdict.tier1_failures)


def test_tier1_check_that_raises_is_treated_as_failure() -> None:
    """A check that throws should not break the gate; it's recorded as a
    failure so the runner can hard-reject and the operator can see why."""

    def _explosive(_output, _snapshot):
        raise RuntimeError("kaboom")

    checks = (StructuralCheck(name="explosive", check=_explosive),)
    gate = EvaluationGate()
    verdict = gate.evaluate(
        agent=_FakeAgent(checks),
        call=_agent_call_for(_fixture_report()),
        output=_fixture_report(),
    )
    assert not verdict.tier1_passed
    assert any("explosive" in f and "kaboom" in f for f in verdict.tier1_failures)


# ---------------------------------------------------------------------------
# Tier-2 modes
# ---------------------------------------------------------------------------


def test_tier2_off_by_default_makes_no_provider_call() -> None:
    """The flagged invariant: with default (off), the gate touches no LLM."""
    provider = MagicMock()  # would record any call; should remain idle
    gate = EvaluationGate(llm_provider=provider)  # mode="off" default
    verdict = gate.evaluate(
        agent=_market_context_agent(),
        call=_agent_call_for(_fixture_report()),
        output=_fixture_report(),
    )
    assert verdict.tier1_passed
    assert not verdict.tier2_ran
    provider.generate_structured.assert_not_called()


def test_tier2_off_works_without_a_provider() -> None:
    gate = EvaluationGate()  # no provider, mode off — must not raise
    verdict = gate.evaluate(
        agent=_market_context_agent(),
        call=_agent_call_for(_fixture_report()),
        output=_fixture_report(),
    )
    assert verdict.tier1_passed
    assert not verdict.tier2_ran


def test_tier2_on_invokes_judge_and_plumbs_verdict() -> None:
    judge_verdict = JudgeVerdict(
        verdict="pass",
        coherence_score=0.82,
        spec_conformance_score=0.91,
        reasons=("All claims trace to the snapshot",),
    )
    provider = MagicMock()
    provider.generate_structured.return_value = (
        judge_verdict,
        AgentResponse(
            agent_name="market_context_agent__judge",
            run_id="judge-ctx-fixture-001",
            tier=ModelTier.CHEAP,
            model="gpt-5.4-nano",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=400, cached_tokens=0, completion_tokens=40),
        ),
    )
    gate = EvaluationGate(llm_provider=provider, tier2_mode="on")
    verdict = gate.evaluate(
        agent=_market_context_agent(),
        call=_agent_call_for(_fixture_report()),
        output=_fixture_report(),
    )
    assert verdict.tier1_passed
    assert verdict.tier2_ran
    assert verdict.tier2_verdict == "pass"
    assert verdict.tier2_reasons == ("All claims trace to the snapshot",)
    # Judge was invoked on the CHEAP tier.
    kwargs = provider.generate_structured.call_args.kwargs
    assert kwargs["tier"] == ModelTier.CHEAP


def test_tier2_flag_is_a_soft_signal_tier1_still_passed() -> None:
    """A judge 'flag' must NOT auto-reject; the gate returns tier1_passed=True
    and surfaces the verdict for the runner to log."""
    judge_verdict = JudgeVerdict(
        verdict="flag",
        coherence_score=0.55,
        spec_conformance_score=0.80,
        reasons=("minor inconsistency between regime confidence and rationale strength",),
    )
    provider = MagicMock()
    provider.generate_structured.return_value = (judge_verdict, _judge_response())
    gate = EvaluationGate(llm_provider=provider, tier2_mode="on")
    verdict = gate.evaluate(
        agent=_market_context_agent(),
        call=_agent_call_for(_fixture_report()),
        output=_fixture_report(),
    )
    assert verdict.tier1_passed is True
    assert verdict.tier2_ran is True
    assert verdict.tier2_verdict == "flag"


def test_tier2_fail_is_also_a_soft_signal() -> None:
    judge_verdict = JudgeVerdict(
        verdict="fail",
        coherence_score=0.20,
        spec_conformance_score=0.50,
        reasons=("output contradicts the brief on two regimes",),
    )
    provider = MagicMock()
    provider.generate_structured.return_value = (judge_verdict, _judge_response())
    gate = EvaluationGate(llm_provider=provider, tier2_mode="on")
    verdict = gate.evaluate(
        agent=_market_context_agent(),
        call=_agent_call_for(_fixture_report()),
        output=_fixture_report(),
    )
    # Tier-1 still passes (the judge's opinion does not invalidate Tier-1).
    assert verdict.tier1_passed is True
    assert verdict.tier2_verdict == "fail"


def test_tier2_sampled_uses_rng() -> None:
    """sampled mode + sample_rate=0.5 + deterministic RNG: alternate runs."""
    judge_verdict = JudgeVerdict(
        verdict="pass",
        coherence_score=0.9,
        spec_conformance_score=0.9,
        reasons=("ok",),
    )
    provider = MagicMock()
    provider.generate_structured.return_value = (judge_verdict, _judge_response())
    # A seeded RNG whose first call returns < 0.5 (run) and next returns >= 0.5 (skip).
    rng = random.Random(42)
    gate = EvaluationGate(
        llm_provider=provider,
        tier2_mode="sampled",
        tier2_sample_rate=0.5,
        rng=rng,
    )
    seen_ran: list[bool] = []
    for _ in range(8):
        v = gate.evaluate(
            agent=_market_context_agent(),
            call=_agent_call_for(_fixture_report()),
            output=_fixture_report(),
        )
        seen_ran.append(v.tier2_ran)
    # At sample_rate=0.5 over 8 trials we expect SOME run and SOME skip.
    assert any(seen_ran)
    assert not all(seen_ran)


def test_judge_unavailable_returns_flag_with_error_in_reasons() -> None:
    provider = MagicMock()
    provider.generate_structured.side_effect = RuntimeError("openai down")
    gate = EvaluationGate(llm_provider=provider, tier2_mode="on")
    verdict = gate.evaluate(
        agent=_market_context_agent(),
        call=_agent_call_for(_fixture_report()),
        output=_fixture_report(),
    )
    assert verdict.tier1_passed
    assert verdict.tier2_ran
    assert verdict.tier2_verdict == "flag"
    assert any("judge unavailable" in r for r in verdict.tier2_reasons)


# ---------------------------------------------------------------------------
# Misconfiguration
# ---------------------------------------------------------------------------


def test_tier2_on_without_provider_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="requires an llm_provider"):
        EvaluationGate(tier2_mode="on")


def test_invalid_sample_rate_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        EvaluationGate(tier2_sample_rate=1.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _judge_response() -> AgentResponse:
    return AgentResponse(
        agent_name="market_context_agent__judge",
        run_id="judge",
        tier=ModelTier.CHEAP,
        model="gpt-5.4-nano",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=400, cached_tokens=0, completion_tokens=40),
    )
