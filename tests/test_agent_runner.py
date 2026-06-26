"""Tests for AgentRunner and BudgetTracker — fully hermetic.

Verifies:
- Success → validated output + populated metrics + Tier-1 verdict.
- Transient errors → retry → eventual success.
- Transient errors exhausted → AgentRunFailure(code="transient_exhausted").
- LlmFatalError → AgentRunFailure(code="fatal_llm_error") (NO retry).
- Per-call budget breach → AgentBudgetExceeded RAISED (cost-side kill switch).
- Per-cycle cost cap breach → AgentBudgetExceeded RAISED.
- Eval Tier-1 failure → AgentRunFailure(code="eval_rejected") + output discarded.
- preparation_error path: prepare_call() raising surfaces as
  AgentRunFailure(code="preparation_error").
- Structured-log shape on success.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ConfigDict, Field

from core.agents.runner import AgentRunner, BudgetTracker, new_run_id
from core.agents.types import (
    Agent,
    AgentBudget,
    AgentBudgetExceeded,
    AgentCall,
    CheckResult,
    EvalVerdict,
    StructuralCheck,
)
from core.llm_client import (
    AgentResponse,
    LlmFatalError,
    LlmTransientError,
    ModelTier,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Small test output model + fake agent
# ---------------------------------------------------------------------------


class TinyReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    title: str
    score: float = Field(ge=0.0, le=1.0)


@dataclass
class FakeAgent:
    """Minimal :class:`Agent` for runner tests.

    Customisable per test: which AgentCall to return, which Tier-1 checks
    to apply, whether prepare_call raises.
    """

    name: str = "fake_agent"
    call: AgentCall | None = None
    checks: tuple[StructuralCheck, ...] = ()
    prepare_error: Exception | None = None

    def prepare_call(self, _inputs: Any) -> AgentCall:
        if self.prepare_error is not None:
            raise self.prepare_error
        assert self.call is not None
        return self.call

    def evaluations(self) -> tuple[StructuralCheck, ...]:
        return self.checks


def _default_call() -> AgentCall:
    return AgentCall(
        stable_system="STABLE PREFIX — pretend this is long and identical.",
        volatile_user="VOLATILE BRIEF for this call.",
        tier=ModelTier.DEFAULT,
        output_model=TinyReport,
        max_output_tokens=512,
        input_snapshot={"run_id": "r1"},
    )


def _ok_response(prompt: int = 1500, cached: int = 900, completion: int = 80) -> AgentResponse:
    return AgentResponse(
        agent_name="fake_agent",
        run_id="r1:attempt-0",
        tier=ModelTier.DEFAULT,
        model="gpt-5.4",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=prompt, cached_tokens=cached, completion_tokens=completion),
        extra_metadata={"attempts": "1"},
    )


def _passing_gate() -> Any:
    gate = MagicMock()
    gate.evaluate.return_value = EvalVerdict(tier1_passed=True)
    return gate


def _failing_gate(failures: tuple[str, ...] = ("planted: planted reason",)) -> Any:
    gate = MagicMock()
    gate.evaluate.return_value = EvalVerdict(tier1_passed=False, tier1_failures=failures)
    return gate


def _flag_gate() -> Any:
    gate = MagicMock()
    gate.evaluate.return_value = EvalVerdict(
        tier1_passed=True, tier2_ran=True, tier2_verdict="flag",
        tier2_reasons=("looks borderline",),
    )
    return gate


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_success_returns_output_metrics_and_passing_verdict() -> None:
    llm = MagicMock()
    parsed = TinyReport(title="ok", score=0.5)
    llm.generate_structured.return_value = (parsed, _ok_response())
    runner = AgentRunner(
        llm_provider=llm,
        evaluation_gate=_passing_gate(),
        budget=AgentBudget(),
    )
    agent = FakeAgent(call=_default_call())

    result = runner.run(agent, inputs=None, run_id="r1")

    assert result.succeeded
    assert result.output is parsed
    assert result.eval_verdict is not None
    assert result.eval_verdict.tier1_passed
    m = result.metrics
    assert m.tier == ModelTier.DEFAULT
    assert m.model == "gpt-5.4"
    assert m.prompt_tokens == 1500
    assert m.cached_tokens == 900
    assert m.completion_tokens == 80
    assert m.cache_hit_ratio == pytest.approx(0.6)
    assert m.estimated_cost_usd > 0
    assert m.attempts == 1
    assert m.latency_seconds >= 0.0


# ---------------------------------------------------------------------------
# Transient retry + exhaustion
# ---------------------------------------------------------------------------


def test_transient_error_then_success_retries_and_returns_output() -> None:
    parsed = TinyReport(title="ok", score=0.5)
    llm = MagicMock()
    llm.generate_structured.side_effect = [
        LlmTransientError("network blip"),
        (parsed, _ok_response()),
    ]
    runner = AgentRunner(
        llm_provider=llm,
        evaluation_gate=_passing_gate(),
        budget=AgentBudget(),
        sleep=lambda _: None,
        base_backoff_seconds=0.0,
    )
    result = runner.run(FakeAgent(call=_default_call()), inputs=None, run_id="r1")
    assert result.succeeded
    assert llm.generate_structured.call_count == 2


def test_transient_exhausted_returns_failure_no_crash() -> None:
    llm = MagicMock()
    llm.generate_structured.side_effect = LlmTransientError("persistent 5xx")
    runner = AgentRunner(
        llm_provider=llm,
        evaluation_gate=_passing_gate(),
        budget=AgentBudget(),
        retries=2,
        sleep=lambda _: None,
        base_backoff_seconds=0.0,
    )
    result = runner.run(FakeAgent(call=_default_call()), inputs=None, run_id="r1")
    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code == "transient_exhausted"
    # 1 initial + 2 retries = 3 attempts.
    assert llm.generate_structured.call_count == 3
    assert result.metrics.attempts == 3


def test_fatal_llm_error_does_not_retry() -> None:
    llm = MagicMock()
    llm.generate_structured.side_effect = LlmFatalError("400 bad request")
    runner = AgentRunner(
        llm_provider=llm,
        evaluation_gate=_passing_gate(),
        budget=AgentBudget(),
        retries=2,
        sleep=lambda _: None,
    )
    result = runner.run(FakeAgent(call=_default_call()), inputs=None, run_id="r1")
    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code == "fatal_llm_error"
    assert llm.generate_structured.call_count == 1


# ---------------------------------------------------------------------------
# Budget enforcement — RAISES (cost-side kill switch)
# ---------------------------------------------------------------------------


def test_pre_call_completion_cap_breach_raises() -> None:
    llm = MagicMock()
    runner = AgentRunner(
        llm_provider=llm,
        evaluation_gate=_passing_gate(),
        budget=AgentBudget(max_completion_tokens_per_call=100),
    )
    # Call asks for 512 completion tokens; cap is 100.
    with pytest.raises(AgentBudgetExceeded, match="completion"):
        runner.run(FakeAgent(call=_default_call()), inputs=None, run_id="r1")
    # The LLM was never called — pre-flight gate fired.
    llm.generate_structured.assert_not_called()


def test_pre_call_prompt_estimate_breach_raises() -> None:
    llm = MagicMock()
    big = _default_call().__class__(
        stable_system="x" * 200_000,  # ~50k tokens estimate
        volatile_user="y",
        tier=ModelTier.DEFAULT,
        output_model=TinyReport,
        max_output_tokens=128,
    )
    runner = AgentRunner(
        llm_provider=llm,
        evaluation_gate=_passing_gate(),
        budget=AgentBudget(max_prompt_tokens_per_call=10_000),
    )
    with pytest.raises(AgentBudgetExceeded, match="prompt tokens"):
        runner.run(FakeAgent(call=big), inputs=None, run_id="r1")
    llm.generate_structured.assert_not_called()


def test_post_call_prompt_cap_breach_raises_after_one_call() -> None:
    """Pre-flight passes (small estimate) but the actual prompt comes back
    huge from the LLM usage report — runner records & raises."""
    llm = MagicMock()
    parsed = TinyReport(title="x", score=0.5)
    llm.generate_structured.return_value = (
        parsed,
        _ok_response(prompt=200_000, cached=0, completion=10),
    )
    runner = AgentRunner(
        llm_provider=llm,
        evaluation_gate=_passing_gate(),
        budget=AgentBudget(max_prompt_tokens_per_call=10_000),
    )
    with pytest.raises(AgentBudgetExceeded, match="actual prompt tokens"):
        runner.run(FakeAgent(call=_default_call()), inputs=None, run_id="r1")


def test_cycle_cost_cap_breach_raises() -> None:
    """A single call whose cost exceeds the cycle cap raises.

    Loosen the per-call prompt cap so it doesn't fire first; we want to
    isolate the cycle-cost gate.
    """
    parsed = TinyReport(title="x", score=0.5)
    llm = MagicMock()
    llm.generate_structured.return_value = (
        parsed,
        # 2M prompt + 5k completion on gpt-5.4 → well above $0.01.
        _ok_response(prompt=2_000_000, cached=0, completion=5_000),
    )
    runner = AgentRunner(
        llm_provider=llm,
        evaluation_gate=_passing_gate(),
        budget=AgentBudget(
            max_prompt_tokens_per_call=5_000_000,
            max_cost_per_cycle_usd=Decimal("0.01"),
        ),
    )
    with pytest.raises(AgentBudgetExceeded, match="cycle cost"):
        runner.run(FakeAgent(call=_default_call()), inputs=None, run_id="r1")


def test_budget_tracker_reset_cycle_zeroes_running_cost() -> None:
    bt = BudgetTracker(AgentBudget(max_cost_per_cycle_usd=Decimal("1000")))
    bt.record(prompt_tokens=10, cost_usd=Decimal("0.5"))
    assert bt.cycle_cost_usd == Decimal("0.5")
    bt.reset_cycle()
    assert bt.cycle_cost_usd == Decimal("0")


# ---------------------------------------------------------------------------
# Evaluation rejection
# ---------------------------------------------------------------------------


def test_tier1_failure_returns_eval_rejected_and_discards_output() -> None:
    parsed = TinyReport(title="bad", score=0.5)
    llm = MagicMock()
    llm.generate_structured.return_value = (parsed, _ok_response())
    runner = AgentRunner(
        llm_provider=llm,
        evaluation_gate=_failing_gate(failures=("planted: trade call",)),
        budget=AgentBudget(),
    )
    result = runner.run(FakeAgent(call=_default_call()), inputs=None, run_id="r1")
    assert not result.succeeded
    assert result.output is None
    assert result.failure is not None
    assert result.failure.code == "eval_rejected"
    assert result.failure.tier1_failures == ("planted: trade call",)
    # Metrics ARE populated even on eval rejection (we made the call).
    assert result.metrics.prompt_tokens == 1500


def test_tier2_flag_is_soft_signal_output_still_returned() -> None:
    parsed = TinyReport(title="borderline", score=0.5)
    llm = MagicMock()
    llm.generate_structured.return_value = (parsed, _ok_response())
    runner = AgentRunner(
        llm_provider=llm,
        evaluation_gate=_flag_gate(),
        budget=AgentBudget(),
    )
    result = runner.run(FakeAgent(call=_default_call()), inputs=None, run_id="r1")
    assert result.succeeded
    assert result.output is parsed
    assert result.eval_verdict is not None
    assert result.eval_verdict.tier2_verdict == "flag"


# ---------------------------------------------------------------------------
# Preparation error
# ---------------------------------------------------------------------------


def test_prepare_call_exception_becomes_preparation_error() -> None:
    llm = MagicMock()
    runner = AgentRunner(
        llm_provider=llm,
        evaluation_gate=_passing_gate(),
        budget=AgentBudget(),
    )
    agent = FakeAgent(prepare_error=RuntimeError("provider blew up"))
    result = runner.run(agent, inputs=None, run_id="r1")
    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code == "preparation_error"
    llm.generate_structured.assert_not_called()


# ---------------------------------------------------------------------------
# Structured log shape on success
# ---------------------------------------------------------------------------


def test_success_emits_one_agent_run_ok_record(capsys) -> None:
    parsed = TinyReport(title="ok", score=0.5)
    llm = MagicMock()
    llm.generate_structured.return_value = (parsed, _ok_response())
    runner = AgentRunner(
        llm_provider=llm,
        evaluation_gate=_passing_gate(),
        budget=AgentBudget(),
    )
    runner.run(FakeAgent(call=_default_call()), inputs=None, run_id="r1")
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert '"event": "agent_run_ok"' in out
    # Fields that must be present in every success record.
    for field in (
        "agent",
        "run_id",
        "tier",
        "model",
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "cache_hit_ratio",
        "estimated_cost_usd",
        "latency_seconds",
        "cycle_cost_usd",
    ):
        assert f'"{field}"' in out, f"missing {field} in log"


def test_new_run_id_is_unique() -> None:
    assert new_run_id("ctx") != new_run_id("ctx")


# Tag the unused imports so ruff/flake8 leave them.
_ = (CheckResult, Callable, Agent)
