"""Shared types for the agent execution + evaluation layer.

Kept in a single module so :mod:`core.agents.runner` and
:mod:`core.agents.evaluation` can depend on the same data shapes without
circular imports.

The :class:`Agent` Protocol is the surface every domain agent implements.
The :class:`AgentRunner` is the only place that issues LLM calls; agents
provide :meth:`Agent.prepare_call` and :meth:`Agent.evaluations`, the runner
provides timeouts, retries, budget enforcement, and structured logging.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from core.llm_client import ModelTier

# ---------------------------------------------------------------------------
# Agent surface (Protocol every domain agent implements)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgentCall:
    """A runner-ready description of one LLM call.

    The agent builds this in :meth:`Agent.prepare_call`; the runner issues
    it through :class:`core.llm_client.LLMProvider`; the evaluator inspects
    ``(input_snapshot, output)``.

    ``input_snapshot`` is a JSON-serialisable view of the agent's input that
    the evaluator and any auditor needs to see. For
    :mod:`market_context_agent` it's the brief as JSON.
    """

    stable_system: str
    volatile_user: str
    tier: ModelTier
    output_model: type[BaseModel]
    max_output_tokens: int = 2048
    input_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of one structural / policy check."""

    passed: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class StructuralCheck:
    """A single Tier-1 deterministic check.

    ``check`` takes ``(output, input_snapshot)`` and returns
    :class:`CheckResult`. The function should be pure — no I/O, no LLM
    calls; that's the whole point of Tier 1.
    """

    name: str
    check: Callable[[BaseModel, dict[str, Any]], CheckResult]


class Agent(Protocol):
    """The interface every domain agent implements.

    Agents do NOT call the LLM directly — they only describe what call
    should happen. The :class:`AgentRunner` issues the call and surrounds
    it with mitigations.
    """

    name: str

    def prepare_call(self, inputs: Any) -> AgentCall:
        """Build the :class:`AgentCall` for ``inputs``. Pure / deterministic.

        May read from injected providers (market data, context feeds) but
        must not call the LLM.

        Raises any exception type the agent wants — the runner catches and
        wraps it.
        """
        ...

    def evaluations(self) -> tuple[StructuralCheck, ...]:
        """Per-agent Tier-1 deterministic checks. Always on, free, hard-reject."""
        ...


# ---------------------------------------------------------------------------
# Evaluation verdict
# ---------------------------------------------------------------------------


Tier2VerdictLiteral = Literal["pass", "flag", "fail"]


@dataclass(frozen=True, slots=True)
class EvalVerdict:
    """Combined Tier-1 + Tier-2 evaluation verdict.

    ``tier1_passed=False`` is a HARD REJECT — the runner discards the
    output. ``tier2_verdict='flag'`` is a SOFT signal — logged for human
    review, NOT auto-rejected. ``tier2_verdict='fail'`` is also soft by
    default; whether it rejects is the orchestrator's decision (we surface
    it cleanly; we do not silently bin output a judge LLM disliked).
    """

    tier1_passed: bool
    tier1_failures: tuple[str, ...] = ()
    tier2_ran: bool = False
    tier2_verdict: Tier2VerdictLiteral | None = None
    tier2_reasons: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgentBudget:
    """Token and cost caps for one runner.

    Per-call caps protect against a single call going wild; the per-cycle
    cost cap is the *cost-side analog of the kill switch* — a breach
    :raises: :class:`AgentBudgetExceeded` and the runner stops, requiring
    a human or supervisor to reset.
    """

    max_prompt_tokens_per_call: int = 50_000
    max_completion_tokens_per_call: int = 8_192
    max_cost_per_cycle_usd: Decimal | None = Decimal("1.00")
    # Optional: cap the per-call timeout. None lets the SDK default ride.
    per_call_timeout_seconds: float | None = 60.0


class AgentBudgetExceeded(Exception):
    """The cost-side analog of the kill switch.

    Raised when a single call's prompt or completion ceiling is breached,
    or when the per-cycle cost cap is. The runner DOES NOT catch this —
    the caller is expected to halt the loop and surface to a human.
    """


# ---------------------------------------------------------------------------
# Metrics + result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgentMetrics:
    """Observability metrics for one runner invocation."""

    agent_name: str
    run_id: str
    tier: ModelTier | None
    model: str | None
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    estimated_cost_usd: Decimal
    cache_hit_ratio: float
    latency_seconds: float
    attempts: int  # 1 on success-first-try; > 1 if retries happened


FailureCode = Literal[
    "preparation_error",
    "transient_exhausted",
    "fatal_llm_error",
    "eval_rejected",
    "unknown_error",
]


@dataclass(frozen=True, slots=True)
class AgentRunFailure:
    """Returned by :meth:`AgentRunner.run` on non-budget failure.

    Budget failures :raise: :class:`AgentBudgetExceeded` instead of being
    wrapped here — the cost-side kill switch is *not* a routine outcome.
    """

    code: FailureCode
    reason: str
    tier1_failures: tuple[str, ...] = ()  # populated only when code == "eval_rejected"


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Outcome of one :meth:`AgentRunner.run` call.

    Exactly one of ``output`` and ``failure`` is set. ``metrics`` is always
    populated. ``eval_verdict`` is set when the runner reached the eval
    stage (i.e. the LLM call returned something).
    """

    output: BaseModel | None
    failure: AgentRunFailure | None
    metrics: AgentMetrics
    eval_verdict: EvalVerdict | None = None

    @property
    def succeeded(self) -> bool:
        return self.output is not None


__all__ = [
    "Agent",
    "AgentBudget",
    "AgentBudgetExceeded",
    "AgentCall",
    "AgentMetrics",
    "AgentRunFailure",
    "AgentRunResult",
    "CheckResult",
    "EvalVerdict",
    "FailureCode",
    "StructuralCheck",
    "Tier2VerdictLiteral",
]
