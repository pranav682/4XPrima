"""AgentRunner — the thin reusable wrapper every agent call goes through.

What it does (and only what it does):

1. Builds the LLM call via :meth:`Agent.prepare_call`.
2. Pre-flight budget check; if the call would breach the per-call cap,
   :raises: :class:`AgentBudgetExceeded` — the cost-side analog of the
   kill switch.
3. Issues the call through the injected :class:`LLMProvider` with a
   per-call timeout, retrying on :class:`LlmTransientError` with bounded
   exponential backoff + jitter. After ``retries`` attempts, returns
   ``AgentRunFailure(code="transient_exhausted")``.
4. Post-call budget check (the live numbers); breach :raises:
   :class:`AgentBudgetExceeded`.
5. Runs the :class:`EvaluationGate`. Tier-1 failure → hard reject (the
   output is discarded; failure returned). Tier-2 ``flag``/``fail`` is
   logged and surfaced, NOT auto-rejected.
6. Emits one structured-log record per run with everything — the
   observability spine.

What it deliberately does NOT do:

- Crash the loop. All operational failures come back as
  :class:`AgentRunFailure`. The only exception that escapes is
  :class:`AgentBudgetExceeded` — and that's by design.
- Decide which model to use. The agent's :meth:`prepare_call` picks the
  tier; the runner respects it.
- Reach into the agent's internals. The runner only knows the Protocol.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from decimal import Decimal
from typing import Any

import structlog

from core.agents.cost import estimate_cost
from core.agents.types import (
    Agent,
    AgentBudget,
    AgentBudgetExceeded,
    AgentMetrics,
    AgentRunFailure,
    AgentRunResult,
    EvalVerdict,
)
from core.llm_client import (
    AgentResponse,
    LlmFatalError,
    LLMProvider,
    LlmTransientError,
)

# ---------------------------------------------------------------------------
# Budget tracker (mutable per-cycle counter)
# ---------------------------------------------------------------------------


def _estimate_prompt_tokens(stable_system: str, volatile_user: str) -> int:
    """Rough chars/4 estimate — good enough for a pre-flight ceiling check.

    Deliberately conservative (slightly under-counts) so the *post*-call
    enforcement is the authoritative gate; pre-flight just rejects orders
    of magnitude wrong.
    """
    return (len(stable_system) + len(volatile_user)) // 4


class BudgetTracker:
    """Accumulates per-cycle spend and enforces caps.

    Not thread-safe — the slow loop is single-threaded by design.
    """

    def __init__(self, budget: AgentBudget) -> None:
        self._budget = budget
        self._cycle_cost_usd = Decimal("0")

    @property
    def cycle_cost_usd(self) -> Decimal:
        return self._cycle_cost_usd

    def reset_cycle(self) -> None:
        self._cycle_cost_usd = Decimal("0")

    def check_pre_call(
        self,
        *,
        stable_system: str,
        volatile_user: str,
        max_output_tokens: int,
    ) -> None:
        if max_output_tokens > self._budget.max_completion_tokens_per_call:
            raise AgentBudgetExceeded(
                f"requested completion tokens {max_output_tokens} > cap "
                f"{self._budget.max_completion_tokens_per_call}"
            )
        est = _estimate_prompt_tokens(stable_system, volatile_user)
        if est > self._budget.max_prompt_tokens_per_call:
            raise AgentBudgetExceeded(
                f"estimated prompt tokens {est} > per-call cap "
                f"{self._budget.max_prompt_tokens_per_call}"
            )

    def record(self, *, prompt_tokens: int, cost_usd: Decimal) -> None:
        if prompt_tokens > self._budget.max_prompt_tokens_per_call:
            raise AgentBudgetExceeded(
                f"actual prompt tokens {prompt_tokens} > per-call cap "
                f"{self._budget.max_prompt_tokens_per_call}"
            )
        self._cycle_cost_usd += cost_usd
        if (
            self._budget.max_cost_per_cycle_usd is not None
            and self._cycle_cost_usd > self._budget.max_cost_per_cycle_usd
        ):
            raise AgentBudgetExceeded(
                f"cycle cost ${self._cycle_cost_usd} > cap "
                f"${self._budget.max_cost_per_cycle_usd}"
            )


# ---------------------------------------------------------------------------
# The Runner
# ---------------------------------------------------------------------------


class _EvalGateProtocol:
    """Inline duck-typed Protocol for the evaluation gate.

    Defined here (and not as a real :class:`typing.Protocol`) to keep
    ``core.agents.runner`` import-side from referencing
    ``core.agents.evaluation`` directly; the test suite can pass a mock
    that implements ``.evaluate(agent, call, output) -> EvalVerdict``.
    """


class AgentRunner:
    """The only path that issues agent LLM calls.

    Construction:

        >>> runner = AgentRunner(
        ...     llm_provider=provider,
        ...     evaluation_gate=gate,
        ...     budget=AgentBudget(),
        ... )

    Use:

        >>> result = runner.run(agent, request, run_id="r-001")
        >>> if result.succeeded:
        ...     consume(result.output)
        ... else:
        ...     log(result.failure)
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        evaluation_gate: Any,        # duck-typed: has .evaluate(agent, call, output)
        budget: AgentBudget | None = None,
        retries: int = 2,
        base_backoff_seconds: float = 0.5,
        sleep: Any = time.sleep,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        self._llm = llm_provider
        self._gate = evaluation_gate
        self._budget_tracker = BudgetTracker(budget or AgentBudget())
        self._budget = budget or AgentBudget()
        self._retries = retries
        self._base_backoff = base_backoff_seconds
        self._sleep = sleep
        self._logger = (
            logger if logger is not None else _default_logger()
        ).bind(component="agent_runner")

    # ------------------------------------------------------------- props

    @property
    def budget_tracker(self) -> BudgetTracker:
        """Exposed for tests + dev tooling so spend can be inspected mid-cycle."""
        return self._budget_tracker

    # --------------------------------------------------------------- run

    def run(self, agent: Agent, inputs: Any, *, run_id: str) -> AgentRunResult:
        start = time.perf_counter()
        log = self._logger.bind(agent=agent.name, run_id=run_id)

        # 1. Prepare the call (deterministic; agent-side).
        try:
            call = agent.prepare_call(inputs)
        except Exception as e:
            log.error("agent_prepare_call_failed", error=repr(e))
            return self._failure_result(
                agent_name=agent.name,
                run_id=run_id,
                start=start,
                attempts=0,
                failure=AgentRunFailure(
                    code="preparation_error",
                    reason=f"{type(e).__name__}: {e}",
                ),
            )

        # 2. Pre-flight budget check. Breach RAISES (cost-side kill switch).
        self._budget_tracker.check_pre_call(
            stable_system=call.stable_system,
            volatile_user=call.volatile_user,
            max_output_tokens=call.max_output_tokens,
        )

        # 3. Issue the LLM call with retry on transient errors.
        try:
            output, response = self._call_with_retry(
                agent_name=agent.name, run_id=run_id, call=call, log=log
            )
        except _TransientExhausted as e:
            return self._failure_result(
                agent_name=agent.name,
                run_id=run_id,
                start=start,
                attempts=e.attempts,
                failure=AgentRunFailure(
                    code="transient_exhausted",
                    reason=e.reason,
                ),
            )
        except LlmFatalError as e:
            return self._failure_result(
                agent_name=agent.name,
                run_id=run_id,
                start=start,
                attempts=1,
                failure=AgentRunFailure(
                    code="fatal_llm_error",
                    reason=f"{type(e).__name__}: {e}",
                ),
            )

        # 4. Post-call budget check — uses the real prompt count and cost.
        cost = estimate_cost(
            response.model,
            prompt_tokens=response.usage.prompt_tokens,
            cached_tokens=response.usage.cached_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
        # Budget breach RAISES.
        self._budget_tracker.record(
            prompt_tokens=response.usage.prompt_tokens, cost_usd=cost
        )

        # 5. Evaluation gate.
        verdict: EvalVerdict = self._gate.evaluate(agent=agent, call=call, output=output)

        attempts = response.extra_metadata.get("attempts", "1")
        try:
            attempts_int = int(attempts)
        except ValueError:
            attempts_int = 1

        metrics = AgentMetrics(
            agent_name=agent.name,
            run_id=run_id,
            tier=response.tier,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            cached_tokens=response.usage.cached_tokens,
            completion_tokens=response.usage.completion_tokens,
            estimated_cost_usd=cost,
            cache_hit_ratio=response.usage.cache_hit_ratio,
            latency_seconds=time.perf_counter() - start,
            attempts=attempts_int,
        )

        if not verdict.tier1_passed:
            log.warning(
                "agent_run_eval_rejected",
                tier1_failures=list(verdict.tier1_failures),
                tier2_verdict=verdict.tier2_verdict,
            )
            return AgentRunResult(
                output=None,
                failure=AgentRunFailure(
                    code="eval_rejected",
                    reason="Tier-1 policy check failed: "
                    + "; ".join(verdict.tier1_failures),
                    tier1_failures=verdict.tier1_failures,
                ),
                metrics=metrics,
                eval_verdict=verdict,
            )

        # Tier-2 'flag' / 'fail' is SOFT — log it, but pass the output.
        if verdict.tier2_ran and verdict.tier2_verdict in ("flag", "fail"):
            log.warning(
                "agent_run_tier2_soft_signal",
                tier2_verdict=verdict.tier2_verdict,
                tier2_reasons=list(verdict.tier2_reasons),
            )

        log.info(
            "agent_run_ok",
            tier=response.tier.value,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            cached_tokens=response.usage.cached_tokens,
            completion_tokens=response.usage.completion_tokens,
            cache_hit_ratio=f"{response.usage.cache_hit_ratio:.3f}",
            estimated_cost_usd=str(cost),
            latency_seconds=f"{metrics.latency_seconds:.3f}",
            attempts=attempts_int,
            tier2_ran=verdict.tier2_ran,
            tier2_verdict=verdict.tier2_verdict,
            cycle_cost_usd=str(self._budget_tracker.cycle_cost_usd),
        )

        return AgentRunResult(
            output=output, failure=None, metrics=metrics, eval_verdict=verdict
        )

    # ---------------------------------------------------------- internals

    def _call_with_retry(
        self,
        *,
        agent_name: str,
        run_id: str,
        call: Any,
        log: structlog.stdlib.BoundLogger,
    ) -> tuple[Any, AgentResponse]:
        last_reason = ""
        for attempt in range(self._retries + 1):
            try:
                output, response = self._llm.generate_structured(
                    agent_name=agent_name,
                    run_id=f"{run_id}:attempt-{attempt}",
                    tier=call.tier,
                    stable_system=call.stable_system,
                    volatile_user=call.volatile_user,
                    output_model=call.output_model,
                    max_output_tokens=call.max_output_tokens,
                    timeout_seconds=self._budget.per_call_timeout_seconds,
                    extra_metadata={"attempts": str(attempt + 1)},
                )
                return output, response
            except LlmTransientError as e:
                last_reason = f"{type(e).__name__}: {e}"
                log.warning(
                    "agent_run_transient_retry",
                    attempt=attempt,
                    error=last_reason,
                )
                if attempt == self._retries:
                    raise _TransientExhausted(
                        reason=last_reason, attempts=attempt + 1
                    ) from e
                self._sleep(self._backoff_seconds(attempt))
        # Unreachable — the loop either returns or raises above.
        raise _TransientExhausted(reason=last_reason, attempts=self._retries + 1)

    def _backoff_seconds(self, attempt: int) -> float:
        base = self._base_backoff * (2**attempt)
        # `random` is fine here — backoff jitter is not security-sensitive.
        return base + random.uniform(0, base)  # noqa: S311

    def _failure_result(
        self,
        *,
        agent_name: str,
        run_id: str,
        start: float,
        attempts: int,
        failure: AgentRunFailure,
    ) -> AgentRunResult:
        metrics = AgentMetrics(
            agent_name=agent_name,
            run_id=run_id,
            tier=None,
            model=None,
            prompt_tokens=0,
            cached_tokens=0,
            completion_tokens=0,
            estimated_cost_usd=Decimal("0"),
            cache_hit_ratio=0.0,
            latency_seconds=time.perf_counter() - start,
            attempts=attempts,
        )
        self._logger.warning(
            "agent_run_failed",
            agent=agent_name,
            run_id=run_id,
            code=failure.code,
            reason=failure.reason,
            attempts=attempts,
        )
        return AgentRunResult(
            output=None, failure=failure, metrics=metrics, eval_verdict=None
        )


# ---------------------------------------------------------------------------
# Internal signal type for retries exhausted
# ---------------------------------------------------------------------------


class _TransientExhausted(Exception):
    def __init__(self, *, reason: str, attempts: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.attempts = attempts


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _default_logger() -> structlog.stdlib.BoundLogger:
    if not structlog.is_configured():
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    return structlog.get_logger("core.agents.runner")


__all__ = [
    "AgentRunner",
    "BudgetTracker",
]


# A run_id helper for the CLI / callers who don't want to think about it.
def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"
