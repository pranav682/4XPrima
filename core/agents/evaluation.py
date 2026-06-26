"""EvaluationGate — two-tier quality control on agent output.

**Important framing.** This module enforces *internal consistency* and
*spec conformance*. It does NOT verify predictive correctness — that's the
backtester's job, not another model's opinion. A passing Tier-2 verdict is
NOT a signal that the agent is right about the market; it's only a signal
that the output is coherent and respects the schema.

Two tiers:

- **Tier 1 (deterministic, free, ALWAYS on).** Per-agent structural and
  policy checks defined by :meth:`Agent.evaluations`. Failure → hard reject
  (the runner discards the output and returns
  ``AgentRunFailure(code="eval_rejected")``).
- **Tier 2 (LLM-as-judge, OPTIONAL, OFF by default).** Runs on the CHEAP
  tier. Reads the input snapshot + the output, scores coherence and
  spec-conformance, returns ``pass`` / ``flag`` / ``fail``. ``flag`` and
  ``fail`` are SOFT signals — logged, surfaced via the verdict, but NOT
  auto-rejecting. The orchestrator decides what to do.

Configurable Tier-2 modes:

- ``"off"``    : never invoked. No provider call. ``tier2_ran=False``.
- ``"sampled"``: invoked with probability ``tier2_sample_rate`` per run.
- ``"on"``     : invoked on every run.

See ``docs/llm-conventions.md`` for the rationale on why this is quality
control, not a predictive check.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

from core.agents.types import (
    Agent,
    AgentCall,
    EvalVerdict,
)
from core.llm_client import LLMProvider, ModelTier

# ---------------------------------------------------------------------------
# Judge schema
# ---------------------------------------------------------------------------


class JudgeVerdict(BaseModel):
    """Structured output the Tier-2 judge returns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: Literal["pass", "flag", "fail"]
    coherence_score: float = Field(ge=0.0, le=1.0)
    spec_conformance_score: float = Field(ge=0.0, le=1.0)
    reasons: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Judge prompt — kept as a stable constant so OpenAI auto-cache can hit it
# across many judge calls. (~1024 token threshold, see llm-conventions.md.)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are an evaluation judge for 4xPrima slow-loop agents. Your role is \
NARROW and you must respect it absolutely:

YOU VERIFY:
- internal consistency of the agent's output (claims don't contradict each \
other);
- conformance to the output schema's intent (e.g. neutral language, no \
trade calls, citations match the input);
- whether the output stays grounded in the input snapshot the agent was \
given.

YOU DO NOT VERIFY:
- whether the agent is right about the market;
- whether the regime / sentiment / surprise reads will be borne out;
- any forward-looking question.

Predictive correctness is the backtester's job — never another model's \
opinion. Saying "this looks accurate" or "the agent's bias seems right" is \
out of scope. Only judge consistency and spec-conformance.

INPUT FORMAT (the user message is a single JSON object):

{
  "input_snapshot": <agent-specific JSON describing what the agent was \
given>,
  "output":         <agent-specific JSON of the agent's structured output>
}

OUTPUT FORMAT (the structured response schema):

{
  "verdict":               "pass" | "flag" | "fail",
  "coherence_score":       0.0 .. 1.0,
  "spec_conformance_score":0.0 .. 1.0,
  "reasons":               [up to 3 short strings, ≤ 200 chars each]
}

SCORING RUBRIC:

- coherence_score:
  1.0 = every claim consistent; numbers and labels line up across fields.
  0.7 = minor inconsistencies (e.g. a regime confidence that doesn't match \
its rationale strength).
  0.4 = at least one direct contradiction between fields.
  0.0 = the output reads as multiple independent guesses.

- spec_conformance_score:
  1.0 = no trade calls, no invented data, language is neutral.
  0.7 = neutral but light citations / weak grounding.
  0.4 = leaks a "should" or "bias" word, or invents a number not in the \
snapshot.
  0.0 = recommends trades or fabricates events.

- verdict:
  pass: BOTH scores ≥ 0.7.
  flag: one or both scores in [0.4, 0.7).
  fail: either score < 0.4.

Reasons: list up to 3 short, specific reasons supporting the verdict. \
Avoid vague language. If everything looks fine, return one reason like \
"All claims trace to the snapshot; no schema drift."

Restate (this is the rule, not advice): you are an evaluation judge. You \
do not assess market-call quality. Predictive correctness is the \
backtester's job.\
"""


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


Tier2Mode = Literal["off", "sampled", "on"]


class EvaluationGate:
    """Two-tier evaluation gate.

    Tier-1 always runs (free, deterministic). Tier-2 runs per the mode:

    - ``"off"`` (default): no judge call ever happens.
    - ``"sampled"``: judge runs with probability ``tier2_sample_rate``.
    - ``"on"``: judge runs on every Tier-1-pass.

    The judge needs an :class:`LLMProvider` — if the mode is not
    ``"off"`` you MUST pass one; otherwise the constructor raises.
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider | None = None,
        tier2_mode: Tier2Mode = "off",
        tier2_sample_rate: float = 0.0,
        tier2_tier: ModelTier = ModelTier.CHEAP,
        rng: random.Random | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        if tier2_mode != "off" and llm_provider is None:
            raise ValueError(
                f"tier2_mode={tier2_mode!r} requires an llm_provider"
            )
        if not (0.0 <= tier2_sample_rate <= 1.0):
            raise ValueError("tier2_sample_rate must be in [0, 1]")
        self._llm = llm_provider
        self._tier2_mode = tier2_mode
        self._tier2_sample_rate = tier2_sample_rate
        self._tier2_tier = tier2_tier
        # Tier-2 sampling is not security-sensitive — `random.Random` is fine.
        self._rng = rng or random.Random()  # noqa: S311
        self._logger = (
            logger if logger is not None else _default_logger()
        ).bind(component="evaluation_gate", tier2_mode=tier2_mode)

    @property
    def tier2_mode(self) -> Tier2Mode:
        return self._tier2_mode

    # ---------------------------------------------------------- evaluate

    def evaluate(
        self,
        *,
        agent: Agent,
        call: AgentCall,
        output: BaseModel,
    ) -> EvalVerdict:
        # Tier 1 — always runs.
        tier1_failures: list[str] = []
        for check in agent.evaluations():
            try:
                result = check.check(output, call.input_snapshot)
            except Exception as e:
                tier1_failures.append(f"{check.name}: check raised {type(e).__name__}: {e}")
                continue
            if not result.passed:
                tier1_failures.append(f"{check.name}: {result.reason}")
        tier1_passed = not tier1_failures

        # Tier 2 — only on Tier-1 pass AND per mode.
        ran = False
        verdict: Literal["pass", "flag", "fail"] | None = None
        reasons: tuple[str, ...] = ()
        if tier1_passed and self._should_run_tier2():
            ran = True
            verdict, reasons = self._run_judge(agent=agent, call=call, output=output)

        return EvalVerdict(
            tier1_passed=tier1_passed,
            tier1_failures=tuple(tier1_failures),
            tier2_ran=ran,
            tier2_verdict=verdict,
            tier2_reasons=reasons,
        )

    # -------------------------------------------------------- internals

    def _should_run_tier2(self) -> bool:
        if self._tier2_mode == "off":
            return False
        if self._tier2_mode == "on":
            return True
        # sampled
        return self._rng.random() < self._tier2_sample_rate

    def _run_judge(
        self, *, agent: Agent, call: AgentCall, output: BaseModel
    ) -> tuple[Literal["pass", "flag", "fail"] | None, tuple[str, ...]]:
        # The judge needs a compact JSON view of the input and output.
        payload = {
            "input_snapshot": call.input_snapshot,
            "output": output.model_dump(mode="json"),
        }
        volatile = json.dumps(payload, sort_keys=True, default=str)
        try:
            assert self._llm is not None  # checked at construction
            parsed, _ = self._llm.generate_structured(
                agent_name=f"{agent.name}__judge",
                run_id=f"judge-{call.input_snapshot.get('run_id', '?')}",
                tier=self._tier2_tier,
                stable_system=_JUDGE_SYSTEM,
                volatile_user=volatile,
                output_model=JudgeVerdict,
                max_output_tokens=512,
            )
        except Exception as e:
            # Judge failures are SOFT — we treat as "flag" with the error so
            # the orchestrator notices but the parent agent's output is not
            # auto-rejected on judge unavailability.
            self._logger.warning("evaluation_judge_unavailable", error=repr(e))
            return ("flag", (f"judge unavailable: {type(e).__name__}: {e}",))
        return parsed.verdict, parsed.reasons


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
    return structlog.get_logger("core.agents.evaluation")


__all__ = [
    "EvaluationGate",
    "JudgeVerdict",
    "Tier2Mode",
]
