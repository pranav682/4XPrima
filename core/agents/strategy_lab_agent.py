"""strategy_lab_agent — the first 4xPrima agent that PROPOSES strategies.

Reads its spec: ``specs/agents/strategy_lab_agent.md``.

What this agent does:

1. Consumes a :class:`core.models.MarketContextReport` plus an allowed universe
   and timeframe set (passed in — it fetches NOTHING; that is
   ``market_context_agent``'s job, and it keeps the cache hot).
2. Proposes a small set (1-3) of :class:`core.models.StrategyCandidate`s as
   typed, BOUNDED specs the deterministic backtester can run verbatim. Each
   candidate picks a strategy archetype from the FIXED
   ``core.strategy.STRATEGY_REGISTRY`` and supplies concrete parameters plus
   bounded ``parameter_ranges`` (the optimizer's later sandbox).

What this agent does NOT do — by construction:

- It does not backtest, optimize, execute, deploy, or self-approve anything.
- It cannot invent a strategy type the engine can't construct: the archetype
  is restricted to the registry, and Tier-1 evaluation builds every candidate
  before the output is accepted.
- It cannot signal intent to trade live: the ``StrategyCandidate`` model
  scrubs execution/deployment language, and Tier-1 re-checks it.

Runs on the DEFAULT tier (``gpt-5.4``). The orchestrator may later promote it
to the HEAVY tier for explicit novel-design sessions; routine candidate
generation does not justify that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from core.agents.types import (
    Agent,
    AgentCall,
    CheckResult,
    StructuralCheck,
)
from core.llm_client import ModelTier
from core.models import (
    _EXECUTION_INTENT_PATTERNS,
    Granularity,
    MarketContextReport,
    StrategyProposal,
)
from core.strategy import (
    STRATEGY_REGISTRY,
    archetype_catalog,
    build_strategy,
    validate_candidate,
)

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

DEFAULT_UNIVERSE: tuple[str, ...] = ("USDJPY", "EURUSD", "GBPUSD", "USDCAD", "AUDUSD")
DEFAULT_TIMEFRAMES: tuple[Granularity, ...] = (
    Granularity.H1,
    Granularity.H4,
    Granularity.D,
)


@dataclass(frozen=True, slots=True)
class StrategyLabRequest:
    """One run of the strategy lab.

    ``allowed_universe`` and ``allowed_timeframes`` are passed in (not
    hardcoded) — typically the screened universe from
    :mod:`core.analysis.pair_screener`.
    """

    run_id: str
    market_context: MarketContextReport
    allowed_universe: tuple[str, ...] = DEFAULT_UNIVERSE
    allowed_timeframes: tuple[Granularity, ...] = DEFAULT_TIMEFRAMES
    n_candidates: int = 3  # the agent proposes 1..n, not a flood
    tier: ModelTier = ModelTier.DEFAULT
    archetypes: tuple[str, ...] = field(
        default_factory=lambda: tuple(a.value for a in STRATEGY_REGISTRY)
    )


# ---------------------------------------------------------------------------
# Stable system prefix — built once, byte-stable across calls so OpenAI
# auto-caches it. (Instructions + registry catalog + schema + rules.)
# ---------------------------------------------------------------------------

_INSTRUCTIONS = """\
You are strategy_lab_agent for an algorithmic FX trading system. You PROPOSE \
candidate trading-strategy specifications for a deterministic backtester to \
judge. You do NOT backtest, optimize, execute, deploy, or approve anything — \
you only write bounded specs. Profitability is decided later by the \
backtester, an out-of-sample split, and an adversarial critic, never by you.

Read the MarketContextReport and the allowed universe/timeframes in the user \
message, then emit a StrategyProposal containing a SMALL set (1 to the \
requested n_candidates, never more) of StrategyCandidates that are consistent \
with the regime tags. Tie each candidate's rationale to specific evidence in \
the MarketContextReport (e.g. a trending_up regime on USDJPY -> a \
trend-following crossover on USDJPY).

HARD RULES (violating any of these gets the whole proposal hard-rejected):

- archetype: pick ONLY from the STRATEGY ARCHETYPE CATALOG below. You may not \
invent a strategy type. If no archetype fits the context, propose fewer \
candidates (even zero) rather than forcing a bad fit.
- parameters: supply EVERY parameter the chosen archetype lists, each a \
concrete value inside the archetype's stated limits. Integer parameters must \
be whole numbers. Respect archetype-specific constraints (e.g. ma_crossover \
requires fast_period < slow_period).
- parameter_ranges: for EVERY parameter, supply a bounded [low, high] range \
the later optimizer may explore. low < high, the range must sit inside the \
archetype limits, and the concrete value must sit inside its own range. \
Integer parameters need whole-number bounds.
- instrument: ONLY a pair from allowed_universe in the user message.
- timeframe: ONLY a granularity from allowed_timeframes in the user message.
- rationale: <= 500 characters, grounded in the MarketContextReport. NO \
execution, deployment, or live-trading language anywhere — no "go live", \
"deploy", "execute the trade", "place the order", "real money". You describe \
a spec; you never signal that it should be traded.
- candidate_id: a short deterministic-looking id unique within the proposal.
- copy run_id from the user message into the proposal and every candidate.\
"""

_SCHEMA_AND_RULES = """\
OUTPUT SCHEMA (StrategyProposal):

- run_id (string): copy from the user message.
- as_of (ISO-8601 UTC datetime): copy the MarketContextReport's as_of.
- schema_version: "1.0".
- candidates: list (length 1..n_candidates) of StrategyCandidate:
  - candidate_id (string): unique within the proposal.
  - run_id (string): copy from the user message.
  - archetype (enum): one of the catalog archetype keys.
  - instrument (string): a six-letter pair from allowed_universe (e.g. USDJPY).
  - timeframe (enum): a granularity from allowed_timeframes (e.g. H1).
  - parameters: list of {name, value} covering every archetype parameter.
  - parameter_ranges: list of {name, low, high} covering every parameter.
  - rationale (string <= 500 chars): why this spec, tied to the context.

OUTPUT EXAMPLE (shape, NOT content — copy structure, not values):

{
  "run_id": "<from user message>",
  "as_of": "<from MarketContextReport, ISO-8601 Z>",
  "schema_version": "1.0",
  "candidates": [
    {
      "candidate_id": "mac-usdjpy-1",
      "run_id": "<from user message>",
      "archetype": "ma_crossover",
      "instrument": "USDJPY",
      "timeframe": "H1",
      "parameters": [
        {"name": "fast_period", "value": "12"},
        {"name": "slow_period", "value": "48"},
        {"name": "size", "value": "1000"},
        {"name": "stop_distance_pips", "value": "70"}
      ],
      "parameter_ranges": [
        {"name": "fast_period", "low": "5", "high": "30"},
        {"name": "slow_period", "low": "30", "high": "120"},
        {"name": "size", "low": "500", "high": "2000"},
        {"name": "stop_distance_pips", "low": "40", "high": "120"}
      ],
      "rationale": "Context tags USDJPY trending_up with elevated vol; a \
fast/slow crossover rides continuation. Stop in pips (70) suits a JPY pair."
    }
  ]
}

Restate (this is the rule, not advice): you propose bounded specs only. You do \
not run them, optimize them, or signal that they should be traded. If you feel \
an urge to say "deploy" or "go live", you are in the wrong agent.\
"""

_STABLE_SYSTEM = (
    _INSTRUCTIONS
    + "\n\nSTRATEGY ARCHETYPE CATALOG (propose ONLY from these):\n"
    + archetype_catalog()
    + "\n\n"
    + _SCHEMA_AND_RULES
)


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class StrategyLabAgent:
    """Prepares one strategy-proposal call. Implements the
    :class:`core.agents.types.Agent` Protocol. Does NOT call the LLM — the
    :class:`core.agents.runner.AgentRunner` does, with the eval gate around it.
    """

    name: str = "strategy_lab_agent"

    def __init__(self, *, tier: ModelTier = ModelTier.DEFAULT) -> None:
        self._tier = tier

    # --------------------------------------------------------- Protocol

    def prepare_call(self, inputs: StrategyLabRequest) -> AgentCall:
        context_json = inputs.market_context.model_dump(mode="json")
        snapshot: dict[str, Any] = {
            "run_id": inputs.run_id,
            "allowed_universe": [p.upper() for p in inputs.allowed_universe],
            "allowed_timeframes": [t.value for t in inputs.allowed_timeframes],
            "max_candidates": inputs.n_candidates,
            "archetypes": [a.value for a in STRATEGY_REGISTRY],
            "market_context": context_json,
        }
        volatile_user = (
            "Propose strategy candidates for the deterministic backtester. "
            f"Return at most {inputs.n_candidates} candidate(s). Use ONLY the "
            "archetypes in the catalog, instruments in allowed_universe, and "
            "timeframes in allowed_timeframes.\n\n"
            f"run_id: {inputs.run_id}\n"
            f"allowed_universe: {snapshot['allowed_universe']}\n"
            f"allowed_timeframes: {snapshot['allowed_timeframes']}\n\n"
            f"MARKET_CONTEXT:\n{_pretty(context_json)}"
        )
        # The tier on the inputs wins over the agent default (lets the
        # orchestrator promote to HEAVY for a novel-design session).
        tier = inputs.tier if inputs.tier is not None else self._tier
        return AgentCall(
            stable_system=_STABLE_SYSTEM,
            volatile_user=volatile_user,
            tier=tier,
            output_model=StrategyProposal,
            max_output_tokens=4096,
            input_snapshot=snapshot,
        )

    def evaluations(self) -> tuple[StructuralCheck, ...]:
        return _TIER1_CHECKS


# ---------------------------------------------------------------------------
# Tier-1 deterministic checks (always on, hard-reject)
# ---------------------------------------------------------------------------


def _as_proposal(output: Any) -> StrategyProposal:
    if not isinstance(output, StrategyProposal):
        raise TypeError(f"expected StrategyProposal, got {type(output).__name__}")
    return output


def _check_run_id_preserved(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    proposal = _as_proposal(output)
    expected = snapshot.get("run_id")
    if expected is None:
        return CheckResult(True)
    if proposal.run_id != expected:
        return CheckResult(False, f"proposal run_id {proposal.run_id!r} != input {expected!r}")
    for c in proposal.candidates:
        if c.run_id != expected:
            return CheckResult(
                False, f"candidate {c.candidate_id!r} run_id {c.run_id!r} != {expected!r}"
            )
    return CheckResult(True)


def _check_candidate_count(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    proposal = _as_proposal(output)
    n = len(proposal.candidates)
    max_n = int(snapshot.get("max_candidates", 3))
    if n < 1:
        return CheckResult(False, "proposal has no candidates")
    if n > max_n:
        return CheckResult(False, f"{n} candidates > requested max {max_n}")
    return CheckResult(True)


def _check_archetypes_in_registry(output: Any, _snapshot: dict[str, Any]) -> CheckResult:
    proposal = _as_proposal(output)
    for c in proposal.candidates:
        if c.archetype not in STRATEGY_REGISTRY:
            return CheckResult(
                False, f"candidate {c.candidate_id!r} archetype {c.archetype!r} not in registry"
            )
    return CheckResult(True)


def _check_pairs_in_universe(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    proposal = _as_proposal(output)
    universe = {p.upper() for p in snapshot.get("allowed_universe", [])}
    if not universe:
        return CheckResult(True)  # nothing to check against
    for c in proposal.candidates:
        if c.instrument not in universe:
            return CheckResult(
                False,
                f"candidate {c.candidate_id!r} instrument {c.instrument!r} not in universe",
            )
    return CheckResult(True)


def _check_timeframes_allowed(output: Any, snapshot: dict[str, Any]) -> CheckResult:
    proposal = _as_proposal(output)
    allowed = set(snapshot.get("allowed_timeframes", []))
    if not allowed:
        return CheckResult(True)
    for c in proposal.candidates:
        if c.timeframe.value not in allowed:
            return CheckResult(
                False,
                f"candidate {c.candidate_id!r} timeframe {c.timeframe.value!r} not allowed",
            )
    return CheckResult(True)


def _check_params_and_ranges_sane(output: Any, _snapshot: dict[str, Any]) -> CheckResult:
    proposal = _as_proposal(output)
    for c in proposal.candidates:
        errors = validate_candidate(c)
        if errors:
            return CheckResult(False, f"candidate {c.candidate_id!r}: " + "; ".join(errors))
    return CheckResult(True)


def _check_candidates_constructible(output: Any, _snapshot: dict[str, Any]) -> CheckResult:
    proposal = _as_proposal(output)
    for c in proposal.candidates:
        try:
            build_strategy(c)
        except Exception as e:  # any failure means not constructible
            return CheckResult(
                False,
                f"candidate {c.candidate_id!r} not constructible: {type(e).__name__}: {e}",
            )
    return CheckResult(True)


def _check_no_execution_intent(output: Any, _snapshot: dict[str, Any]) -> CheckResult:
    """Re-scan candidate free-text for execution/deployment/live-trading intent.

    The pydantic validator already rejects this at construction, but a
    ``model_construct(...)`` path bypasses validation — this catches that
    escape too. Defence-in-depth, mirroring market_context's no-trade-call check.
    """
    proposal = _as_proposal(output)
    for c in proposal.candidates:
        for pattern in _EXECUTION_INTENT_PATTERNS:
            m = pattern.search(c.rationale)
            if m:
                return CheckResult(
                    False,
                    f"candidate {c.candidate_id!r} rationale has execution intent: {m.group()!r}",
                )
    return CheckResult(True)


_TIER1_CHECKS: tuple[StructuralCheck, ...] = (
    StructuralCheck(name="run_id_preserved", check=_check_run_id_preserved),
    StructuralCheck(name="candidate_count", check=_check_candidate_count),
    StructuralCheck(name="archetypes_in_registry", check=_check_archetypes_in_registry),
    StructuralCheck(name="pairs_in_universe", check=_check_pairs_in_universe),
    StructuralCheck(name="timeframes_allowed", check=_check_timeframes_allowed),
    StructuralCheck(name="params_and_ranges_sane", check=_check_params_and_ranges_sane),
    StructuralCheck(name="candidates_constructible", check=_check_candidates_constructible),
    StructuralCheck(name="no_execution_intent", check=_check_no_execution_intent),
)


# Module-level sanity: this agent must implement the Agent Protocol.
_: Agent = StrategyLabAgent.__new__(StrategyLabAgent)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pretty(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ": "), default=str)


__all__ = [
    "DEFAULT_TIMEFRAMES",
    "DEFAULT_UNIVERSE",
    "StrategyLabAgent",
    "StrategyLabRequest",
]
