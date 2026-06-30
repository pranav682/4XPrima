"""Orchestrator — the DETERMINISTIC coordinator of the slow loop.

It sequences the four worker agents into one cycle —
``market_context → strategy_lab → backtest_agent → critic`` — passing each
typed output to the next, tracks champion/challenger state in the registry,
enforces a per-cycle USD budget, and routes survivors to the human approval
queue. **The orchestrator itself makes NO LLM call**: the workers each make
their own calls through :class:`core.agents.runner.AgentRunner`; sequencing and
state here are a deterministic state machine, not a model. (The LLM cycle
summarisation the spec imagines belongs to ``reporting_agent``, deferred.)

TWO HARD WALLS (structural):

1. **It cannot promote or trade.** Surviving the critic routes a candidate to
   the approval QUEUE — an entry awaiting a human decision — and that is the
   terminus. There is no method/path here that promotes a challenger to
   champion, sets a strategy live, or flips the paper/live flag. Promotion is a
   human-only Stage-4 action (not built). The registry's ``set_state`` is typed
   to a writable subset that excludes the risk-authorizing states.
2. **It cannot touch sacred config.** It READS ``RiskConfig`` (inside the
   injected ``BacktestRunConfig``) to pass into backtests, but never mutates
   ``RiskConfig``, trips/resets the kill switch, or changes the paper/live flag.

One pass only: :meth:`Orchestrator.run_cycle`. No scheduler / daemon / loop —
running on a schedule is a deliberate later decision.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import TypeVar, cast

import structlog
from pydantic import BaseModel, ConfigDict

from core.agents.backtest_agent import BacktestAgentRequest
from core.agents.backtest_harness import (
    BacktestRunConfig,
    HarnessError,
    run_candidate_artifacts,
    run_proposal,
    run_robustness,
)
from core.agents.critic_agent import CriticRequest
from core.agents.market_context_agent import MarketContextRequest
from core.agents.runner import AgentRunner
from core.agents.strategy_lab_agent import DEFAULT_TIMEFRAMES, StrategyLabRequest
from core.agents.types import Agent, AgentBudgetExceeded
from core.market_data import CandleProvider
from core.models import (
    BacktestEvidence,
    BacktestTriage,
    BacktestVerdictSet,
    CriticVerdictKind,
    CriticVerdictSet,
    Granularity,
    MarketContextReport,
    RobustnessEvidence,
    StrategyCandidate,
    StrategyProposal,
)
from core.orchestration.approval_queue import ApprovalQueue
from core.orchestration.artifact_store import BacktestArtifactStore
from core.orchestration.registry import (
    ChampionChallengerRegistry,
    RegistryState,
)

# ---------------------------------------------------------------------------
# Config + result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    """Per-cycle knobs (deterministic; no model tier — there is no LLM here)."""

    max_cost_per_cycle_usd: Decimal = Decimal("5.00")
    # The budgeted ceiling for any single worker call. Before each call the
    # orchestrator checks running_cost + this <= the cycle cap; if not, it aborts
    # BEFORE making the call (the cost-side kill switch).
    per_call_cost_ceiling_usd: Decimal = Decimal("1.00")
    n_candidates: int = 3
    allowed_timeframes: tuple[Granularity, ...] = DEFAULT_TIMEFRAMES
    backtest_config: BacktestRunConfig = field(default_factory=BacktestRunConfig)
    upcoming_hours: int = 168
    recent_hours: int = 24


class CycleOutcome(StrEnum):
    COMPLETED = "completed"
    ABORTED_BUDGET = "aborted_budget"
    ABORTED_FAILURE = "aborted_failure"


_Output = TypeVar("_Output", bound=BaseModel)


class CycleResult(BaseModel):
    """Structured record of one cycle. Serialisable for the dev CLI / audit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cycle_id: str
    outcome: CycleOutcome
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    total_cost_usd: Decimal
    stage_costs_usd: dict[str, str]
    candidates_proposed: int
    candidates_killed: int
    candidates_queued: int
    queued_identities: tuple[str, ...]
    abort_reason: str | None = None


# ---------------------------------------------------------------------------
# Internal control-flow signals
# ---------------------------------------------------------------------------


class _BudgetAbort(Exception):
    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


class _WorkerFailure(Exception):
    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Run one slow-loop cycle. All collaborators are injected so the cycle is
    fully testable with mocks — the orchestrator itself is pure coordination."""

    def __init__(
        self,
        *,
        market_context_agent: Agent,
        strategy_lab_agent: Agent,
        backtest_agent: Agent,
        critic_agent: Agent,
        runner: AgentRunner,
        candle_provider: CandleProvider,
        registry: ChampionChallengerRegistry,
        approval_queue: ApprovalQueue,
        artifact_store: BacktestArtifactStore | None = None,
        config: OrchestratorConfig | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        self._market_context_agent = market_context_agent
        self._strategy_lab_agent = strategy_lab_agent
        self._backtest_agent = backtest_agent
        self._critic_agent = critic_agent
        self._runner = runner
        self._candle_provider = candle_provider
        self._registry = registry
        self._queue = approval_queue
        # Optional: when provided, the rich per-bar equity-curve artifacts for
        # critiqued candidates are persisted here for the read-only dashboard.
        # Does not affect routing, evidence, or the LLM prompts.
        self._artifact_store = artifact_store
        self._config = config or OrchestratorConfig()
        self._logger = (logger if logger is not None else _default_logger()).bind(
            component="orchestrator"
        )
        # Per-cycle accumulators (reset at the top of each run_cycle).
        self._running_cost: Decimal = Decimal("0")
        self._stage_costs: dict[str, str] = {}

    # ----------------------------------------------------------------- public

    def run_cycle(
        self, universe: tuple[str, ...], *, cycle_id: str, now: datetime | None = None
    ) -> CycleResult:
        """Run ONE full pass over ``universe``. Deterministic sequencing;
        fail-closed; never promotes; respects the per-cycle budget."""
        started_at = now or datetime.now(UTC)
        start_perf = time.perf_counter()
        self._running_cost = Decimal("0")
        self._stage_costs = {}
        log = self._logger.bind(cycle_id=cycle_id, universe=list(universe))

        proposed = 0
        killed = 0
        queued: list[str] = []
        outcome = CycleOutcome.COMPLETED
        abort_reason: str | None = None

        try:
            proposed, killed, queued = self._run_pipeline(universe, cycle_id, started_at, log)
        except _BudgetAbort as e:
            outcome = CycleOutcome.ABORTED_BUDGET
            abort_reason = (
                f"per-cycle budget ${self._config.max_cost_per_cycle_usd} would be "
                f"exceeded before stage {e.stage!r} (running ${self._running_cost})"
            )
            log.warning("orchestrator_cycle_aborted_budget", stage=e.stage)
        except _WorkerFailure as e:
            outcome = CycleOutcome.ABORTED_FAILURE
            abort_reason = f"stage {e.stage!r} failed: {e.reason}"
            log.warning("orchestrator_cycle_aborted_failure", stage=e.stage, reason=e.reason)
        except AgentBudgetExceeded as e:
            outcome = CycleOutcome.ABORTED_BUDGET
            abort_reason = f"agent budget exceeded: {e}"
            log.warning("orchestrator_cycle_aborted_agent_budget", error=str(e))
        except HarnessError as e:
            outcome = CycleOutcome.ABORTED_FAILURE
            abort_reason = f"deterministic harness failure: {e}"
            log.warning("orchestrator_cycle_aborted_harness", error=str(e))

        ended_at = datetime.now(UTC)
        result = CycleResult(
            cycle_id=cycle_id,
            outcome=outcome,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=time.perf_counter() - start_perf,
            total_cost_usd=self._running_cost,
            stage_costs_usd=dict(self._stage_costs),
            candidates_proposed=proposed,
            candidates_killed=killed,
            candidates_queued=len(queued),
            queued_identities=tuple(queued),
            abort_reason=abort_reason,
        )
        log.info(
            "orchestrator_cycle",
            outcome=outcome.value,
            total_cost_usd=str(result.total_cost_usd),
            stage_costs=result.stage_costs_usd,
            candidates_proposed=proposed,
            candidates_killed=killed,
            candidates_queued=len(queued),
            duration_seconds=f"{result.duration_seconds:.3f}",
            abort_reason=abort_reason,
        )
        return result

    # --------------------------------------------------------------- pipeline

    def _run_pipeline(
        self,
        universe: tuple[str, ...],
        cycle_id: str,
        ts: datetime,
        log: structlog.stdlib.BoundLogger,
    ) -> tuple[int, int, list[str]]:
        killed = 0
        queued: list[str] = []

        # 1. Market context (one call).
        report = self._run_agent(
            self._market_context_agent,
            MarketContextRequest(
                run_id=cycle_id,
                as_of=ts,
                watchlist=universe,
                upcoming_hours=self._config.upcoming_hours,
                recent_hours=self._config.recent_hours,
            ),
            run_id=f"{cycle_id}-ctx",
            stage="market_context",
            expected=MarketContextReport,
        )

        # 2. Strategy lab proposes candidates from the context.
        proposal = self._run_agent(
            self._strategy_lab_agent,
            StrategyLabRequest(
                run_id=cycle_id,
                market_context=report,
                allowed_universe=universe,
                allowed_timeframes=self._config.allowed_timeframes,
                n_candidates=self._config.n_candidates,
            ),
            run_id=f"{cycle_id}-lab",
            stage="strategy_lab",
            expected=StrategyProposal,
        )
        candidates: dict[str, StrategyCandidate] = {c.candidate_id: c for c in proposal.candidates}
        identity: dict[str, str] = {
            cid: self._registry.upsert_proposed(c, run_id=cycle_id, now=ts)
            for cid, c in candidates.items()
        }
        if not candidates:
            return 0, 0, []

        # 3. Deterministic in-sample backtests + interpretation.
        evidence, skipped = run_proposal(
            proposal, candle_provider=self._candle_provider, config=self._config.backtest_config
        )
        for cid, reason in skipped:
            log.info("orchestrator_candidate_skipped", candidate_id=cid, reason=reason)
        ev_by_id: dict[str, BacktestEvidence] = {e.candidate_id: e for e in evidence}
        for e in evidence:
            self._registry.record_backtest(identity[e.candidate_id], e, now=ts)
        if not evidence:
            return len(candidates), killed, queued

        bt_verdicts = self._run_agent(
            self._backtest_agent,
            BacktestAgentRequest(run_id=cycle_id, proposal=proposal, evidence=evidence),
            run_id=f"{cycle_id}-bt",
            stage="backtest_agent",
            expected=BacktestVerdictSet,
        )
        triage = {v.candidate_id: v.triage for v in bt_verdicts.verdicts}

        # Candidates the backtest rejected are killed here and not stressed.
        survivors: list[StrategyCandidate] = []
        for cid, cand in candidates.items():
            if cid not in ev_by_id:
                continue
            if triage.get(cid) == BacktestTriage.REJECT:
                self._registry.set_state(identity[cid], RegistryState.KILLED, now=ts)
                killed += 1
            else:
                survivors.append(cand)
        if not survivors:
            return len(candidates), killed, queued

        # 4. Robustness (deterministic, opens the OOS holdout) + critic.
        robustness: list[RobustnessEvidence] = []
        rob_by_id: dict[str, RobustnessEvidence] = {}
        for cand in survivors:
            try:
                rb = run_robustness(
                    cand, candle_provider=self._candle_provider, config=self._config.backtest_config
                )
            except HarnessError as e:
                log.info(
                    "orchestrator_robustness_skipped",
                    candidate_id=cand.candidate_id,
                    reason=str(e),
                )
                continue
            robustness.append(rb)
            rob_by_id[cand.candidate_id] = rb
            self._persist_artifacts(cand, log)
        if not robustness:
            return len(candidates), killed, queued

        critic_verdicts = self._run_agent(
            self._critic_agent,
            CriticRequest(
                run_id=cycle_id, robustness=tuple(robustness), backtest_verdicts=bt_verdicts
            ),
            run_id=f"{cycle_id}-crit",
            stage="critic",
            expected=CriticVerdictSet,
        )

        # 5. Route verdicts. KILL → recorded, NOT queued. survive_for_now →
        #    recorded + appended to the human approval queue (the terminus).
        for v in critic_verdicts.verdicts:
            ident = identity.get(v.candidate_id)
            verdict_candidate = candidates.get(v.candidate_id)
            if ident is None or verdict_candidate is None:
                continue
            verdict_rb = rob_by_id.get(v.candidate_id)
            oos = verdict_rb.out_of_sample if verdict_rb is not None else None
            if v.verdict == CriticVerdictKind.KILL:
                self._registry.record_critic(
                    ident, v, state=RegistryState.KILLED, out_of_sample_evidence=oos, now=ts
                )
                killed += 1
            else:  # survive_for_now — never an approval, only a queued pending entry
                self._registry.record_critic(
                    ident,
                    v,
                    state=RegistryState.SURVIVED_FOR_NOW,
                    out_of_sample_evidence=oos,
                    now=ts,
                )
                self._queue.append(
                    cycle_id=cycle_id,
                    identity=ident,
                    candidate=verdict_candidate,
                    in_sample_evidence=ev_by_id[v.candidate_id],
                    critic_verdict=v,
                    out_of_sample_evidence=oos,
                    now=ts,
                )
                self._registry.set_state(ident, RegistryState.QUEUED_FOR_APPROVAL, now=ts)
                queued.append(ident)

        return len(candidates), killed, queued

    # --------------------------------------------------------------- helpers

    def _persist_artifacts(
        self, candidate: StrategyCandidate, log: structlog.stdlib.BoundLogger
    ) -> None:
        """Best-effort: persist the candidate's IS + OOS equity-curve artifacts
        for the dashboard. Never affects routing, evidence, or verdicts; a
        failure here only means the curve won't be browsable."""
        if self._artifact_store is None:
            return
        try:
            artifacts = run_candidate_artifacts(
                candidate,
                candle_provider=self._candle_provider,
                config=self._config.backtest_config,
            )
        except HarnessError as e:
            log.info(
                "orchestrator_artifact_skipped", candidate_id=candidate.candidate_id, reason=str(e)
            )
            return
        for artifact in artifacts:
            self._artifact_store.save(artifact)

    def _run_agent(
        self,
        agent: Agent,
        request: object,
        *,
        run_id: str,
        stage: str,
        expected: type[_Output],
    ) -> _Output:
        """Budget pre-check → run the agent → account cost → unwrap output.

        Raises :class:`_BudgetAbort` BEFORE the call if running cost + the
        per-call ceiling would breach the cycle cap, or :class:`_WorkerFailure`
        if the agent run did not succeed (fail-closed)."""
        if (
            self._running_cost + self._config.per_call_cost_ceiling_usd
            > self._config.max_cost_per_cycle_usd
        ):
            raise _BudgetAbort(stage)
        result = self._runner.run(agent, request, run_id=run_id)
        self._running_cost += result.metrics.estimated_cost_usd
        self._stage_costs[stage] = str(result.metrics.estimated_cost_usd)
        if not result.succeeded or result.output is None:
            reason = result.failure.reason if result.failure is not None else "no output"
            raise _WorkerFailure(stage, reason)
        if not isinstance(result.output, expected):
            raise _WorkerFailure(
                stage, f"expected {expected.__name__}, got {type(result.output).__name__}"
            )
        return result.output


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
    return cast("structlog.stdlib.BoundLogger", structlog.get_logger("core.orchestration"))


__all__ = [
    "CycleOutcome",
    "CycleResult",
    "Orchestrator",
    "OrchestratorConfig",
]
