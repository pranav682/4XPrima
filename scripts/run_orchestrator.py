"""Dev CLI: run ONE slow-loop cycle through the deterministic orchestrator.

market_context -> strategy_lab -> backtest_agent (in-sample) -> critic (opens
OOS). Tracks champion/challenger state in a persistent registry, enforces a
per-cycle USD budget, and routes survivors to the human approval queue. Prints a
readable cycle summary and saves the CycleResult to ``samples/`` (gitignored);
the registry + queue persist under ``--state-dir`` (default ``data/orchestration``).

This is ONE pass — there is no scheduler/daemon here, by design. The
orchestrator makes no LLM call itself; each worker agent does, through the
runner. Nothing is promoted or traded: survivors land in the approval queue for
a human, which is the terminus.

Usage:
    python -m scripts.run_orchestrator
    python -m scripts.run_orchestrator --universe USDJPY EURUSD --budget-usd 1.50
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

from core.agents.backtest_agent import BacktestAgent
from core.agents.backtest_harness import BacktestRunConfig
from core.agents.critic_agent import CriticAgent
from core.agents.evaluation import EvaluationGate
from core.agents.market_context_agent import MarketContextAgent
from core.agents.runner import AgentRunner, new_run_id
from core.agents.strategy_lab_agent import StrategyLabAgent
from core.agents.types import AgentBudget
from core.config import FredSettings, OandaSettings, OpenAISettings
from core.context_data import (
    ForexFactoryCalendarProvider,
    FredMacroDataProvider,
    GdeltNewsProvider,
)
from core.llm_client import OpenAIProvider
from core.market_data import OandaPriceProvider
from core.orchestration import (
    ApprovalQueue,
    BacktestArtifactStore,
    ChampionChallengerRegistry,
    CycleResult,
    Orchestrator,
    OrchestratorConfig,
)

DEFAULT_UNIVERSE = ("USDJPY", "EURUSD", "GBPUSD", "USDCAD", "AUDUSD")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", nargs="+", default=list(DEFAULT_UNIVERSE), help="Screened pairs.")
    p.add_argument("--budget-usd", type=str, default="2.00", help="Per-cycle USD cap.")
    p.add_argument("--per-call-usd", type=str, default="0.50", help="Per-call budgeted ceiling.")
    p.add_argument("--n-candidates", type=int, default=3, help="Candidates strategy_lab proposes.")
    p.add_argument("--lookback", type=int, default=1500, help="Bars per candidate.")
    p.add_argument("--oos-fraction", type=float, default=0.2, help="Held-out OOS tail.")
    p.add_argument(
        "--state-dir", type=Path, default=Path("data/orchestration"), help="Registry + queue dir."
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/orchestration/cycles"),
        help="Cycle summary dir (the dashboard API reads cycles from here).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)

    universe = tuple(p.upper() for p in args.universe)
    cycle_id = new_run_id("cycle")
    config = OrchestratorConfig(
        max_cost_per_cycle_usd=Decimal(args.budget_usd),
        per_call_cost_ceiling_usd=Decimal(args.per_call_usd),
        n_candidates=args.n_candidates,
        backtest_config=BacktestRunConfig(
            lookback_count=args.lookback, oos_fraction=args.oos_fraction
        ),
    )

    registry = ChampionChallengerRegistry(args.state_dir / "registry.json")
    queue = ApprovalQueue(args.state_dir / "approval_queue.json")
    artifacts = BacktestArtifactStore(args.state_dir / "backtests")

    fred = FredSettings()  # type: ignore[call-arg]
    openai = OpenAISettings()  # type: ignore[call-arg]
    oanda = OandaSettings()  # type: ignore[call-arg]

    # The orchestrator enforces the per-cycle budget; the runner's own cost cap
    # is disabled so the cycle cap is the single authority.
    runner = AgentRunner(
        llm_provider=OpenAIProvider(openai),
        evaluation_gate=EvaluationGate(),
        budget=AgentBudget(max_cost_per_cycle_usd=None),
    )

    with (
        ForexFactoryCalendarProvider() as cal,
        FredMacroDataProvider(fred) as macro,
        GdeltNewsProvider() as news,
        OandaPriceProvider(oanda) as candles,
    ):
        orchestrator = Orchestrator(
            market_context_agent=MarketContextAgent(
                calendar_provider=cal, macro_provider=macro, news_provider=news
            ),
            strategy_lab_agent=StrategyLabAgent(),
            backtest_agent=BacktestAgent(),
            critic_agent=CriticAgent(),
            runner=runner,
            candle_provider=candles,
            registry=registry,
            approval_queue=queue,
            artifact_store=artifacts,
            config=config,
        )
        result = orchestrator.run_cycle(universe, cycle_id=cycle_id)

    print(_summarise(result, queue))
    out_path = args.out / f"{cycle_id}.json"
    out_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
    print(f"\nCycle summary saved to {out_path}")
    print(f"Registry + approval queue persisted under {args.state_dir}/")
    return 0 if result.outcome.value == "completed" else 1


def _summarise(r: CycleResult, queue: ApprovalQueue) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("ORCHESTRATOR CYCLE (deterministic; routes to human approval, never promotes)")
    lines.append("=" * 80)
    lines.append(f"cycle_id: {r.cycle_id}   outcome: {r.outcome.value.upper()}")
    lines.append(
        f"proposed={r.candidates_proposed}  killed={r.candidates_killed}  "
        f"queued_for_approval={r.candidates_queued}"
    )
    lines.append(f"total_cost=${r.total_cost_usd}   duration={r.duration_seconds:.1f}s")
    lines.append(f"stage_costs: {r.stage_costs_usd}")
    if r.abort_reason:
        lines.append(f"ABORTED: {r.abort_reason}")
    pending = queue.pending()
    lines.append("")
    lines.append(
        f"APPROVAL QUEUE — {len(pending)} pending (awaiting a HUMAN decision; not approved):"
    )
    for e in pending:
        oos = e.out_of_sample_evidence
        oos_str = "no OOS" if oos is None else f"OOS sharpe={oos.metrics.sharpe_ratio:.2f}"
        lines.append(
            f"  [{e.identity}] {e.candidate.instrument} {e.candidate.timeframe.value} "
            f"{e.candidate.archetype.value} — critic={e.critic_verdict.verdict.value} ({oos_str})"
        )
    lines.append("=" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
