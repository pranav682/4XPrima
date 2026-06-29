"""Dev CLI: run strategy_lab_agent once against a saved MarketContextReport.

Feed it a MarketContextReport JSON (e.g. one saved by
``scripts/run_market_context.py`` into ``samples/``). The agent proposes
StrategyCandidates; this prints them readably and saves the full
StrategyProposal JSON to ``samples/`` (gitignored).

This is a PROPOSE-only step: nothing is backtested, optimized, executed, or
deployed. Issues one DEFAULT-tier (gpt-5.4) OpenAI call through the AgentRunner
+ evaluation gate.

Usage:
    python -m scripts.run_strategy_lab --context samples/ctx-XXXX.json
    python -m scripts.run_strategy_lab --context samples/ctx-XXXX.json \\
        --universe USDJPY EURUSD GBPUSD --n 3
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

from core.agents.evaluation import EvaluationGate
from core.agents.runner import AgentRunner, new_run_id
from core.agents.strategy_lab_agent import (
    DEFAULT_TIMEFRAMES,
    DEFAULT_UNIVERSE,
    StrategyLabAgent,
    StrategyLabRequest,
)
from core.agents.types import AgentBudget
from core.config import OpenAISettings
from core.llm_client import OpenAIProvider
from core.models import Granularity, MarketContextReport, StrategyProposal


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--context",
        type=Path,
        required=True,
        help="Path to a saved MarketContextReport JSON.",
    )
    p.add_argument(
        "--universe",
        nargs="+",
        default=list(DEFAULT_UNIVERSE),
        help=f"Allowed pairs (default: {' '.join(DEFAULT_UNIVERSE)}).",
    )
    p.add_argument(
        "--timeframes",
        nargs="+",
        default=[t.value for t in DEFAULT_TIMEFRAMES],
        choices=[g.value for g in Granularity],
        help="Allowed timeframes (default: H1 H4 D).",
    )
    p.add_argument("--n", type=int, default=3, help="Max candidates (default: 3).")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("samples"),
        help="Output directory for the saved proposal JSON.",
    )
    p.add_argument(
        "--budget-usd",
        type=str,
        default="1.00",
        help="Per-cycle cost cap in USD (default: 1.00).",
    )
    return p.parse_args()


def _summarise(proposal: StrategyProposal) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("STRATEGY LAB — proposed candidates (specs for the backtester, NOT trades)")
    lines.append("=" * 78)
    lines.append(f"run_id: {proposal.run_id}")
    lines.append(f"as_of:  {proposal.as_of.isoformat()}")
    lines.append(f"candidates: {len(proposal.candidates)}")
    for i, c in enumerate(proposal.candidates, start=1):
        lines.append("")
        lines.append(
            f"  {i}. {c.candidate_id}  [{c.archetype.value}]  {c.instrument} {c.timeframe.value}"
        )
        params = ", ".join(f"{p.name}={p.value}" for p in c.parameters)
        lines.append(f"     params:  {params}")
        ranges = ", ".join(f"{r.name}[{r.low},{r.high}]" for r in c.parameter_ranges)
        lines.append(f"     ranges:  {ranges}")
        lines.append(f"     why:     {c.rationale}")
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.context.exists():
        print(f"\nFAILED: context file not found: {args.context}", file=sys.stderr)
        return 1
    context = MarketContextReport(**json.loads(args.context.read_text()))

    run_id = new_run_id("lab")
    request = StrategyLabRequest(
        run_id=run_id,
        market_context=context,
        allowed_universe=tuple(p.upper() for p in args.universe),
        allowed_timeframes=tuple(Granularity(t) for t in args.timeframes),
        n_candidates=args.n,
    )

    provider = OpenAIProvider(OpenAISettings())  # type: ignore[call-arg]
    runner = AgentRunner(
        llm_provider=provider,
        evaluation_gate=EvaluationGate(),  # Tier-2 off by default
        budget=AgentBudget(max_cost_per_cycle_usd=Decimal(args.budget_usd)),
    )
    result = runner.run(StrategyLabAgent(), request, run_id=run_id)

    if not result.succeeded:
        print(f"\nFAILED: {result.failure}", file=sys.stderr)
        return 1

    proposal = result.output
    assert isinstance(proposal, StrategyProposal)
    print(_summarise(proposal))

    out_path = args.out / f"{run_id}.json"
    out_path.write_text(json.dumps(proposal.model_dump(mode="json"), indent=2, default=str))
    print(f"\nProposal saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
