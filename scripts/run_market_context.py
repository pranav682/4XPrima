"""Dev CLI: run market_context_agent once, live, against the real providers.

Reads OPENAI_API_KEY + FRED_API_KEY from `.env`. Calls Forex Factory, FRED,
GDELT for the brief, then issues one DEFAULT-tier OpenAI call through the
AgentRunner. Prints a readable summary and saves the full report JSON to
`samples/` (gitignored).

Usage:
    python -m scripts.run_market_context --watchlist EURUSD USDJPY GBPUSD

By default the watchlist is ``EURUSD USDJPY``. Tier-2 LLM-as-judge is OFF
by default — pass ``--tier2 on`` to enable, or ``--tier2 sampled --sample 0.5``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from core.agents.evaluation import EvaluationGate
from core.agents.market_context_agent import (
    MarketContextAgent,
    MarketContextRequest,
)
from core.agents.runner import AgentRunner, new_run_id
from core.agents.types import AgentBudget
from core.config import FredSettings, OpenAISettings
from core.context_data import (
    ForexFactoryCalendarProvider,
    FredMacroDataProvider,
    GdeltNewsProvider,
)
from core.llm_client import OpenAIProvider


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--watchlist",
        nargs="+",
        default=["EURUSD", "USDJPY"],
        help="Pairs to focus on (default: EURUSD USDJPY).",
    )
    p.add_argument(
        "--upcoming-hours",
        type=int,
        default=72,
        help="How far ahead to scan the economic calendar (default: 72).",
    )
    p.add_argument(
        "--recent-hours",
        type=int,
        default=24,
        help="How far back to scan recent surprises + headlines (default: 24).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("samples"),
        help="Output directory for the saved report JSON.",
    )
    p.add_argument(
        "--tier2",
        choices=["off", "sampled", "on"],
        default="off",
        help="Tier-2 LLM-as-judge mode (default: off).",
    )
    p.add_argument(
        "--sample",
        type=float,
        default=0.0,
        help="Tier-2 sample rate when --tier2=sampled (0..1).",
    )
    p.add_argument(
        "--budget-usd",
        type=str,
        default="1.00",
        help="Per-cycle cost cap in USD (default: 1.00).",
    )
    return p.parse_args()


def _summarise(report: object, metrics: object, verdict: object) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("MARKET CONTEXT REPORT")
    lines.append("=" * 70)
    lines.append(f"run_id:     {report.run_id}")  # type: ignore[attr-defined]
    lines.append(f"as_of:      {report.as_of.isoformat()}")  # type: ignore[attr-defined]
    lines.append(f"confidence: {report.confidence}")  # type: ignore[attr-defined]
    lines.append("")
    lines.append("REGIMES")
    for r in report.regimes:  # type: ignore[attr-defined]
        lines.append(
            f"  {r.pair:8} risk={r.risk_state:8} trend={r.trend_state:16} "
            f"vol={r.vol_state:9} conf={r.confidence}"
        )
        if r.rationale:
            lines.append(f"           {r.rationale}")
    if report.key_scheduled_events:  # type: ignore[attr-defined]
        lines.append("")
        lines.append("KEY SCHEDULED EVENTS")
        for e in report.key_scheduled_events:  # type: ignore[attr-defined]
            lines.append(
                f"  {e.when.isoformat():25} {e.currency} {e.impact:8} {e.name}"
            )
    if report.notable_surprises:  # type: ignore[attr-defined]
        lines.append("")
        lines.append("NOTABLE SURPRISES")
        for n in report.notable_surprises:  # type: ignore[attr-defined]
            lines.append(
                f"  {n.when.isoformat():25} {n.currency} {n.name}: "
                f"actual={n.actual} forecast={n.forecast} surprise={n.surprise}"
            )
            if n.significance:
                lines.append(f"           {n.significance}")
    if report.sentiment:  # type: ignore[attr-defined]
        lines.append("")
        lines.append("SENTIMENT")
        for s in report.sentiment:  # type: ignore[attr-defined]
            lines.append(
                f"  {s.currency} {s.label:9} score={s.score}"
                + (f" — {s.rationale}" if s.rationale else "")
            )
    if report.risk_flags:  # type: ignore[attr-defined]
        lines.append("")
        lines.append("RISK FLAGS")
        for f in report.risk_flags:  # type: ignore[attr-defined]
            lines.append(f"  {f.code:25} {f.severity:6} {f.description}")
    if report.notes:  # type: ignore[attr-defined]
        lines.append("")
        lines.append("NOTES")
        lines.append(f"  {report.notes}")  # type: ignore[attr-defined]
    lines.append("")
    lines.append("METRICS")
    lines.append(f"  model:           {metrics.model}")  # type: ignore[attr-defined]
    lines.append(f"  prompt_tokens:   {metrics.prompt_tokens}")  # type: ignore[attr-defined]
    lines.append(f"  cached_tokens:   {metrics.cached_tokens}")  # type: ignore[attr-defined]
    lines.append(f"  completion_tokens: {metrics.completion_tokens}")  # type: ignore[attr-defined]
    lines.append(f"  cache_hit_ratio: {metrics.cache_hit_ratio:.1%}")  # type: ignore[attr-defined]
    lines.append(f"  estimated_cost:  ${metrics.estimated_cost_usd}")  # type: ignore[attr-defined]
    lines.append(f"  latency:         {metrics.latency_seconds:.2f}s")  # type: ignore[attr-defined]
    lines.append("")
    lines.append("EVAL VERDICT")
    lines.append(f"  tier1_passed: {verdict.tier1_passed}")  # type: ignore[attr-defined]
    if verdict.tier1_failures:  # type: ignore[attr-defined]
        for f in verdict.tier1_failures:  # type: ignore[attr-defined]
            lines.append(f"    - {f}")
    lines.append(f"  tier2_ran:    {verdict.tier2_ran}")  # type: ignore[attr-defined]
    if verdict.tier2_ran:  # type: ignore[attr-defined]
        lines.append(f"  tier2_verdict: {verdict.tier2_verdict}")  # type: ignore[attr-defined]
        for r in verdict.tier2_reasons:  # type: ignore[attr-defined]
            lines.append(f"    - {r}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fred_settings = FredSettings()
    openai_settings = OpenAISettings()

    run_id = new_run_id("ctx")
    request = MarketContextRequest(
        run_id=run_id,
        as_of=datetime.now(UTC),
        watchlist=tuple(p.upper() for p in args.watchlist),
        upcoming_hours=args.upcoming_hours,
        recent_hours=args.recent_hours,
    )

    with (
        ForexFactoryCalendarProvider() as cal,
        FredMacroDataProvider(fred_settings) as macro,
        GdeltNewsProvider() as news,
    ):
        provider = OpenAIProvider(openai_settings)
        agent = MarketContextAgent(
            calendar_provider=cal,
            macro_provider=macro,
            news_provider=news,
        )
        from decimal import Decimal as _D
        gate = EvaluationGate(
            llm_provider=provider if args.tier2 != "off" else None,
            tier2_mode=args.tier2,
            tier2_sample_rate=args.sample,
        )
        runner = AgentRunner(
            llm_provider=provider,
            evaluation_gate=gate,
            budget=AgentBudget(max_cost_per_cycle_usd=_D(args.budget_usd)),
        )
        result = runner.run(agent, request, run_id=run_id)

    if not result.succeeded:
        print(f"\nFAILED: {result.failure}", file=sys.stderr)  # type: ignore[union-attr]
        return 1

    print(_summarise(result.output, result.metrics, result.eval_verdict))
    out_path = args.out / f"{run_id}.json"
    out_path.write_text(
        json.dumps(result.output.model_dump(mode="json"), indent=2, default=str)  # type: ignore[union-attr]
    )
    print(f"\nReport saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
