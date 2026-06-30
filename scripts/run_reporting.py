"""Dev CLI: turn a saved orchestrator CycleResult into a human-readable report.

Loads a ``CycleResult`` (as written by ``scripts/run_orchestrator.py``) plus the
pending approval-queue items for that cycle, runs reporting_agent, and prints +
saves the honest summary. The agent REPORTS only — it recommends nothing and
authorizes nothing; survivors are framed as "the critic did not kill this".

Usage:
    python -m scripts.run_orchestrator              # produces samples/<cycle>.json
    python -m scripts.run_reporting --cycle samples/<cycle>.json
    python -m scripts.run_reporting --cycle samples/<cycle>.json --tier cheap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.agents.evaluation import EvaluationGate
from core.agents.reporting_agent import ReportingAgent, ReportingRequest
from core.agents.runner import AgentRunner
from core.agents.types import AgentBudget
from core.config import OpenAISettings
from core.llm_client import ModelTier, OpenAIProvider
from core.models import CycleReport
from core.orchestration import ApprovalQueue, CycleResult

DEFAULT_UNIVERSE = ("USDJPY", "EURUSD", "GBPUSD", "USDCAD", "AUDUSD")
_TIERS = {"default": ModelTier.DEFAULT, "cheap": ModelTier.CHEAP}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cycle", type=Path, required=True, help="Saved CycleResult JSON.")
    p.add_argument(
        "--state-dir", type=Path, default=Path("data/orchestration"), help="Registry + queue dir."
    )
    p.add_argument("--universe", nargs="+", default=list(DEFAULT_UNIVERSE), help="Pairs covered.")
    p.add_argument("--tier", choices=sorted(_TIERS), default="default", help="LLM tier.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/orchestration/reports"),
        help="Report output dir (the dashboard API reads reports from here).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cycle = CycleResult.model_validate_json(args.cycle.read_text())
    queue = ApprovalQueue(args.state_dir / "approval_queue.json")
    queued = tuple(e for e in queue.pending() if e.cycle_id == cycle.cycle_id)

    request = ReportingRequest(
        run_id=cycle.cycle_id,
        cycle_result=cycle,
        universe=tuple(p.upper() for p in args.universe),
        queued_items=queued,
        tier=_TIERS[args.tier],
    )

    runner = AgentRunner(
        llm_provider=OpenAIProvider(OpenAISettings()),  # type: ignore[call-arg]
        evaluation_gate=EvaluationGate(),
        budget=AgentBudget(),
    )
    result = runner.run(ReportingAgent(), request, run_id=cycle.cycle_id)

    if not result.succeeded or not isinstance(result.output, CycleReport):
        print(f"FAILED: {result.failure}", file=sys.stderr)
        return 1

    print(_render(result.output))
    out_path = args.out / f"report-{cycle.cycle_id}.json"
    out_path.write_text(json.dumps(result.output.model_dump(mode="json"), indent=2, default=str))
    print(f"\nReport saved to {out_path}")
    return 0


def _render(report: CycleReport) -> str:
    s = report.summary
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("CYCLE REPORT (honest summary; reports only — recommends/authorizes nothing)")
    lines.append("=" * 80)
    lines.append(report.headline)
    lines.append("")
    lines.append(f"cycle_id: {s.cycle_id}   outcome: {s.outcome}")
    lines.append(f"pairs_covered: {', '.join(s.pairs_covered)}")
    lines.append(
        f"proposed={s.candidates_proposed}  killed={s.candidates_killed}  "
        f"queued={s.candidates_queued}   cost=${s.total_cost_usd}   "
        f"duration={s.duration_seconds:.1f}s"
    )
    lines.append("")
    lines.append(f"AWAITING THE OPERATOR — {len(report.queued_for_approval)} candidate(s):")
    for q in report.queued_for_approval:
        ism = q.in_sample_metrics
        oosm = q.out_of_sample_metrics
        oos = (
            "no OOS"
            if oosm is None
            else f"OOS sharpe={oosm.sharpe_ratio:.2f} return={oosm.total_return_pct}"
        )
        lines.append("")
        lines.append(f"  [{q.identity}] {q.instrument} {q.timeframe.value} {q.archetype.value}")
        lines.append(f"    critic verdict: {q.critic_verdict} (NOT validation)")
        lines.append(f"    in-sample: sharpe={ism.sharpe_ratio:.2f} return={ism.total_return_pct}")
        lines.append(f"    out-sample: {oos}")
        lines.append("    surviving critic concerns:")
        for c in q.surviving_concerns:
            lines.append(f"      - [{c.item.value}] {c.finding}")
        lines.append(f"    what to look at: {q.explanation}")
    lines.append("")
    lines.append(f"NOTICE: {report.operator_decision_notice}")
    lines.append("=" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
