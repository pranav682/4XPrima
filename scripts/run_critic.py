"""Dev CLI: run the robustness/OOS harness + critic_agent.

Feed it a saved StrategyProposal (for the candidate specs) and, optionally, the
backtest_agent's BacktestVerdictSet (so only candidates that SURVIVED backtest
triage are stressed). For each candidate the deterministic harness runs:
in-sample + the token-gated OUT-OF-SAMPLE slice + cost-sensitivity +
parameter-sensitivity + trade-concentration. The critic (HEAVY tier) then tries
to KILL each candidate per the overfitting checklist. Prints the verdicts and
saves the CriticVerdictSet JSON to ``samples/`` (gitignored).

The OOS holdout opens ONLY inside the deterministic harness; the LLM receives
the resulting evidence, never the token. The critic can only kill or
survive_for_now — it never approves or authorizes trading.

Usage:
    python -m scripts.run_critic --proposal samples/lab-XXXX.json
    python -m scripts.run_critic --proposal samples/lab-XXXX.json \\
        --verdicts samples/bt-YYYY.json
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

from core.agents.backtest_harness import BacktestRunConfig, HarnessError, run_robustness
from core.agents.critic_agent import CriticAgent, CriticRequest
from core.agents.evaluation import EvaluationGate
from core.agents.runner import AgentRunner, new_run_id
from core.agents.types import AgentBudget
from core.config import OandaSettings, OpenAISettings
from core.llm_client import OpenAIProvider
from core.market_data import OandaPriceProvider
from core.models import (
    BacktestTriage,
    BacktestVerdictSet,
    CriticVerdictSet,
    RobustnessEvidence,
    StrategyProposal,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--proposal", type=Path, required=True, help="Saved StrategyProposal JSON (candidates)."
    )
    p.add_argument(
        "--verdicts",
        type=Path,
        default=None,
        help="Saved BacktestVerdictSet JSON (optional; restrict to survivors).",
    )
    p.add_argument("--lookback", type=int, default=1500, help="Bars to fetch per candidate.")
    p.add_argument("--oos-fraction", type=float, default=0.2, help="Held-out OOS tail.")
    p.add_argument("--out", type=Path, default=Path("samples"), help="Output directory.")
    p.add_argument("--budget-usd", type=str, default="2.00", help="Per-cycle cost cap (USD).")
    return p.parse_args()


def _summarise(verdicts: CriticVerdictSet, robustness: tuple[RobustnessEvidence, ...]) -> str:
    by_id = {r.candidate_id: r for r in robustness}
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("CRITIC — adversarial verdicts (kill / survive_for_now; NEVER authorizes trading)")
    lines.append("=" * 80)
    lines.append(f"run_id: {verdicts.run_id}   verdicts: {len(verdicts.verdicts)}")
    for v in verdicts.verdicts:
        lines.append("")
        lines.append(f"  {v.candidate_id}  >>> {v.verdict.value.upper()} <<<")
        m = v.in_sample_metrics
        lines.append(
            f"     in-sample:  return={m.total_return_pct} "
            f"sharpe={m.sharpe_ratio:.3f} PF={m.profit_factor}"
        )
        if v.out_of_sample_metrics is not None:
            o = v.out_of_sample_metrics
            lines.append(
                f"     out-sample: return={o.total_return_pct} "
                f"sharpe={o.sharpe_ratio:.3f} PF={o.profit_factor}"
            )
        rb = by_id.get(v.candidate_id)
        if rb is not None and rb.trade_concentration is not None:
            tc = rb.trade_concentration
            lines.append(
                f"     concentration: top1={tc.top_trade_profit_share:.2f} "
                f"top5={tc.top5_profit_share:.2f}"
            )
        for c in v.concerns:
            lines.append(f"     concern [{c.item.value}]: {c.finding}")
        lines.append(f"     verdict read: {v.assessment}")
        if v.caveats:
            lines.append(f"     caveats: {v.caveats}")
    lines.append("=" * 80)
    return "\n".join(lines)


def _survivors(proposal: StrategyProposal, verdicts_path: Path | None) -> StrategyProposal:
    """Restrict the proposal to candidates backtest_agent did not reject."""
    if verdicts_path is None:
        return proposal
    bt = BacktestVerdictSet(**json.loads(verdicts_path.read_text()))
    survived = {v.candidate_id for v in bt.verdicts if v.triage != BacktestTriage.REJECT}
    kept = tuple(c for c in proposal.candidates if c.candidate_id in survived)
    return proposal.model_copy(update={"candidates": kept})


def main() -> int:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.proposal.exists():
        print(f"\nFAILED: proposal file not found: {args.proposal}", file=sys.stderr)
        return 1
    proposal = StrategyProposal(**json.loads(args.proposal.read_text()))
    proposal = _survivors(proposal, args.verdicts)
    if not proposal.candidates:
        print("\nNothing to critique (no surviving candidates).", file=sys.stderr)
        return 1

    bt_verdicts = None
    if args.verdicts is not None and args.verdicts.exists():
        bt_verdicts = BacktestVerdictSet(**json.loads(args.verdicts.read_text()))

    config = BacktestRunConfig(lookback_count=args.lookback, oos_fraction=args.oos_fraction)
    oanda = OandaSettings()  # type: ignore[call-arg]
    robustness: list[RobustnessEvidence] = []
    with OandaPriceProvider(oanda) as candles:
        for candidate in proposal.candidates:
            try:
                robustness.append(run_robustness(candidate, candle_provider=candles, config=config))
            except HarnessError as e:
                print(f"skipped {candidate.candidate_id}: {e}", file=sys.stderr)

    if not robustness:
        print("\nFAILED: no candidate could be stressed.", file=sys.stderr)
        return 1

    run_id = new_run_id("crit")
    provider = OpenAIProvider(OpenAISettings())  # type: ignore[call-arg]
    runner = AgentRunner(
        llm_provider=provider,
        evaluation_gate=EvaluationGate(),
        budget=AgentBudget(max_cost_per_cycle_usd=Decimal(args.budget_usd)),
    )
    result = runner.run(
        CriticAgent(),
        CriticRequest(run_id=run_id, robustness=tuple(robustness), backtest_verdicts=bt_verdicts),
        run_id=run_id,
    )

    if not result.succeeded:
        print(f"\nFAILED: {result.failure}", file=sys.stderr)
        return 1

    verdicts = result.output
    assert isinstance(verdicts, CriticVerdictSet)
    print(_summarise(verdicts, tuple(robustness)))

    out_path = args.out / f"{run_id}.json"
    out_path.write_text(json.dumps(verdicts.model_dump(mode="json"), indent=2, default=str))
    print(f"\nVerdicts saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
