"""Dev CLI: run the deterministic backtest harness + backtest_agent.

Feed it a saved StrategyProposal JSON (e.g. one from
``scripts/run_strategy_lab.py``). For each candidate the harness builds the real
strategy and runs the Stage-2 engine over the IN-SAMPLE window (the OOS holdout
stays sealed), producing deterministic BacktestEvidence. The agent then
INTERPRETS that evidence — it never computes or alters a number. Prints the
verdicts readably and saves the full BacktestVerdictSet JSON to ``samples/``
(gitignored).

Issues one DEFAULT-tier (gpt-5.4) OpenAI call through the AgentRunner + gate.
Candles come live from OANDA (read-only). No orders, no live trading.

Usage:
    python -m scripts.run_backtest_agent --proposal samples/lab-XXXX.json
    python -m scripts.run_backtest_agent --proposal samples/lab-XXXX.json --lookback 1500
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

from core.agents.backtest_agent import BacktestAgent, BacktestAgentRequest
from core.agents.backtest_harness import BacktestRunConfig, run_proposal
from core.agents.evaluation import EvaluationGate
from core.agents.runner import AgentRunner, new_run_id
from core.agents.types import AgentBudget
from core.config import OandaSettings, OpenAISettings
from core.llm_client import OpenAIProvider
from core.market_data import OandaPriceProvider
from core.models import BacktestEvidence, BacktestVerdictSet, StrategyProposal


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--proposal", type=Path, required=True, help="Path to a saved StrategyProposal JSON."
    )
    p.add_argument("--lookback", type=int, default=1500, help="Bars to fetch per candidate.")
    p.add_argument(
        "--oos-fraction", type=float, default=0.2, help="Held-out OOS tail (sealed, not run)."
    )
    p.add_argument("--out", type=Path, default=Path("samples"), help="Output directory.")
    p.add_argument("--budget-usd", type=str, default="1.00", help="Per-cycle cost cap (USD).")
    return p.parse_args()


def _summarise(verdicts: BacktestVerdictSet, evidence: tuple[BacktestEvidence, ...]) -> str:
    by_id = {e.candidate_id: e for e in evidence}
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("BACKTEST AGENT — interpreted in-sample verdicts (NOT predictive, NOT trades)")
    lines.append("=" * 80)
    lines.append(f"run_id: {verdicts.run_id}   verdicts: {len(verdicts.verdicts)}")
    for v in verdicts.verdicts:
        m = v.metrics
        lines.append("")
        lines.append(f"  {v.candidate_id}  [{v.triage.value}]  config_hash={v.config_hash}")
        lines.append(
            f"     metrics: return={m.total_return_pct} sharpe={m.sharpe_ratio:.3f} "
            f"maxDD={m.max_drawdown_pct} PF={m.profit_factor} trades={m.trade_count}"
        )
        gates = ", ".join(f"{g.name}={'ok' if g.passed else 'FAIL'}" for g in v.gates)
        lines.append(f"     gates:   {gates}")
        ev = by_id.get(v.candidate_id)
        if ev is not None:
            lines.append(
                f"     window:  {ev.window_start.date()} .. {ev.window_end.date()} "
                f"({ev.bars_total} bars in-sample)"
            )
        lines.append(f"     read:    {v.assessment}")
        if v.concerns:
            lines.append(f"     concerns: {'; '.join(v.concerns)}")
        lines.append(f"     caveats: {v.caveats}")
    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.proposal.exists():
        print(f"\nFAILED: proposal file not found: {args.proposal}", file=sys.stderr)
        return 1
    proposal = StrategyProposal(**json.loads(args.proposal.read_text()))

    config = BacktestRunConfig(lookback_count=args.lookback, oos_fraction=args.oos_fraction)
    oanda = OandaSettings()  # type: ignore[call-arg]
    with OandaPriceProvider(oanda) as candles:
        evidence, skipped = run_proposal(proposal, candle_provider=candles, config=config)

    for candidate_id, reason in skipped:
        print(f"skipped {candidate_id}: {reason}", file=sys.stderr)
    if not evidence:
        print("\nFAILED: no candidate could be backtested.", file=sys.stderr)
        return 1

    run_id = new_run_id("bt")
    provider = OpenAIProvider(OpenAISettings())  # type: ignore[call-arg]
    runner = AgentRunner(
        llm_provider=provider,
        evaluation_gate=EvaluationGate(),
        budget=AgentBudget(max_cost_per_cycle_usd=Decimal(args.budget_usd)),
    )
    result = runner.run(
        BacktestAgent(),
        BacktestAgentRequest(run_id=run_id, proposal=proposal, evidence=evidence),
        run_id=run_id,
    )

    if not result.succeeded:
        print(f"\nFAILED: {result.failure}", file=sys.stderr)
        return 1

    verdicts = result.output
    assert isinstance(verdicts, BacktestVerdictSet)
    print(_summarise(verdicts, evidence))

    out_path = args.out / f"{run_id}.json"
    out_path.write_text(json.dumps(verdicts.model_dump(mode="json"), indent=2, default=str))
    print(f"\nVerdicts saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
