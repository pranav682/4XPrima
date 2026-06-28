"""Dev CLI: run the structural pair screener live against OANDA.

Pulls daily candles (and a current quote for the spread) for the candidate
pairs, runs the deterministic :class:`core.analysis.pair_screener.PairScreener`,
prints the readable report, and saves the full report JSON to ``samples/``
(gitignored).

This is STRUCTURAL screening to narrow the universe — it does NOT rank pairs by
profitability. No LLM, no strategy, no orders.

Usage:
    python -m scripts.run_pair_screen
    python -m scripts.run_pair_screen --pairs EURUSD USDJPY GBPUSD --count 500
    python -m scripts.run_pair_screen --granularity D --shortlist 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from core.analysis.pair_screener import DEFAULT_MAJORS, PairScreener, ScreenConfig
from core.config import OandaSettings
from core.market_data import MarketDataError, OandaPriceProvider
from core.models import Granularity


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pairs",
        nargs="+",
        default=list(DEFAULT_MAJORS),
        help=f"Candidate pairs (default: {' '.join(DEFAULT_MAJORS)}).",
    )
    p.add_argument(
        "--granularity",
        default=Granularity.D.value,
        choices=[g.value for g in Granularity],
        help="Candle granularity (default: D).",
    )
    p.add_argument(
        "--count",
        type=int,
        default=500,
        help="Lookback in bars (default: 500 ~= 2 trading years of dailies).",
    )
    p.add_argument(
        "--shortlist",
        type=int,
        default=5,
        help="Target shortlist size (default: 5).",
    )
    p.add_argument(
        "--max-correlation",
        type=float,
        default=0.80,
        help="Drop a pair whose |correlation| with a picked pair exceeds this.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("samples"),
        help="Output directory for the saved report JSON.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = ScreenConfig(
        pairs=tuple(p.upper() for p in args.pairs),
        granularity=Granularity(args.granularity),
        lookback_count=args.count,
        shortlist_size=args.shortlist,
        max_correlation=args.max_correlation,
    )

    # pydantic-settings populates api_token / account_id from env / .env;
    # mypy can't see that without the pydantic plugin.
    settings = OandaSettings()  # type: ignore[call-arg]
    as_of = datetime.now(UTC)
    with OandaPriceProvider(settings) as provider:
        screener = PairScreener(provider, price_provider=provider)
        try:
            report = screener.screen(config, as_of=as_of)
        except MarketDataError as e:
            print(f"\nFAILED: market-data error: {type(e).__name__}", file=sys.stderr)
            return 1

    print(report.render())

    stamp = as_of.strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out / f"pair_screen_{stamp}.json"
    out_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, default=str))
    print(f"\nReport saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
