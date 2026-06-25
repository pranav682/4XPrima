"""market_context_agent — the first 4xPrima slow-loop agent.

Reads its spec: ``specs/agents/market_context_agent.md``.

What this agent does:

1. **Deterministic pre-step.** Pull from the three context-data providers
   (:class:`core.context_data.EconomicCalendarProvider`,
   :class:`MacroDataProvider`, :class:`NewsProvider`) and distil into a
   compact :class:`MarketContextBrief` — upcoming/recent material events
   with computed surprises, a small named-FRED-series snapshot, and the
   deduped recent headlines.
2. **LLM step.** The brief is the *volatile user message*; the agent's
   instructions and the format spec are the *stable system prefix* (~auto-
   cached by OpenAI once it crosses ~1024 tokens). Runs on the DEFAULT tier
   (``gpt-5.4``). Output is forced into
   :class:`core.models.MarketContextReport` via OpenAI Structured Outputs.

What this agent does NOT do:

- It does not place orders, propose strategies, or recommend trades. The
  ``MarketContextReport`` model rejects trade-call language in its free-text
  fields as a belt-and-braces machine check.
- It does not ingest GDELT's dictionary tone score — it reads headlines
  itself. See the spec for why.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from core.context_data import (
    FRED_SERIES,
    EconomicCalendarProvider,
    MacroDataProvider,
    NewsProvider,
)
from core.llm_client import (
    AgentResponse,
    LLMProvider,
    ModelTier,
)
from core.models import (
    EconomicEvent,
    ImpactLevel,
    MacroSeriesPoint,
    MarketContextReport,
)

# ---------------------------------------------------------------------------
# Configuration / request
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarketContextRequest:
    """One run of the agent."""

    run_id: str
    as_of: datetime                                  # UTC
    watchlist: tuple[str, ...]                       # e.g. ("EURUSD", "USDJPY")
    upcoming_hours: int = 168                        # 7 days
    recent_hours: int = 24
    macro_series_names: tuple[str, ...] = (
        "US_CPI",
        "US_CORE_CPI",
        "US_FED_FUNDS_RATE",
        "US_DOLLAR_INDEX",
        "US_2Y_YIELD",
        "US_10Y_YIELD",
    )
    news_query: str = "forex EUR USD JPY GBP CHF AUD CAD ECB Fed BoE BoJ"
    max_headlines: int = 25


# ---------------------------------------------------------------------------
# The deterministic brief
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HeadlineRow:
    timestamp: datetime
    title: str
    source: str
    url: str


@dataclass(frozen=True, slots=True)
class MacroSnapshot:
    name: str
    series_id: str
    latest_date: str | None
    latest_value: str | None
    previous_value: str | None


@dataclass(frozen=True, slots=True)
class MarketContextBrief:
    """The volatile payload handed to the LLM as the user message.

    Compact and shape-stable. The LLM does NOT see raw provider data —
    only this brief.
    """

    run_id: str
    as_of: datetime
    watchlist: tuple[str, ...]
    upcoming_events: tuple[EconomicEvent, ...]
    recent_surprises: tuple[EconomicEvent, ...]
    macro: tuple[MacroSnapshot, ...]
    headlines: tuple[HeadlineRow, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "as_of": self.as_of.isoformat(),
            "watchlist": list(self.watchlist),
            "upcoming_events": [
                {
                    "when": e.when.isoformat(),
                    "currency": e.currency,
                    "name": e.name,
                    "impact": e.impact.value,
                    "forecast": str(e.forecast) if e.forecast is not None else None,
                    "previous": str(e.previous) if e.previous is not None else None,
                }
                for e in self.upcoming_events
            ],
            "recent_surprises": [
                {
                    "when": e.when.isoformat(),
                    "currency": e.currency,
                    "name": e.name,
                    "impact": e.impact.value,
                    "actual": str(e.actual) if e.actual is not None else None,
                    "forecast": str(e.forecast) if e.forecast is not None else None,
                    "previous": str(e.previous) if e.previous is not None else None,
                    "surprise": str(e.surprise) if e.surprise is not None else None,
                }
                for e in self.recent_surprises
            ],
            "macro": [
                {
                    "name": m.name,
                    "series_id": m.series_id,
                    "latest_date": m.latest_date,
                    "latest_value": m.latest_value,
                    "previous_value": m.previous_value,
                }
                for m in self.macro
            ],
            "headlines": [
                {
                    "timestamp": h.timestamp.isoformat(),
                    "title": h.title,
                    "source": h.source,
                    "url": h.url,
                }
                for h in self.headlines
            ],
        }


# ---------------------------------------------------------------------------
# Stable system prefix — byte-identical across calls so OpenAI auto-caches it.
# ---------------------------------------------------------------------------

# Kept as a module constant so accidental whitespace churn can't break the
# cache. The prefix is intentionally long (instructions + schema notes) so it
# crosses OpenAI's ~1024-token automatic-caching threshold — see
# docs/llm-conventions.md §2.

_STABLE_SYSTEM = """\
You are market_context_agent, a deterministic distiller of macro and \
microstructure context for an algorithmic FX trading system.

Your job: read the deterministic BRIEF in the user message and emit a single \
MarketContextReport via the structured-output schema. Cite numbers and \
events from the brief; do not invent.

Hard rules:

- You may NOT recommend trades. No "buy", "sell", "go long", "go short", \
"long bias", "short bias", "target price", "stop loss", "take profit", or \
any imperative trade language anywhere in your output. Describe regimes, \
surprises, sentiment, and risk flags neutrally.
- All claims must trace to data in the brief. If the brief lacks evidence \
for a claim, omit it.
- GDELT's dictionary tone is NOT ingested into the brief. Derive sentiment \
from the HEADLINE TEXT itself.
- Be specific: regime tags per pair, surprise reads per release, sentiment \
reads per currency, named risk flags.
- Confidence (per regime and overall) reflects how much the brief actually \
supports your call. Low evidence → low confidence, not a confident guess.

OUTPUT SCHEMA (MarketContextReport):

- run_id (string): copy from the brief.
- as_of (ISO-8601 UTC datetime): copy from the brief.
- schema_version: "1.0".
- regimes: list of RegimeAssessment(pair, risk_state, trend_state, vol_state, \
confidence in [0,1], rationale).
  - risk_state: risk_on | risk_off | neutral | unknown.
  - trend_state: trending_up | trending_down | mean_reverting | range_tight | \
range_wide | event_driven | unknown.
  - vol_state: low | normal | elevated | high | unknown.
- key_scheduled_events: list of ScheduledEventSummary(when, currency, name, \
impact). impact: high | medium | low | holiday | unknown.
- notable_surprises: list of NotableSurprise(when, currency, name, actual, \
forecast, surprise, significance). Include only releases with a meaningful \
surprise; cite numbers from the brief.
- sentiment: list of SentimentRead(currency, label, score in [-1,1], \
rationale). label: positive | neutral | negative | mixed.
- risk_flags: list of RiskFlagOut(code, severity, description). severity: \
info | warn | alert. Use a stable code taxonomy (e.g. FOMC_T+18h, NFP_T+4d, \
thin_liquidity_window, low_regime_confidence:<pair>).
- notes (≤1000 chars): neutral commentary. No trade calls.
- confidence in [0,1]: overall confidence in the report.

Currency codes are ISO-4217 three letters (USD, EUR, GBP, JPY, ...). Pair \
codes are six letters with no separator (EURUSD, USDJPY, ...).

Restate (this is the rule, not advice): you are NOT a strategy. You describe \
context. The strategy lab and critic come later in the pipeline.

REGIME RUBRIC (how to choose tags):

- risk_state = risk_on when equity / EM / commodity-FX show coordinated \
strength and safe havens (USD/JPY/CHF) are softer; risk_off the mirror image; \
neutral when no coordinated move is visible; unknown when the brief lacks \
the evidence to support either.
- trend_state = trending_up / trending_down when the brief or recent \
surprises clearly tilt the pair's direction over the lookback; mean_reverting \
when prior moves are being undone; range_tight when realised vol is at the \
low end of recent regime; range_wide when high; event_driven when one \
scheduled release dominates near-term price discovery; unknown when the brief \
is too thin to choose.
- vol_state = low / normal / elevated / high relative to the pair's own \
recent norms — not absolute. A spread-widening pre-news window is elevated, \
not high.

RISK-FLAG TAXONOMY (codes are stable and machine-parseable):

- FOMC_T+<hours> / ECB_T+<hours> / BOJ_T+<hours> / BOE_T+<hours> — a tier-1 \
central-bank decision or speech in the near future.
- NFP_T+<days> / CPI_T+<days> — a tier-1 data release imminent.
- thin_liquidity_window — Asia open Sunday, US holiday, year-end illiquidity.
- gap_risk_weekend — a position carrying weekend gap exposure.
- low_regime_confidence:<pair> — confidence < 0.4 for that pair's regime.
- stale_macro:<series> — the most recent FRED observation is older than 60 \
days (releases lag).

SENTIMENT LABEL SEMANTICS:

- positive: headlines tilt the currency stronger (hawkish CB, beat-data, \
growth optimism, safe-haven inflow when applicable).
- negative: the mirror — softer-data, dovish CB, capital outflow risk.
- mixed: meaningful headlines on both sides with no dominant signal.
- neutral: thin or balanced coverage, no actionable read.
The score is intensity, not direction (label carries the sign). +0.8 = strong \
positive coverage; -0.4 = mildly negative. Use small magnitudes when evidence \
is thin.

OUTPUT EXAMPLE (shape, NOT content — copy structure, not values):

{
  "run_id": "<from brief>",
  "as_of": "<from brief, ISO-8601 Z>",
  "schema_version": "1.0",
  "regimes": [
    {
      "pair": "EURUSD",
      "risk_state": "risk_off",
      "trend_state": "trending_down",
      "vol_state": "elevated",
      "confidence": 0.62,
      "rationale": "Softer EZ data and firmer US yields in the brief."
    }
  ],
  "key_scheduled_events": [
    {"when": "<ISO-8601 Z>", "currency": "USD", "name": "FOMC Statement", \
"impact": "high"}
  ],
  "notable_surprises": [
    {"when": "<ISO-8601 Z>", "currency": "USD", "name": "Flash PMI", \
"actual": "55.0", "forecast": "54.0", "surprise": "1.0", \
"significance": "Mildly positive print; coverage notes broad services strength."}
  ],
  "sentiment": [
    {"currency": "USD", "label": "positive", "score": 0.3, \
"rationale": "Hawkish coverage dominates this morning's headlines."}
  ],
  "risk_flags": [
    {"code": "FOMC_T+24h", "severity": "warn", "description": "FOMC speakers \
tomorrow; expect wider spreads in the news window."}
  ],
  "notes": "USD firmer across the board; EUR softer on growth read. EM \
mixed.",
  "confidence": 0.55
}

Note the absence of "buy", "sell", "go long", "target", or any imperative — \
this is a description, not a trade. If you feel an urge to recommend, you are \
in the wrong agent.\
"""


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class MarketContextAgent:
    """Builds a brief from deterministic sources and asks the LLM to interpret it.

    Constructor takes the LLM provider and the three data providers. The
    agent only ever reads — it never places orders or proposes strategies.
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        calendar_provider: EconomicCalendarProvider,
        macro_provider: MacroDataProvider,
        news_provider: NewsProvider,
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> None:
        self._llm = llm_provider
        self._calendar = calendar_provider
        self._macro = macro_provider
        self._news = news_provider
        self._tier = tier

    # ----------------------------------------------------- brief assembly

    def assemble_brief(self, request: MarketContextRequest) -> MarketContextBrief:
        upcoming = self._calendar.upcoming(
            timedelta(hours=request.upcoming_hours),
            min_impact=ImpactLevel.MEDIUM,
        )
        recent = self._calendar.recent(
            timedelta(hours=request.recent_hours),
            min_impact=ImpactLevel.MEDIUM,
        )
        # Only releases whose `actual` and `forecast` are both numeric have
        # a numeric `surprise` — that's what qualifies for the surprise list.
        recent_surprises = tuple(e for e in recent if e.surprise is not None)

        macro = tuple(self._fetch_macro(request))
        headlines = tuple(self._fetch_headlines(request))

        return MarketContextBrief(
            run_id=request.run_id,
            as_of=request.as_of,
            watchlist=request.watchlist,
            upcoming_events=_pick_material(upcoming, request.watchlist),
            recent_surprises=recent_surprises,
            macro=macro,
            headlines=headlines,
        )

    def _fetch_macro(self, request: MarketContextRequest) -> Iterable[MacroSnapshot]:
        for name in request.macro_series_names:
            series_id = FRED_SERIES.get(name)
            if series_id is None:
                continue
            try:
                points = self._macro.get_series(series_id)
            except Exception:
                # Provider error in one series shouldn't kill the whole brief.
                yield MacroSnapshot(
                    name=name,
                    series_id=series_id,
                    latest_date=None,
                    latest_value=None,
                    previous_value=None,
                )
                continue
            yield _summarise_series(name, series_id, points)

    def _fetch_headlines(
        self, request: MarketContextRequest
    ) -> Iterable[HeadlineRow]:
        end = request.as_of
        start = end - timedelta(hours=max(request.recent_hours, 6))
        try:
            events = self._news.search(
                request.news_query,
                start=start,
                end=end,
                max_results=request.max_headlines,
            )
        except Exception:
            return ()
        return tuple(
            HeadlineRow(
                timestamp=e.timestamp, title=e.title, source=e.source, url=e.url
            )
            for e in events
        )

    # -------------------------------------------------------- LLM step

    def run(self, request: MarketContextRequest) -> tuple[MarketContextReport, AgentResponse]:
        brief = self.assemble_brief(request)
        return self.run_from_brief(brief)

    def run_from_brief(
        self, brief: MarketContextBrief
    ) -> tuple[MarketContextReport, AgentResponse]:
        """Call the LLM with an already-assembled brief.

        Split from :meth:`run` so tests can drive the LLM step in isolation
        from the providers, and so other agents (or replay) can reuse the
        same prompt assembly without re-fetching.
        """
        volatile_user = (
            "Distil the following deterministic brief into a "
            "MarketContextReport. Cite numbers from the brief; do not invent.\n\n"
            f"BRIEF:\n{_pretty(brief.to_json())}"
        )
        return self._llm.generate_structured(
            agent_name="market_context_agent",
            run_id=brief.run_id,
            tier=self._tier,
            stable_system=_STABLE_SYSTEM,
            volatile_user=volatile_user,
            output_model=MarketContextReport,
            max_output_tokens=4096,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_material(
    events: Iterable[EconomicEvent], watchlist: Iterable[str]
) -> tuple[EconomicEvent, ...]:
    """Keep HIGH + MEDIUM events on watchlist currencies; for non-watchlist
    currencies keep only HIGH-impact releases that affect majors broadly."""
    wl_currencies = {ccy for pair in watchlist for ccy in (pair[:3], pair[3:])}
    out: list[EconomicEvent] = []
    for e in events:
        if e.currency in wl_currencies:
            if e.impact in (ImpactLevel.HIGH, ImpactLevel.MEDIUM):
                out.append(e)
        elif e.impact == ImpactLevel.HIGH:
            out.append(e)
    return tuple(out)


def _summarise_series(
    name: str, series_id: str, points: list[MacroSeriesPoint]
) -> MacroSnapshot:
    """Pick the two latest non-missing observations and render them."""
    non_missing = [p for p in points if p.value is not None]
    if not non_missing:
        return MacroSnapshot(
            name=name,
            series_id=series_id,
            latest_date=None,
            latest_value=None,
            previous_value=None,
        )
    latest = non_missing[-1]
    previous_value = non_missing[-2].value if len(non_missing) >= 2 else None
    return MacroSnapshot(
        name=name,
        series_id=series_id,
        latest_date=latest.date.isoformat(),
        latest_value=str(latest.value),
        previous_value=str(previous_value) if previous_value is not None else None,
    )


def _pretty(obj: dict[str, Any]) -> str:
    """Deterministic compact JSON for the brief — sorted keys."""
    return json.dumps(obj, sort_keys=True, separators=(",", ": "))


__all__ = [
    "HeadlineRow",
    "MacroSnapshot",
    "MarketContextAgent",
    "MarketContextBrief",
    "MarketContextRequest",
]
