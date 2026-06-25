"""market_context_agent — the first 4xPrima slow-loop agent.

Reads its spec: ``specs/agents/market_context_agent.md``.

What this agent does:

1. **Deterministic pre-step.** Pull from the three context-data providers
   (:class:`core.context_data.EconomicCalendarProvider`,
   :class:`MacroDataProvider`, :class:`NewsProvider`) and distil into a
   compact :class:`MarketContextBrief` — upcoming/recent material events
   with computed surprises, a small named-FRED-series snapshot, and the
   deduped recent headlines.
2. **LLM step.** The brief is the *volatile tail*; the agent's instructions
   and the format spec are the *stable cached prefix*. Runs on the Sonnet
   tier (good perf/cost balance, 1k cache minimum makes the prefix worth
   caching). Output is forced into the
   :class:`core.models.MarketContextReport` schema by the LLM client's
   structured-output mechanism.

What this agent does NOT do:

- It does not place orders, propose strategies, or recommend trades. The
  ``MarketContextReport`` model rejects trade-call language in its free-text
  fields as a belt-and-braces machine check.
- It does not ingest GDELT's dictionary tone score — it reads headlines
  itself. See the spec for why.
"""

from __future__ import annotations

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
    AgentRequest,
    AgentResponse,
    ContextManagementConfig,
    LLMClient,
    ModelTier,
    StableBlock,
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
    watchlist: tuple[str, ...]                        # e.g. ("EURUSD", "USDJPY")
    upcoming_hours: int = 168                         # 7 days
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
    latest_value: str | None    # rendered as string; None when missing
    previous_value: str | None


@dataclass(frozen=True, slots=True)
class MarketContextBrief:
    """The volatile payload handed to the LLM as the final user message.

    Compact and shape-stable: a JSON-serialisable summary of upcoming /
    recent events, a small macro snapshot, and headline rows. The LLM does
    NOT see raw provider data — only this brief.
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
# Stable prompt — the cacheable system prefix
# ---------------------------------------------------------------------------

# Kept as module constants so they're byte-identical across calls (a churn
# of even whitespace would invalidate the prompt cache). Keep these compact;
# they live in every call.

_AGENT_INSTRUCTIONS = """\
You are market_context_agent, a deterministic distiller of macro and \
microstructure context for an algorithmic FX trading system.

Your job: read the deterministic BRIEF in the user message and emit a single \
structured MarketContextReport via the submit_structured_output tool.

Hard rules:

- You may not recommend trades. No "buy", "sell", "go long/short", "long bias", \
"target", "stop loss", or any imperative trade language anywhere in your output. \
Describe regimes, surprises, sentiment, and risk flags neutrally.
- All claims must trace to data in the brief. Do not invent numbers, events, or \
headlines. If the brief lacks evidence for a claim, omit it.
- GDELT's dictionary tone is NOT available to you here; derive sentiment from \
the HEADLINE TEXT in the brief.
- Be specific: regime tags per pair, surprise reads per release, sentiment reads \
per currency, named risk flags.
- Confidence per regime and the overall confidence reflect how much the brief \
actually supports your call — low-evidence calls get low confidence, not a \
high-confidence guess.

The submit_structured_output tool input MUST match the MarketContextReport schema.\
"""

_OUTPUT_SCHEMA_NOTE = """\
The MarketContextReport schema has these top-level fields:

- run_id (string): copy from the brief.
- as_of (ISO-8601 UTC datetime): copy from the brief.
- schema_version: "1.0".
- regimes: list of RegimeAssessment(pair, risk_state, trend_state, vol_state, \
confidence in [0,1], rationale).
  - risk_state: risk_on | risk_off | neutral | unknown.
  - trend_state: trending_up | trending_down | mean_reverting | range_tight | \
range_wide | event_driven | unknown.
  - vol_state: low | normal | elevated | high | unknown.
- key_scheduled_events: list of ScheduledEventSummary(when, currency, name, impact).
  - impact: high | medium | low | holiday | unknown.
- notable_surprises: list of NotableSurprise(when, currency, name, actual, \
forecast, surprise, significance). Include only releases with a meaningful \
surprise; cite numbers from the brief.
- sentiment: list of SentimentRead(currency, label, score in [-1,1], rationale).
  - label: positive | neutral | negative | mixed.
- risk_flags: list of RiskFlagOut(code, severity, description).
  - severity: info | warn | alert.
  - codes use a stable taxonomy, e.g. FOMC_T+18h, NFP_T+4d, thin_liquidity_window, \
low_regime_confidence:<pair>.
- notes (≤1000 chars): neutral commentary. No trade calls.
- confidence in [0,1]: overall confidence in the report.

Currency codes must be ISO-4217 three letters (USD, EUR, GBP, JPY, ...). Pair \
codes must be six letters with no separator (EURUSD, USDJPY, ...).\
"""

_STABLE_BLOCKS: tuple[StableBlock, ...] = (
    StableBlock(kind="spec", name="agent_instructions", text=_AGENT_INSTRUCTIONS),
    StableBlock(kind="schema", name="output_schema_note", text=_OUTPUT_SCHEMA_NOTE),
)


# Impact filtering helpers — shared between brief assembly and tests.

_HIGH_OR_TIER1 = frozenset({ImpactLevel.HIGH})


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class MarketContextAgent:
    """Builds a brief from deterministic sources and asks the LLM to interpret it.

    Constructor takes the LLM client and the three data providers. The agent
    only ever reads — it never places orders or proposes strategies.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        calendar_provider: EconomicCalendarProvider,
        macro_provider: MacroDataProvider,
        news_provider: NewsProvider,
        tier: ModelTier = ModelTier.SONNET,
    ) -> None:
        self._llm = llm_client
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
        # Only releases whose `actual` is in and whose forecast was numeric
        # qualify as "surprises" — surprise is None otherwise.
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
                # Unknown name — skip rather than fail the whole brief.
                continue
            try:
                points = self._macro.get_series(series_id)
            except Exception:
                # Provider error in one series shouldn't kill the whole brief.
                # The provider already logs the failure; we surface a blank
                # snapshot and let the agent reason on what's left.
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
        user_message = {
            "role": "user",
            "content": (
                "Distil the following deterministic brief into a "
                "MarketContextReport. Use the submit_structured_output tool. "
                "Cite numbers from the brief; do not invent.\n\n"
                f"BRIEF:\n{_pretty(brief.to_json())}"
            ),
        }
        ctx_mgmt = ContextManagementConfig(
            enable_compaction=False,  # one-shot structured call; no long history
            enable_clear_thinking=False,
            enable_clear_tool_uses=False,
        )
        ag_request = AgentRequest(
            agent_name="market_context_agent",
            run_id=brief.run_id,
            tier=self._tier,
            tools=(),
            stable_system_blocks=_STABLE_BLOCKS,
            volatile_policy=None,
            messages=(user_message,),
            max_output_tokens=4096,
            context_management=ctx_mgmt,
            cache_ttl="5m",
        )
        return self._llm.call_structured(ag_request, output_model=MarketContextReport)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_material(
    events: Iterable[EconomicEvent], watchlist: Iterable[str]
) -> tuple[EconomicEvent, ...]:
    """Keep tier-1 (HIGH) events on watchlist currencies; for non-watchlist
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
    """Pretty-stable JSON serialisation for the brief — sorted keys, compact."""
    import json

    return json.dumps(obj, sort_keys=True, separators=(",", ": "))


__all__ = [
    "HeadlineRow",
    "MacroSnapshot",
    "MarketContextAgent",
    "MarketContextBrief",
    "MarketContextRequest",
]
