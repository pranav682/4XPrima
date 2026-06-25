"""Tests for market_context_agent — fully hermetic.

Mocks the llm_client and the three providers (calendar, macro, news) so we
exercise:

- Brief assembly from fixture provider data.
- Parsing the LLM's canned structured response into a MarketContextReport.
- The no-trade-call invariant (the report model rejects trade language).
- Defensive behavior: provider exceptions don't kill the brief.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from core.agents.market_context_agent import (
    MarketContextAgent,
    MarketContextRequest,
)
from core.llm_client import (
    AgentResponse,
    ModelTier,
    TokenUsage,
)
from core.models import (
    EconomicEvent,
    FlagSeverity,
    ImpactLevel,
    MacroSeriesPoint,
    MarketContextReport,
    NewsEvent,
    RegimeAssessment,
    RiskFlagOut,
    RiskState,
    SentimentLabel,
    SentimentRead,
    TrendState,
    VolState,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 6, 24, 12, 0, tzinfo=UTC)


@pytest.fixture
def upcoming_events(now: datetime) -> list[EconomicEvent]:
    return [
        # USD HIGH within 7 days — should be in brief.
        EconomicEvent(
            when=now + timedelta(days=1),
            currency="USD",
            name="Non-Farm Employment Change",
            impact=ImpactLevel.HIGH,
            forecast=Decimal("180000"),
            previous=Decimal("199000"),
        ),
        # JPY HIGH — agent watches USDJPY → relevant currency.
        EconomicEvent(
            when=now + timedelta(days=2),
            currency="JPY",
            name="BoJ Policy Rate",
            impact=ImpactLevel.HIGH,
            forecast=Decimal("0.5"),
            previous=Decimal("0.25"),
        ),
        # Off-watchlist + LOW — should be filtered out.
        EconomicEvent(
            when=now + timedelta(days=3),
            currency="CHF",
            name="SECO Consumer Climate",
            impact=ImpactLevel.LOW,
            forecast=Decimal("-10"),
            previous=Decimal("-12"),
        ),
    ]


@pytest.fixture
def recent_events(now: datetime) -> list[EconomicEvent]:
    return [
        # USD with a surprise — should appear under recent_surprises.
        EconomicEvent(
            when=now - timedelta(hours=20),
            currency="USD",
            name="Flash PMI",
            impact=ImpactLevel.MEDIUM,
            actual=Decimal("55.0"),
            forecast=Decimal("54.0"),
            previous=Decimal("53.0"),
        ),
        # Forecast missing → surprise=None → excluded.
        EconomicEvent(
            when=now - timedelta(hours=18),
            currency="EUR",
            name="Trade Balance",
            impact=ImpactLevel.MEDIUM,
            actual=Decimal("20.0"),
            previous=Decimal("18.0"),
        ),
    ]


@pytest.fixture
def macro_series_data() -> dict[str, list[MacroSeriesPoint]]:
    base = [
        MacroSeriesPoint(series_id="CPIAUCSL", date=date(2026, 3, 1), value=Decimal("319.0")),
        MacroSeriesPoint(series_id="CPIAUCSL", date=date(2026, 4, 1), value=None),
        MacroSeriesPoint(series_id="CPIAUCSL", date=date(2026, 5, 1), value=Decimal("321.0")),
    ]
    return {
        "CPIAUCSL": base,
        "DFF": base,
        "DTWEXBGS": base,
        "CPILFESL": base,
        "DGS2": base,
        "DGS10": base,
    }


@pytest.fixture
def headlines(now: datetime) -> list[NewsEvent]:
    return [
        NewsEvent(
            timestamp=now - timedelta(hours=2),
            title="Fed minutes hint at hold; USD firm",
            source="reuters.com",
            url="https://reuters.com/a1",
        ),
        NewsEvent(
            timestamp=now - timedelta(hours=4),
            title="BoJ keeps rate; yen weakens",
            source="ft.com",
            url="https://ft.com/a2",
        ),
    ]


def make_calendar(upcoming: list[EconomicEvent], recent: list[EconomicEvent]) -> MagicMock:
    cal = MagicMock()
    cal.upcoming.return_value = list(upcoming)
    cal.recent.return_value = list(recent)
    return cal


def make_macro(series_data: dict[str, list[MacroSeriesPoint]]) -> MagicMock:
    macro = MagicMock()
    macro.get_series.side_effect = lambda series_id, **_: series_data.get(series_id, [])
    return macro


def make_news(headlines: list[NewsEvent]) -> MagicMock:
    news = MagicMock()
    news.search.return_value = list(headlines)
    return news


def make_report(now: datetime) -> MarketContextReport:
    return MarketContextReport(
        run_id="r1",
        as_of=now,
        regimes=(
            RegimeAssessment(
                pair="EURUSD",
                risk_state=RiskState.RISK_OFF,
                trend_state=TrendState.TRENDING_DOWN,
                vol_state=VolState.ELEVATED,
                confidence=Decimal("0.6"),
                rationale="EUR data soft and US yields firm.",
            ),
        ),
        sentiment=(
            SentimentRead(
                currency="USD",
                label=SentimentLabel.POSITIVE,
                score=Decimal("0.3"),
                rationale="Hawkish Fed coverage dominates headlines.",
            ),
        ),
        risk_flags=(
            RiskFlagOut(
                code="FOMC_T+24h",
                severity=FlagSeverity.WARN,
                description="FOMC speakers tomorrow; expect wider spreads.",
            ),
        ),
        notes="USD strength across the board; EUR softer on growth.",
        confidence=Decimal("0.55"),
    )


# ---------------------------------------------------------------------------
# Brief assembly
# ---------------------------------------------------------------------------


def test_brief_assembly_filters_to_material_events(
    now, upcoming_events, recent_events, macro_series_data, headlines
) -> None:
    llm = MagicMock()
    agent = MarketContextAgent(
        llm_provider=llm,
        calendar_provider=make_calendar(upcoming_events, recent_events),
        macro_provider=make_macro(macro_series_data),
        news_provider=make_news(headlines),
    )
    brief = agent.assemble_brief(
        MarketContextRequest(
            run_id="r1", as_of=now, watchlist=("EURUSD", "USDJPY")
        )
    )

    # Upcoming: USD HIGH (watchlist) + JPY HIGH (watchlist) keep; CHF LOW drops.
    upcoming_pairs = {(e.currency, e.name) for e in brief.upcoming_events}
    assert ("USD", "Non-Farm Employment Change") in upcoming_pairs
    assert ("JPY", "BoJ Policy Rate") in upcoming_pairs
    assert all(e.currency != "CHF" for e in brief.upcoming_events)

    # Recent: only the USD PMI has a surprise; EUR trade balance has no
    # forecast in the brief, so surprise=None and it's excluded.
    assert len(brief.recent_surprises) == 1
    surprise = brief.recent_surprises[0]
    assert surprise.currency == "USD"
    assert surprise.surprise == Decimal("1.0")


def test_brief_macro_snapshot_picks_two_latest_non_missing(
    now, upcoming_events, recent_events, macro_series_data, headlines
) -> None:
    llm = MagicMock()
    agent = MarketContextAgent(
        llm_provider=llm,
        calendar_provider=make_calendar(upcoming_events, recent_events),
        macro_provider=make_macro(macro_series_data),
        news_provider=make_news(headlines),
    )
    brief = agent.assemble_brief(
        MarketContextRequest(
            run_id="r1", as_of=now, watchlist=("EURUSD",),
        )
    )

    cpi = next(m for m in brief.macro if m.name == "US_CPI")
    # The middle observation was None → latest = 321.0, previous = 319.0
    assert cpi.latest_value == "321.0"
    assert cpi.previous_value == "319.0"


def test_brief_assembly_survives_provider_exceptions(
    now, upcoming_events, recent_events, headlines
) -> None:
    """A throwing macro / news provider should yield a blank snapshot / empty
    headlines, not crash the brief."""
    llm = MagicMock()
    macro = MagicMock()
    macro.get_series.side_effect = RuntimeError("simulated outage")
    news = MagicMock()
    news.search.side_effect = RuntimeError("simulated outage")
    agent = MarketContextAgent(
        llm_provider=llm,
        calendar_provider=make_calendar(upcoming_events, recent_events),
        macro_provider=macro,
        news_provider=news,
    )
    brief = agent.assemble_brief(
        MarketContextRequest(run_id="r1", as_of=now, watchlist=("EURUSD",))
    )
    # Macro snapshots present but blank-valued.
    assert len(brief.macro) > 0
    assert all(m.latest_value is None for m in brief.macro)
    # Headlines empty.
    assert brief.headlines == ()


def test_brief_to_json_is_deterministic(
    now, upcoming_events, recent_events, macro_series_data, headlines
) -> None:
    llm = MagicMock()
    agent = MarketContextAgent(
        llm_provider=llm,
        calendar_provider=make_calendar(upcoming_events, recent_events),
        macro_provider=make_macro(macro_series_data),
        news_provider=make_news(headlines),
    )
    req = MarketContextRequest(run_id="r1", as_of=now, watchlist=("EURUSD",))
    a = agent.assemble_brief(req).to_json()
    b = agent.assemble_brief(req).to_json()
    assert a == b


# ---------------------------------------------------------------------------
# LLM step (mocked llm_client)
# ---------------------------------------------------------------------------


def test_run_calls_llm_with_default_tier_and_returns_parsed_report(
    now, upcoming_events, recent_events, macro_series_data, headlines
) -> None:
    """The agent dispatches via the provider-agnostic generate_structured
    API on the DEFAULT tier, with the stable system prefix and the
    volatile brief as the user message."""
    canned = make_report(now)
    llm = MagicMock()
    llm.generate_structured.return_value = (
        canned,
        AgentResponse(
            agent_name="market_context_agent",
            run_id="r1",
            tier=ModelTier.DEFAULT,
            model="gpt-5.4",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=1500, cached_tokens=900, completion_tokens=80),
        ),
    )
    agent = MarketContextAgent(
        llm_provider=llm,
        calendar_provider=make_calendar(upcoming_events, recent_events),
        macro_provider=make_macro(macro_series_data),
        news_provider=make_news(headlines),
    )
    report, meta = agent.run(
        MarketContextRequest(
            run_id="r1", as_of=now, watchlist=("EURUSD", "USDJPY"),
        )
    )

    llm.generate_structured.assert_called_once()
    kwargs = llm.generate_structured.call_args.kwargs
    assert kwargs["tier"] == ModelTier.DEFAULT
    assert kwargs["agent_name"] == "market_context_agent"
    assert kwargs["output_model"] is MarketContextReport
    # The stable system prefix is large enough to cross the OpenAI cache
    # threshold (verified separately) — assert presence here.
    assert "market_context_agent" in kwargs["stable_system"]
    # Volatile brief carries the BRIEF tag and the run_id.
    assert "BRIEF" in kwargs["volatile_user"]
    assert "r1" in kwargs["volatile_user"]

    assert report is canned
    assert meta.tier == ModelTier.DEFAULT


def test_report_must_not_contain_trade_calls() -> None:
    """The MarketContextReport schema rejects trade-language in free-text
    fields. This is the belt-and-braces machine check that the user's
    invariant ('MUST NOT contain trade recommendations') is enforced."""
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="trade-recommendation"):
        MarketContextReport(
            run_id="r1",
            as_of=datetime(2026, 6, 24, tzinfo=UTC),
            notes="Should buy EURUSD here on the dip.",
        )
    with pytest.raises(pydantic.ValidationError, match="trade-recommendation"):
        SentimentRead(
            currency="USD",
            label=SentimentLabel.POSITIVE,
            score=Decimal("0.5"),
            rationale="Go long USD now.",
        )
    with pytest.raises(pydantic.ValidationError, match="trade-recommendation"):
        RiskFlagOut(
            code="X",
            severity=FlagSeverity.WARN,
            description="Take profit at the next resistance.",
        )


def test_descriptive_language_passes_through() -> None:
    """Non-recommendation language doesn't trip the validator: 'long-term',
    'USD strength', 'EUR softer' etc. are descriptive, not imperative."""
    r = MarketContextReport(
        run_id="r1",
        as_of=datetime(2026, 6, 24, tzinfo=UTC),
        notes=(
            "USD strength persists across G10. EUR softer on weaker growth "
            "data. Long-term real yields steady."
        ),
    )
    assert "USD strength" in r.notes


def test_malformed_llm_output_propagates_as_validation_error(
    now, upcoming_events, recent_events, macro_series_data, headlines
) -> None:
    """If the LLM client's parsed structured output is malformed, that
    failure surfaces as an LlmClientError raised by call_structured itself
    (mocked here as a side_effect)."""
    from core.llm_client import LlmClientError

    llm = MagicMock()
    llm.generate_structured.side_effect = LlmClientError(
        "model output failed schema validation"
    )
    agent = MarketContextAgent(
        llm_provider=llm,
        calendar_provider=make_calendar(upcoming_events, recent_events),
        macro_provider=make_macro(macro_series_data),
        news_provider=make_news(headlines),
    )
    with pytest.raises(LlmClientError):
        agent.run(MarketContextRequest(run_id="r1", as_of=now, watchlist=("EURUSD",)))
