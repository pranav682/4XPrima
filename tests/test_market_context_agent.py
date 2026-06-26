"""Tests for market_context_agent — fully hermetic.

After the Stage-2-part-2 refactor, the agent only PREPARES calls — it
doesn't make them. The runner makes the call, the gate evaluates it. So
these tests cover:

- Brief assembly from fixture provider data (unchanged behaviour).
- prepare_call() returns an AgentCall with the right tier, schema, and
  caching-friendly shape (stable system + volatile user).
- Tier-1 checks reject (a) a planted trade-call, (b) out-of-range
  confidence, (c) a missing required field — even when the bad output is
  built via model_construct() to bypass pydantic validation.
- End-to-end through AgentRunner with a mocked LLM provider returning a
  fixture report.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.agents.evaluation import EvaluationGate
from core.agents.market_context_agent import (
    MarketContextAgent,
    MarketContextRequest,
)
from core.agents.runner import AgentRunner
from core.agents.types import AgentBudget, AgentCall
from core.llm_client import AgentResponse, ModelTier, TokenUsage
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

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 6, 24, 12, 0, tzinfo=UTC)


@pytest.fixture
def upcoming_events(now: datetime) -> list[EconomicEvent]:
    return [
        EconomicEvent(
            when=now + timedelta(days=1),
            currency="USD",
            name="Non-Farm Employment Change",
            impact=ImpactLevel.HIGH,
            forecast=Decimal("180000"),
            previous=Decimal("199000"),
        ),
        EconomicEvent(
            when=now + timedelta(days=2),
            currency="JPY",
            name="BoJ Policy Rate",
            impact=ImpactLevel.HIGH,
            forecast=Decimal("0.5"),
            previous=Decimal("0.25"),
        ),
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
        EconomicEvent(
            when=now - timedelta(hours=20),
            currency="USD",
            name="Flash PMI",
            impact=ImpactLevel.MEDIUM,
            actual=Decimal("55.0"),
            forecast=Decimal("54.0"),
            previous=Decimal("53.0"),
        ),
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


def _make_agent(
    upcoming_events, recent_events, macro_series_data, headlines
) -> MarketContextAgent:
    return MarketContextAgent(
        calendar_provider=make_calendar(upcoming_events, recent_events),
        macro_provider=make_macro(macro_series_data),
        news_provider=make_news(headlines),
    )


def _fixture_report() -> MarketContextReport:
    data = json.loads((FIXTURES / "market_context_report_sample.json").read_text())
    return MarketContextReport(**data)


# ---------------------------------------------------------------------------
# Brief assembly (unchanged behaviour)
# ---------------------------------------------------------------------------


def test_brief_assembly_filters_to_material_events(
    now, upcoming_events, recent_events, macro_series_data, headlines
) -> None:
    agent = _make_agent(upcoming_events, recent_events, macro_series_data, headlines)
    brief = agent.assemble_brief(
        MarketContextRequest(run_id="r1", as_of=now, watchlist=("EURUSD", "USDJPY"))
    )
    upcoming_pairs = {(e.currency, e.name) for e in brief.upcoming_events}
    assert ("USD", "Non-Farm Employment Change") in upcoming_pairs
    assert ("JPY", "BoJ Policy Rate") in upcoming_pairs
    assert all(e.currency != "CHF" for e in brief.upcoming_events)
    assert len(brief.recent_surprises) == 1
    assert brief.recent_surprises[0].currency == "USD"
    assert brief.recent_surprises[0].surprise == Decimal("1.0")


def test_brief_macro_snapshot_picks_two_latest_non_missing(
    now, upcoming_events, recent_events, macro_series_data, headlines
) -> None:
    agent = _make_agent(upcoming_events, recent_events, macro_series_data, headlines)
    brief = agent.assemble_brief(
        MarketContextRequest(run_id="r1", as_of=now, watchlist=("EURUSD",))
    )
    cpi = next(m for m in brief.macro if m.name == "US_CPI")
    assert cpi.latest_value == "321.0"
    assert cpi.previous_value == "319.0"


def test_brief_assembly_survives_provider_exceptions(now, upcoming_events, recent_events) -> None:
    macro = MagicMock()
    macro.get_series.side_effect = RuntimeError("simulated outage")
    news = MagicMock()
    news.search.side_effect = RuntimeError("simulated outage")
    agent = MarketContextAgent(
        calendar_provider=make_calendar(upcoming_events, recent_events),
        macro_provider=macro,
        news_provider=news,
    )
    brief = agent.assemble_brief(
        MarketContextRequest(run_id="r1", as_of=now, watchlist=("EURUSD",))
    )
    assert len(brief.macro) > 0
    assert all(m.latest_value is None for m in brief.macro)
    assert brief.headlines == ()


def test_brief_to_json_is_deterministic(
    now, upcoming_events, recent_events, macro_series_data, headlines
) -> None:
    agent = _make_agent(upcoming_events, recent_events, macro_series_data, headlines)
    req = MarketContextRequest(run_id="r1", as_of=now, watchlist=("EURUSD",))
    assert agent.assemble_brief(req).to_json() == agent.assemble_brief(req).to_json()


# ---------------------------------------------------------------------------
# Agent Protocol — prepare_call returns the right AgentCall
# ---------------------------------------------------------------------------


def test_agent_name_is_market_context_agent() -> None:
    assert MarketContextAgent.name == "market_context_agent"


def test_prepare_call_shape(
    now, upcoming_events, recent_events, macro_series_data, headlines
) -> None:
    agent = _make_agent(upcoming_events, recent_events, macro_series_data, headlines)
    call = agent.prepare_call(
        MarketContextRequest(run_id="r1", as_of=now, watchlist=("EURUSD", "USDJPY"))
    )
    assert isinstance(call, AgentCall)
    assert call.tier == ModelTier.DEFAULT
    assert call.output_model is MarketContextReport
    # Stable prefix is the per-call-stable instructions; large enough to
    # cross OpenAI's 1024-token auto-cache threshold (regression test).
    assert "market_context_agent" in call.stable_system
    assert len(call.stable_system) >= 4096  # chars; ~1024 tokens at chars/4
    # Volatile carries the brief tag and run_id.
    assert "BRIEF" in call.volatile_user
    assert "r1" in call.volatile_user
    # input_snapshot is the brief.to_json — used by the eval gate.
    assert call.input_snapshot["run_id"] == "r1"
    assert "upcoming_events" in call.input_snapshot


def test_evaluations_returns_required_checks() -> None:
    agent = MarketContextAgent.__new__(MarketContextAgent)
    names = {c.name for c in agent.evaluations()}
    assert {
        "required_fields_present",
        "run_id_preserved",
        "confidences_bounded",
        "no_trade_calls",
    } <= names


# ---------------------------------------------------------------------------
# Tier-1 checks catch planted bad output
# ---------------------------------------------------------------------------


def _construct_with(good: MarketContextReport, **overrides) -> MarketContextReport:
    """Build a malformed report via model_construct so we bypass pydantic
    validators. Nested fields stay as their ORIGINAL pydantic instances
    (model_dump would serialise them to dicts and break iteration)."""
    base = {
        "run_id": good.run_id,
        "as_of": good.as_of,
        "schema_version": good.schema_version,
        "regimes": good.regimes,
        "key_scheduled_events": good.key_scheduled_events,
        "notable_surprises": good.notable_surprises,
        "sentiment": good.sentiment,
        "risk_flags": good.risk_flags,
        "notes": good.notes,
        "confidence": good.confidence,
    }
    base.update(overrides)
    return MarketContextReport.model_construct(**base)


def test_tier1_no_trade_calls_catches_model_construct_bypass() -> None:
    """Pydantic validators reject trade language at construction time. A
    ``model_construct(...)`` path bypasses validation — the Tier-1 check
    runs again at the eval boundary and catches that escape."""
    good = _fixture_report()
    bad = _construct_with(good, notes="Go long EURUSD here.")
    agent = MarketContextAgent.__new__(MarketContextAgent)
    checks = {c.name: c for c in agent.evaluations()}
    res = checks["no_trade_calls"].check(bad, {"run_id": good.run_id})
    assert not res.passed
    assert "go long" in res.reason.lower()


def test_tier1_confidences_bounded_catches_out_of_range() -> None:
    good = _fixture_report()
    bad = _construct_with(good, confidence=Decimal("1.7"))
    agent = MarketContextAgent.__new__(MarketContextAgent)
    checks = {c.name: c for c in agent.evaluations()}
    res = checks["confidences_bounded"].check(bad, {})
    assert not res.passed
    assert "1.7" in res.reason


def test_tier1_required_fields_catches_missing_run_id() -> None:
    good = _fixture_report()
    bad = _construct_with(good, run_id="")
    agent = MarketContextAgent.__new__(MarketContextAgent)
    checks = {c.name: c for c in agent.evaluations()}
    res = checks["required_fields_present"].check(bad, {})
    assert not res.passed
    assert "run_id" in res.reason


def test_tier1_run_id_preserved_catches_mismatch() -> None:
    good = _fixture_report()
    agent = MarketContextAgent.__new__(MarketContextAgent)
    checks = {c.name: c for c in agent.evaluations()}
    res = checks["run_id_preserved"].check(good, {"run_id": "different-id"})
    assert not res.passed
    assert "run_id" in res.reason


# ---------------------------------------------------------------------------
# End-to-end through AgentRunner with a mocked LLM provider + the fixture
# ---------------------------------------------------------------------------


def test_end_to_end_via_runner_with_fixture_report(
    now, upcoming_events, recent_events, macro_series_data, headlines
) -> None:
    """The runner orchestrates: prepare_call → LLM (mocked) → eval → metrics."""
    agent = _make_agent(upcoming_events, recent_events, macro_series_data, headlines)
    canned = _fixture_report()

    llm = MagicMock()
    llm.generate_structured.return_value = (
        canned,
        AgentResponse(
            agent_name="market_context_agent",
            run_id="ctx-fixture-001:attempt-0",
            tier=ModelTier.DEFAULT,
            model="gpt-5.4",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=1500, cached_tokens=900, completion_tokens=80),
            extra_metadata={"attempts": "1"},
        ),
    )
    gate = EvaluationGate()  # Tier-2 off by default
    runner = AgentRunner(
        llm_provider=llm,
        evaluation_gate=gate,
        budget=AgentBudget(),
    )
    result = runner.run(
        agent,
        MarketContextRequest(
            run_id="ctx-fixture-001", as_of=now, watchlist=("EURUSD", "USDJPY")
        ),
        run_id="ctx-fixture-001",
    )

    assert result.succeeded
    assert result.output is canned
    assert result.eval_verdict is not None
    assert result.eval_verdict.tier1_passed
    assert result.metrics.tier == ModelTier.DEFAULT
    assert result.metrics.model == "gpt-5.4"
    assert result.metrics.prompt_tokens == 1500
    assert result.metrics.cached_tokens == 900
    assert result.metrics.completion_tokens == 80
    # Cache-hit ratio matches what TokenUsage computes (900/1500 = 0.6).
    assert abs(result.metrics.cache_hit_ratio - 0.6) < 1e-9
    # Cost is non-zero on a known model.
    assert result.metrics.estimated_cost_usd > 0


# Silence unused-fixture noise — pytest only sees the names used.
_ = (
    RegimeAssessment,
    RiskFlagOut,
    RiskState,
    SentimentLabel,
    SentimentRead,
    TrendState,
    VolState,
    FlagSeverity,
)
