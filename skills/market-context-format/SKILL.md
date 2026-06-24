---
name: market-context-format
description: The exact structured schema market_context_agent must emit, plus field-level rules. Read by reporting_agent (to summarise faithfully) and strategy_lab_agent (to consume reliably).
---

# market-context-format

## When to use

- `market_context_agent` — when producing output. The schema below is the contract.
- `reporting_agent` and `strategy_lab_agent` — when consuming a `MarketContextReport`. Read-only.

## The schema

```python
class MarketContextReport(BaseModel):
    run_id: str
    as_of: datetime                          # UTC, ISO-8601, tz-aware
    schema_version: Literal["1.0"]
    regime: dict[CurrencyPair, RegimeLabel]
    regime_confidence: dict[CurrencyPair, float]   # 0.0–1.0, two-decimal
    scheduled_events: list[ScheduledEvent]   # sorted ascending by start_time
    surprise_vs_expectation: list[SurpriseEvent]
    sentiment: dict[CurrencyPair, SentimentScore]
    risk_flags: list[RiskFlag]
    notes: str                               # max 500 chars
    citations: list[Citation]                # deduped on (source, key)

class RegimeLabel(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    RANGE_TIGHT = "range_tight"
    RANGE_WIDE = "range_wide"
    EVENT_DRIVEN = "event_driven"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    UNKNOWN = "unknown"

class ScheduledEvent(BaseModel):
    start_time: datetime                     # UTC, tz-aware
    end_time: datetime | None
    region: str                              # ISO 3166-1 alpha-3 OR "EU"
    name: str                                # e.g. "FOMC Statement"
    importance: Literal["low", "medium", "high", "tier1"]
    affected_currencies: list[str]           # ISO 4217
    consensus: float | None
    previous: float | None
    citation_id: str                         # join key to Citation.id

class SurpriseEvent(BaseModel):
    released_at: datetime                    # UTC, past or present
    region: str
    name: str
    actual: float
    consensus: float
    previous: float | None
    surprise: float                          # actual - consensus
    z_score: float | None                    # if historical std known
    citation_id: str

class SentimentScore(BaseModel):
    score: float                             # -1.0..+1.0
    sources_polled: int
    methodology: Literal["count", "weighted", "vendor"]
    citation_ids: list[str]

class RiskFlag(BaseModel):
    code: str                                # e.g. "FOMC_T+18h", "thin_liquidity_window"
    severity: Literal["info", "warn", "alert"]
    description: str                         # ≤ 200 chars
    expires_at: datetime | None              # UTC

class Citation(BaseModel):
    id: str                                  # short, stable, unique within the report
    source: str                              # vendor / publisher / dataset name
    key: str                                 # url, ticker, or document id
    fetched_at: datetime                     # UTC
```

## Field-level rules

- **`as_of`** must be the timestamp of the most-recent input the agent used, not the current wall-clock — so consumers can know data freshness.
- **`regime_confidence`** is the agent's own confidence, two decimals. A score < 0.5 should also set a `RiskFlag` of `code = "low_regime_confidence:<pair>"`.
- **`scheduled_events`** must be sorted ascending by `start_time`. The list covers the requested `calendar_horizon_hours`.
- **`surprise_vs_expectation`** covers only releases inside `lookback_days` and only for instruments in `watchlist`.
- **`sentiment.score`** is a single number per pair. If multiple methodologies are used, emit one entry per pair using `methodology = "weighted"` and combine in `citation_ids`.
- **`risk_flags.code`** uses a stable taxonomy: `FOMC_T+<hours>`, `CB_DECISION_T+<hours>`, `NFP_T+<hours>`, `thin_liquidity_window`, `low_regime_confidence:<pair>`, `stale_data:<source>`. Add new codes only when adding to this skill file first.
- **`notes`** is neutral commentary, ≤ 500 chars. **No** directional advice, no trade calls.
- **`citations`** — every numeric field referenced in `surprise_vs_expectation`, every event in `scheduled_events`, and every sentiment score must have at least one corresponding citation in this list.

## What output is rejected

The orchestrator validates the JSON against the schema and rejects the run if:

- Schema validation fails.
- Any citation referenced in another field is missing from `citations`.
- `notes` contains language matching the forbidden patterns (regex on "should buy", "should sell", "long bias", "short bias", etc. — the orchestrator keeps an updateable list).
- `scheduled_events` is not strictly sorted by `start_time`.
- `as_of` is more than `lookback_days + 1` days behind `now`.

## Checklist for the agent

- [ ] Every pair in `watchlist` appears in `regime`, `regime_confidence`, and `sentiment`.
- [ ] Every datum in the body has a backing citation.
- [ ] `risk_flags` includes a low-confidence flag for any `regime_confidence < 0.5`.
- [ ] `notes` ≤ 500 chars and free of directional language.
- [ ] `schema_version` is exactly `"1.0"`.
