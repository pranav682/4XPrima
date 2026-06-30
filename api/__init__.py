"""Read-only web-dashboard API for 4xPrima.

A thin FastAPI service that IMPORTS the existing ``core`` packages and serves
the orchestrator's persisted artefacts (cycle results, the champion/challenger
registry, the approval queue, backtest evidence, saved reports) as JSON.

Hard rules, mirroring the trading core's ethos:

- **Read-only.** No endpoint mutates state, runs the orchestrator, promotes a
  champion, touches RiskConfig / the kill switch, or trades. Only GET routes.
- **Verbatim.** Numbers pass through exactly as persisted — every model is
  serialized with ``model_dump(mode="json")`` so ``Decimal`` becomes a string
  and is never re-floated or recomputed.
- **No duplicated logic.** The API calls into ``core`` models / stores; it never
  re-derives a metric, a curve, or a verdict.

Single-user today, but structured so auth / multi-tenancy can slot in at this
API boundary later without touching ``core``.
"""

__all__: list[str] = []
