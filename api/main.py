"""FastAPI app — READ-ONLY dashboard endpoints over the orchestrator artefacts.

Every route is a GET. There is deliberately no endpoint that mutates state, runs
the orchestrator, promotes a champion, touches RiskConfig / the kill switch, or
trades. Run locally with::

    uvicorn api.main:app --reload --port 8000

See ``docs/webapp.md`` for the full two-layer setup.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.config import ApiSettings
from api.serializers import (
    approval_item,
    backtest_detail,
    cycle_detail,
    cycle_summary,
    registry_entry,
)
from api.store import DataStore
from core.analysis.economics import candidate_economics
from core.orchestration import RegistryEntry


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings()
    store = DataStore(settings)

    app = FastAPI(
        title="4xPrima Dashboard API",
        version="0.1.0",
        summary="Read-only view of the 4xPrima paper-research system. No mutations, no trading.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET"],  # read-only: only GET is permitted
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "4xprima-dashboard-api",
            "mode": "read-only",
            "paper_only": True,
        }

    @app.get("/cycles")
    def list_cycles() -> list[dict[str, Any]]:
        return [cycle_summary(c) for c in store.list_cycles()]

    @app.get("/cycles/{cycle_id}")
    def get_cycle(cycle_id: str) -> dict[str, Any]:
        cycle = store.get_cycle(cycle_id)
        if cycle is None:
            raise HTTPException(status_code=404, detail=f"no cycle {cycle_id!r}")
        return cycle_detail(cycle)

    @app.get("/registry")
    def get_registry() -> list[dict[str, Any]]:
        return [registry_entry(e) for e in store.registry_entries()]

    @app.get("/approval-queue")
    def get_approval_queue() -> list[dict[str, Any]]:
        reports = store.saved_reports()
        return [approval_item(e, reports.get(e.cycle_id)) for e in store.pending_queue()]

    def _economics(entry: RegistryEntry) -> dict[str, Any]:
        # entry.in_sample_evidence is guaranteed non-None by the callers below.
        assert entry.in_sample_evidence is not None
        return candidate_economics(
            entry.in_sample_evidence,
            entry.out_of_sample_evidence,
            amortized_research_cost_usd=store.amortized_research_cost(entry.run_id),
        ).model_dump(mode="json")

    @app.get("/economics")
    def list_economics() -> list[dict[str, Any]]:
        """Net-of-cost economics + historical (IS->OOS) decay read per candidate.
        HISTORICAL backtest economics only — not live/forward-test monitoring."""
        return [_economics(e) for e in store.registry_entries() if e.in_sample_evidence is not None]

    @app.get("/economics/{config_hash}")
    def get_economics(config_hash: str) -> dict[str, Any]:
        entry = store.find_evidence(config_hash)
        if entry is None or entry.in_sample_evidence is None:
            raise HTTPException(status_code=404, detail=f"no economics for {config_hash!r}")
        return _economics(entry)

    @app.get("/backtests/{config_hash}")
    def get_backtest(config_hash: str) -> dict[str, Any]:
        entry = store.find_evidence(config_hash)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"no backtest evidence for {config_hash!r}")
        ins = entry.in_sample_evidence
        oos = entry.out_of_sample_evidence
        return backtest_detail(
            config_hash,
            entry,
            in_sample_artifact=store.artifact(ins.config_hash) if ins is not None else None,
            out_of_sample_artifact=store.artifact(oos.config_hash) if oos is not None else None,
        )

    return app


app = create_app()

__all__ = ["app", "create_app"]
