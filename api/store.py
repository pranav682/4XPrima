"""Read-only access to the orchestrator's persisted artefacts.

Every method READS a file and returns the existing frozen ``core`` models. It
never writes (not even the registry's lazy-create — we parse the JSON directly),
never recomputes, and never mutates. Missing files yield empty results, which is
the honest day-one state.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from api.config import ApiSettings
from core.analysis.pair_screener import ScreeningReport
from core.models import BacktestArtifact, CycleReport
from core.orchestration import (
    ApprovalQueueEntry,
    BacktestArtifactStore,
    CycleResult,
    RegistryEntry,
    RegistryState,
)


class DataStore:
    """Read-only reader over ``data/orchestration``. Construct once per app."""

    def __init__(self, settings: ApiSettings) -> None:
        self._s = settings

    # ------------------------------------------------------------- cycles

    def list_cycles(self) -> list[CycleResult]:
        """All persisted cycle results, newest first."""
        cycles: list[CycleResult] = []
        for path in self._json_files(self._s.cycles_dir):
            parsed = self._try(path, CycleResult)
            if parsed is not None:
                cycles.append(parsed)
        cycles.sort(key=lambda c: c.started_at, reverse=True)
        return cycles

    def get_cycle(self, cycle_id: str) -> CycleResult | None:
        for cycle in self.list_cycles():
            if cycle.cycle_id == cycle_id:
                return cycle
        return None

    # ------------------------------------------------------------- registry

    def registry_entries(self) -> list[RegistryEntry]:
        """All champion/challenger entries, most recently updated first."""
        data = self._load_json(self._s.registry_path)
        if data is None:
            return []
        raw_entries = data.get("entries", {})
        entries = [RegistryEntry.model_validate(v) for v in raw_entries.values()]
        entries.sort(key=lambda e: e.updated_at, reverse=True)
        return entries

    def champion(self) -> RegistryEntry | None:
        for entry in self.registry_entries():
            if entry.state == RegistryState.CHAMPION:
                return entry
        return None

    # ------------------------------------------------------------- approval queue

    def pending_queue(self) -> list[ApprovalQueueEntry]:
        data = self._load_json(self._s.queue_path)
        if data is None:
            return []
        entries = [ApprovalQueueEntry.model_validate(v) for v in data.get("entries", [])]
        pending = [e for e in entries if e.status == "pending"]
        pending.sort(key=lambda e: e.created_at, reverse=True)
        return pending

    # ------------------------------------------------------------- reports

    def saved_reports(self) -> dict[str, CycleReport]:
        """Saved CycleReports keyed by cycle_id (for the reporting-agent framing
        of queued candidates, where one was generated)."""
        reports: dict[str, CycleReport] = {}
        for path in self._json_files(self._s.reports_dir):
            parsed = self._try(path, CycleReport)
            if parsed is not None:
                reports[parsed.cycle_id] = parsed
        return reports

    # ------------------------------------------------------------- universe

    def screening_report(self) -> ScreeningReport | None:
        """The persisted structural pair-screen, if one was run."""
        path = self._s.universe_path
        if not path.is_file():
            return None
        try:
            return ScreeningReport.model_validate_json(path.read_text())
        except (ValueError, OSError):
            return None

    # ------------------------------------------------------------- backtests

    def artifact(self, config_hash: str) -> BacktestArtifact | None:
        """The rich equity-curve artifact for a run, if one was persisted."""
        return BacktestArtifactStore(self._s.artifacts_dir).get(config_hash)

    def amortized_research_cost(self, run_id: str) -> Decimal | None:
        """The cycle's LLM cost (verbatim from its CycleResult) spread across the
        candidates it backtested — the ~flat fixed overhead, independent of
        capital or trade volume. None if the cycle isn't persisted."""
        cycle = self.get_cycle(run_id)
        if cycle is None:
            return None
        backtested = [
            e
            for e in self.registry_entries()
            if e.run_id == run_id and e.in_sample_evidence is not None
        ]
        if not backtested:
            return None
        return cycle.total_cost_usd / Decimal(len(backtested))

    def find_evidence(self, config_hash: str) -> RegistryEntry | None:
        """The registry entry whose in-sample OR out-of-sample evidence carries
        this config_hash. Both segments are returned together so the UI can show
        the in-sample-vs-OOS comparison."""
        for entry in self.registry_entries():
            ins = entry.in_sample_evidence
            oos = entry.out_of_sample_evidence
            if (ins is not None and ins.config_hash == config_hash) or (
                oos is not None and oos.config_hash == config_hash
            ):
                return entry
        return None

    # ------------------------------------------------------------- internals

    @staticmethod
    def _json_files(directory: Path) -> list[Path]:
        if not directory.is_dir():
            return []
        return sorted(directory.glob("*.json"))

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        loaded = json.loads(path.read_text())
        return loaded if isinstance(loaded, dict) else None

    @staticmethod
    def _try(path: Path, model: type[Any]) -> Any | None:
        """Parse a file as ``model``; skip files that don't match (the artefact
        directories may hold unrelated JSON)."""
        try:
            return model.model_validate_json(path.read_text())
        except (ValueError, OSError):
            return None


__all__ = ["DataStore"]
