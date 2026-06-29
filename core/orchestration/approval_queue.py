"""Human approval queue — persistent, append-only from the orchestrator.

When a candidate survives the critic, the orchestrator APPENDS a ``pending``
entry here (candidate + in-sample + out-of-sample evidence + critic verdict +
timestamp). That is the **terminus** of the slow loop: a human, in a deferred
Stage-4 surface, reads the queue and decides. The orchestrator NEVER marks an
entry approved or rejected — this build exposes no such method. Append-only.

This is the hard wall against auto-promotion: routing to this queue is the
strongest thing the orchestrator can do; it cannot promote, deploy, or trade.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.models import BacktestEvidence, CriticVerdict, StrategyCandidate


class ApprovalQueueEntry(BaseModel):
    """One candidate awaiting a human decision. Always ``pending`` when written
    by the orchestrator; the ``status`` field exists for the Stage-4 human
    surface to set later (not in this build)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_id: str
    cycle_id: str
    identity: str
    candidate: StrategyCandidate
    in_sample_evidence: BacktestEvidence
    out_of_sample_evidence: BacktestEvidence | None
    critic_verdict: CriticVerdict
    status: Literal["pending"] = "pending"
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_at must be timezone-aware UTC")
        return v


class _Store(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    entries: list[ApprovalQueueEntry] = Field(default_factory=list)


class ApprovalQueue:
    """JSON-file-backed, append-only approval queue.

    The orchestrator can only :meth:`append`. There is deliberately NO method to
    approve, reject, or remove an entry — deciding is the deferred Stage-4 human
    action.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._save(_Store())

    def _load(self) -> _Store:
        if not self._path.exists():
            return _Store()
        return _Store.model_validate_json(self._path.read_text())

    def _save(self, store: _Store) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(store.model_dump(mode="json"), indent=2, default=str))
        os.replace(tmp, self._path)

    def append(
        self,
        *,
        cycle_id: str,
        identity: str,
        candidate: StrategyCandidate,
        in_sample_evidence: BacktestEvidence,
        critic_verdict: CriticVerdict,
        out_of_sample_evidence: BacktestEvidence | None = None,
        now: datetime | None = None,
    ) -> str:
        """Append a pending entry for human review. Returns the entry id.

        Idempotent per (cycle, identity): re-appending the same candidate in the
        same cycle replaces nothing and adds nothing duplicate — the entry_id is
        derived from cycle_id + identity, and a matching id is left untouched."""
        ts = now or datetime.now(UTC)
        entry_id = f"{cycle_id}:{identity}"
        store = self._load()
        if any(e.entry_id == entry_id for e in store.entries):
            return entry_id
        store.entries.append(
            ApprovalQueueEntry(
                entry_id=entry_id,
                cycle_id=cycle_id,
                identity=identity,
                candidate=candidate,
                in_sample_evidence=in_sample_evidence,
                out_of_sample_evidence=out_of_sample_evidence,
                critic_verdict=critic_verdict,
                created_at=ts,
            )
        )
        self._save(store)
        return entry_id

    def pending(self) -> tuple[ApprovalQueueEntry, ...]:
        """All pending entries (read-only)."""
        return tuple(e for e in self._load().entries if e.status == "pending")

    def all_entries(self) -> tuple[ApprovalQueueEntry, ...]:
        return tuple(self._load().entries)


__all__ = [
    "ApprovalQueue",
    "ApprovalQueueEntry",
]
