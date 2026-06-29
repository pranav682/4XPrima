"""Champion / challenger registry — persistent, deterministic, no LLM.

Tracks every candidate the slow loop has seen, keyed by a **strategy identity**
(a hash of its behaviour-defining spec, so re-proposing the same strategy is
recognised, not duplicated). Records the candidate, its in-sample +
out-of-sample evidence, the critic's verdict, a lifecycle state, the producing
run, and timestamps. JSON-file backed (same spirit as
``core.usage_accounting``).

SAFETY WALL (structural). The orchestrator can only ever write the states in
:data:`OrchestratorWritableState`. The risk-authorizing states (``APPROVED``,
``CHAMPION``, ``LIVE``) exist in :class:`RegistryState` for reading, but THIS
build exposes **no method to set them** — promotion is a human-only Stage-4
action that is not built here. ``set_state`` is typed to the writable subset, so
``set_state(id, RegistryState.CHAMPION)`` is a type error. The orchestrator may
READ the current champion (to compare challengers) but can never write that slot.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.models import (
    BacktestEvidence,
    CriticVerdict,
    StrategyCandidate,
)


class RegistryState(StrEnum):
    """Lifecycle state of a candidate.

    The first five are the ONLY states the orchestrator can write (see
    :data:`OrchestratorWritableState`). The last three authorize risk and are
    reserved for a human / Stage-4 action that this build does not implement.
    """

    PROPOSED = "proposed"
    BACKTESTED = "backtested"
    KILLED = "killed"
    SURVIVED_FOR_NOW = "survived_for_now"
    QUEUED_FOR_APPROVAL = "queued_for_approval"
    # --- risk-authorizing; NOT settable by the orchestrator ---
    APPROVED = "approved"
    CHAMPION = "champion"
    LIVE = "live"


# The subset the orchestrator is allowed to write. mypy rejects passing any
# risk-authorizing state to set_state(); that is the structural wall.
OrchestratorWritableState = Literal[
    RegistryState.PROPOSED,
    RegistryState.BACKTESTED,
    RegistryState.KILLED,
    RegistryState.SURVIVED_FOR_NOW,
    RegistryState.QUEUED_FOR_APPROVAL,
]

# Runtime belt-and-braces for the mypy wall above: even a runtime call with a
# risk-authorizing state is rejected, so promotion can never happen by accident.
_WRITABLE_STATES: frozenset[RegistryState] = frozenset(
    {
        RegistryState.PROPOSED,
        RegistryState.BACKTESTED,
        RegistryState.KILLED,
        RegistryState.SURVIVED_FOR_NOW,
        RegistryState.QUEUED_FOR_APPROVAL,
    }
)


def _require_writable(state: RegistryState) -> None:
    if state not in _WRITABLE_STATES:
        raise ValueError(
            f"{state.value!r} is not orchestrator-writable; promoting a candidate "
            "to a risk-authorizing state is a human-only Stage-4 action"
        )


def strategy_identity(candidate: StrategyCandidate) -> str:
    """A stable hash of the candidate's behaviour-defining spec.

    Identity = (archetype, instrument, timeframe, concrete parameters). The
    optimizer's ``parameter_ranges`` are deliberately excluded — they are the
    sandbox, not the identity. Re-proposing the same strategy yields the same
    identity, so the registry recognises it instead of duplicating it.
    """
    payload = {
        "archetype": candidate.archetype.value,
        "instrument": candidate.instrument,
        "timeframe": candidate.timeframe.value,
        "parameters": {k: str(v) for k, v in sorted(candidate.params_as_dict().items())},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class RegistryEntry(BaseModel):
    """One candidate's full record. Frozen; updates produce a new entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity: str
    candidate: StrategyCandidate
    state: RegistryState
    run_id: str
    in_sample_evidence: BacktestEvidence | None = None
    out_of_sample_evidence: BacktestEvidence | None = None
    critic_verdict: CriticVerdict | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware UTC")
        return v


class _Store(BaseModel):
    """On-disk shape of the registry file."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    entries: dict[str, RegistryEntry] = Field(default_factory=dict)


class ChampionChallengerRegistry:
    """JSON-file-backed registry. Single-process (the slow loop is one process).

    Every write loads, mutates, and atomically re-writes the file — simple and
    correct for the slow loop's low write rate.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._save(_Store())

    # ------------------------------------------------------------- load/save

    def _load(self) -> _Store:
        if not self._path.exists():
            return _Store()
        return _Store.model_validate_json(self._path.read_text())

    def _save(self, store: _Store) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(store.model_dump(mode="json"), indent=2, default=str))
        os.replace(tmp, self._path)  # atomic on POSIX

    # ----------------------------------------------------------- orchestrator API

    def upsert_proposed(
        self, candidate: StrategyCandidate, *, run_id: str, now: datetime | None = None
    ) -> str:
        """Record a freshly proposed candidate (or refresh an existing one to
        ``PROPOSED`` for a new run). Returns its identity."""
        ts = now or datetime.now(UTC)
        identity = strategy_identity(candidate)
        store = self._load()
        existing = store.entries.get(identity)
        created = existing.created_at if existing is not None else ts
        store.entries[identity] = RegistryEntry(
            identity=identity,
            candidate=candidate,
            state=RegistryState.PROPOSED,
            run_id=run_id,
            created_at=created,
            updated_at=ts,
        )
        self._save(store)
        return identity

    def record_backtest(
        self, identity: str, evidence: BacktestEvidence, *, now: datetime | None = None
    ) -> None:
        self._mutate(
            identity,
            now=now,
            updates={"in_sample_evidence": evidence, "state": RegistryState.BACKTESTED},
        )

    def record_critic(
        self,
        identity: str,
        verdict: CriticVerdict,
        *,
        state: OrchestratorWritableState,
        out_of_sample_evidence: BacktestEvidence | None = None,
        now: datetime | None = None,
    ) -> None:
        _require_writable(state)
        self._mutate(
            identity,
            now=now,
            updates={
                "critic_verdict": verdict,
                "out_of_sample_evidence": out_of_sample_evidence,
                "state": state,
            },
        )

    def set_state(
        self, identity: str, state: OrchestratorWritableState, *, now: datetime | None = None
    ) -> None:
        """Set lifecycle state. Typed to the writable subset (mypy wall) AND
        guarded at runtime — the risk-authorizing states cannot be set here."""
        _require_writable(state)
        self._mutate(identity, now=now, updates={"state": state})

    # ------------------------------------------------------------- reads

    def get(self, identity: str) -> RegistryEntry | None:
        return self._load().entries.get(identity)

    def all_entries(self) -> tuple[RegistryEntry, ...]:
        return tuple(self._load().entries.values())

    def current_champion(self) -> RegistryEntry | None:
        """READ the current champion (state == CHAMPION), if any. The
        orchestrator may compare challengers against it but never WRITES it —
        there is no champion-setting method in this build."""
        for entry in self._load().entries.values():
            if entry.state == RegistryState.CHAMPION:
                return entry
        return None

    # ------------------------------------------------------------- internals

    def _mutate(self, identity: str, *, now: datetime | None, updates: dict[str, object]) -> None:
        ts = now or datetime.now(UTC)
        store = self._load()
        entry = store.entries.get(identity)
        if entry is None:
            raise KeyError(f"no registry entry for identity {identity!r}")
        store.entries[identity] = entry.model_copy(update={**updates, "updated_at": ts})
        self._save(store)


__all__ = [
    "ChampionChallengerRegistry",
    "OrchestratorWritableState",
    "RegistryEntry",
    "RegistryState",
    "strategy_identity",
]
