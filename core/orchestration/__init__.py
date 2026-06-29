"""Deterministic slow-loop orchestration — NO LLM.

The orchestrator sequences the four worker agents into one cycle, tracks
champion/challenger state, enforces a per-cycle budget, and routes survivors to
a human approval queue. It cannot promote, trade, or touch sacred config — see
:mod:`core.orchestration.orchestrator`.
"""

from core.orchestration.approval_queue import ApprovalQueue, ApprovalQueueEntry
from core.orchestration.orchestrator import (
    CycleOutcome,
    CycleResult,
    Orchestrator,
    OrchestratorConfig,
)
from core.orchestration.registry import (
    ChampionChallengerRegistry,
    RegistryEntry,
    RegistryState,
    strategy_identity,
)

__all__ = [
    "ApprovalQueue",
    "ApprovalQueueEntry",
    "ChampionChallengerRegistry",
    "CycleOutcome",
    "CycleResult",
    "Orchestrator",
    "OrchestratorConfig",
    "RegistryEntry",
    "RegistryState",
    "strategy_identity",
]
