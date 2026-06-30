"""Persistent store for rich backtest artifacts (equity curves + annotations).

Separate from the registry on purpose: the registry holds the slim, LLM-facing
:class:`~core.models.BacktestEvidence`, while these heavy
:class:`~core.models.BacktestArtifact` records (per-bar curves) are written here
and read ONLY by the read-only web dashboard. One JSON file per ``config_hash``
under ``<data_dir>/backtests/``. Same spirit as the registry/queue stores.
"""

from __future__ import annotations

import os
from pathlib import Path

from core.models import BacktestArtifact


class BacktestArtifactStore:
    """Read/write backtest artifacts, keyed by the run's ``config_hash``."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)

    def save(self, artifact: BacktestArtifact) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{artifact.config_hash}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(artifact.model_dump_json())
        os.replace(tmp, path)  # atomic on POSIX

    def get(self, config_hash: str) -> BacktestArtifact | None:
        path = self._dir / f"{config_hash}.json"
        if not path.is_file():
            return None
        try:
            return BacktestArtifact.model_validate_json(path.read_text())
        except (ValueError, OSError):
            return None

    def all_artifacts(self) -> tuple[BacktestArtifact, ...]:
        if not self._dir.is_dir():
            return ()
        out: list[BacktestArtifact] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                out.append(BacktestArtifact.model_validate_json(path.read_text()))
            except (ValueError, OSError):
                continue
        return tuple(out)


__all__ = ["BacktestArtifactStore"]
