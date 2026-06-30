"""API configuration — where the persisted artefacts live + CORS for local dev.

The orchestrator and reporting CLIs persist under ``data/orchestration/``:
``registry.json``, ``approval_queue.json``, ``cycles/<id>.json`` (CycleResult),
and ``reports/<id>.json`` (CycleReport). This service only READS those paths.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """Read from the environment with the ``API_`` prefix (and ``.env``)."""

    model_config = SettingsConfigDict(env_prefix="API_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data/orchestration")

    # CORS is for LOCAL DEV ONLY — the Vite dev server's default origins.
    cors_origins: tuple[str, ...] = Field(
        default=(
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        )
    )

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry.json"

    @property
    def queue_path(self) -> Path:
        return self.data_dir / "approval_queue.json"

    @property
    def cycles_dir(self) -> Path:
        return self.data_dir / "cycles"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "backtests"


__all__ = ["ApiSettings"]
