"""Per-agent token usage and cache-hit accounting.

Every LLM call routed through :mod:`core.llm_client` writes one row here.
We use this to (a) verify caching is actually working (``cache_hit_ratio``
over time) and (b) attribute spend to specific agents and tiers.

Schema mirrors OpenAI's usage shape: ``prompt_tokens``, ``cached_tokens``
(subset of prompt), and ``completion_tokens``. ``cache_hit_ratio`` is
denormalised at write time so dashboards can read it without doing the
division on every aggregate.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from core.llm_client import TokenUsage

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,                    -- ISO-8601, UTC
    agent_name TEXT NOT NULL,
    run_id TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL,          -- subset of prompt_tokens
    completion_tokens INTEGER NOT NULL,
    cache_hit_ratio REAL NOT NULL,           -- denormalised cached / prompt
    extra_json TEXT NOT NULL DEFAULT '{}'    -- free-form per-call metadata
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_agent_ts
    ON llm_calls (agent_name, ts_utc);
CREATE INDEX IF NOT EXISTS idx_llm_calls_run
    ON llm_calls (run_id);
"""


@dataclass(frozen=True, slots=True)
class AgentRollup:
    """Aggregate stats for one agent over a time window. Used for dashboards."""

    agent_name: str
    call_count: int
    total_prompt_tokens: int
    total_cached_tokens: int
    total_completion_tokens: int
    weighted_cache_hit_ratio: float


class SQLiteUsageRecorder:
    """Persist one row per LLM call.

    Concurrency: SQLite's default journal mode is fine for our single-process
    slow loop. If/when we parallelise agents, switch to WAL mode at connection
    time. The :class:`core.llm_client.UsageRecorder` Protocol is the interface
    other callers depend on, so swapping implementations is a one-line change.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def record(
        self,
        *,
        agent_name: str,
        run_id: str,
        model: str,
        usage: TokenUsage,
        extra: dict[str, str],
    ) -> None:
        row = (
            datetime.now(UTC).isoformat(),
            agent_name,
            run_id,
            model,
            usage.prompt_tokens,
            usage.cached_tokens,
            usage.completion_tokens,
            usage.cache_hit_ratio,
            json.dumps(extra, sort_keys=True),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_calls
                (ts_utc, agent_name, run_id, model,
                 prompt_tokens, cached_tokens, completion_tokens,
                 cache_hit_ratio, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            conn.commit()

    def rollup_by_agent(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[AgentRollup]:
        """Aggregate per-agent stats over an optional time window."""
        clauses: list[str] = []
        params: list[str] = []
        if since is not None:
            clauses.append("ts_utc >= ?")
            params.append(since.astimezone(UTC).isoformat())
        if until is not None:
            clauses.append("ts_utc < ?")
            params.append(until.astimezone(UTC).isoformat())
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        # The only f-string interpolation is `where`, built from a closed set
        # of literal clause strings — values go through `?` params. Hence
        # the S608 suppression: no untrusted input reaches the SQL.
        sql = f"""
            SELECT
                agent_name,
                COUNT(*),
                SUM(prompt_tokens),
                SUM(cached_tokens),
                SUM(completion_tokens),
                CASE
                    WHEN SUM(prompt_tokens) = 0 THEN 0.0
                    ELSE CAST(SUM(cached_tokens) AS REAL)
                         / CAST(SUM(prompt_tokens) AS REAL)
                END
            FROM llm_calls
            {where}
            GROUP BY agent_name
            ORDER BY SUM(prompt_tokens) DESC
        """  # noqa: S608
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            AgentRollup(
                agent_name=r[0],
                call_count=int(r[1]),
                total_prompt_tokens=int(r[2] or 0),
                total_cached_tokens=int(r[3] or 0),
                total_completion_tokens=int(r[4] or 0),
                weighted_cache_hit_ratio=float(r[5] or 0.0),
            )
            for r in rows
        ]
