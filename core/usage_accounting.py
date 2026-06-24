"""Per-agent token usage and cache-hit accounting.

Every LLM call routed through `core.llm_client.LLMClient` writes one row here.
We use this to (a) verify caching is actually working (`cache_hit_ratio` over
time) and (b) attribute spend to specific agents and run phases.

Backed by a small SQLite database for portability. The schema is intentionally
flat: one row per call, with denormalised model/agent/run fields so simple SQL
suffices for the dashboards we'll write later.

This is a STUB: the schema and the `record()` signature are the real contract;
the persistence path is wired but not battle-tested.
"""

from __future__ import annotations

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
    input_tokens INTEGER NOT NULL,
    cache_read_input_tokens INTEGER NOT NULL,
    cache_creation_input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_input_tokens INTEGER NOT NULL,     -- denormalised for fast SUMs
    cache_hit_ratio REAL NOT NULL,           -- denormalised for fast aggregations
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
    total_input_tokens: int
    total_cache_read_tokens: int
    total_cache_creation_tokens: int
    total_output_tokens: int
    weighted_cache_hit_ratio: float


class SQLiteUsageRecorder:
    """Persist one row per LLM call.

    Concurrency: SQLite's default journal mode is fine for our single-process
    slow loop. If/when we parallelise agents, switch to WAL mode at connection
    time. The `core.llm_client.UsageRecorder` Protocol is the interface other
    callers depend on, so swapping implementations is a one-line change.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    def _connect(self) -> sqlite3.Connection:
        # `isolation_level=None` would mean autocommit; we keep transactions
        # explicit so a crash mid-write doesn't tear a row.
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
        import json

        row = (
            datetime.now(UTC).isoformat(),
            agent_name,
            run_id,
            model,
            usage.input_tokens,
            usage.cache_read_input_tokens,
            usage.cache_creation_input_tokens,
            usage.output_tokens,
            usage.total_input_tokens,
            usage.cache_hit_ratio,
            json.dumps(extra, sort_keys=True),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_calls
                (ts_utc, agent_name, run_id, model,
                 input_tokens, cache_read_input_tokens, cache_creation_input_tokens,
                 output_tokens, total_input_tokens, cache_hit_ratio, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        """Aggregate per-agent stats over an optional time window.

        Stub-quality query — fine for the small volumes the slow loop produces.
        Optimise if we ever cross a million rows (we won't soon).
        """
        clauses: list[str] = []
        params: list[str] = []
        if since is not None:
            clauses.append("ts_utc >= ?")
            params.append(since.astimezone(UTC).isoformat())
        if until is not None:
            clauses.append("ts_utc < ?")
            params.append(until.astimezone(UTC).isoformat())
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        # The only f-string interpolation is `where`, which is built from a
        # closed set of literal clause strings — values go through `?` params.
        # Hence the ruff S608 suppression: no untrusted input reaches the SQL.
        sql = f"""
            SELECT
                agent_name,
                COUNT(*),
                SUM(total_input_tokens),
                SUM(cache_read_input_tokens),
                SUM(cache_creation_input_tokens),
                SUM(output_tokens),
                CASE
                    WHEN SUM(total_input_tokens) = 0 THEN 0.0
                    ELSE CAST(SUM(cache_read_input_tokens) AS REAL)
                         / CAST(SUM(total_input_tokens) AS REAL)
                END
            FROM llm_calls
            {where}
            GROUP BY agent_name
            ORDER BY SUM(total_input_tokens) DESC
        """  # noqa: S608
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            AgentRollup(
                agent_name=r[0],
                call_count=int(r[1]),
                total_input_tokens=int(r[2] or 0),
                total_cache_read_tokens=int(r[3] or 0),
                total_cache_creation_tokens=int(r[4] or 0),
                total_output_tokens=int(r[5] or 0),
                weighted_cache_hit_ratio=float(r[6] or 0.0),
            )
            for r in rows
        ]
