# Web dashboard — two-layer architecture

A personal, **read-only** view of the 4xPrima paper-research system. Single-user
today, but structured so auth / multi-tenancy can slot in at the API boundary
later without touching `core/`. This slice only **reads and displays** — there is
no mutation, no orchestrator run, no champion promotion, no kill-switch/RiskConfig
access, and no trading.

## The two layers

```
            ┌────────────────────────┐        ┌──────────────────────────┐
 browser ──▶│ web/  (React + Vite)   │ ──/api─▶│ api/  (FastAPI, read-only)│──▶ core/ models
            │  typed client, 4 views │        │  serializes core verbatim │    + data/orchestration
            └────────────────────────┘        └──────────────────────────┘
```

- **`api/`** — a FastAPI service that **imports `core/`** and serializes the
  existing frozen pydantic models to JSON. It does **not** duplicate or recompute
  any logic: numbers pass through verbatim (every model via `model_dump(mode="json")`,
  so `Decimal` → string, never re-floated). Every route is a GET; there is no
  endpoint that mutates state, runs the orchestrator, promotes a champion, touches
  RiskConfig / the kill switch, or trades. (Same integrity principle as the agents.)
- **`web/`** — a React + TypeScript + Vite + Tailwind SPA in its own subtree with
  its own `package.json`. It is **not** part of the Python quality gate;
  `scripts/check.sh` stays Python-only.

## Where the data comes from

The orchestrator and reporting CLIs persist under `data/orchestration/`
(gitignored runtime state):

| File / dir | Written by | Read by API |
| --- | --- | --- |
| `registry.json` | `ChampionChallengerRegistry` | `/registry`, `/backtests/{hash}` |
| `approval_queue.json` | `ApprovalQueue` | `/approval-queue` |
| `cycles/<id>.json` | `scripts/run_orchestrator.py` | `/cycles`, `/cycles/{id}` |
| `reports/<id>.json` | `scripts/run_reporting.py` | `/approval-queue` (framing) |

Day one there is **zero** data — the API returns empty lists / 404s and the UI
shows designed empty states. Produce some data with:

```bash
python -m scripts.run_orchestrator                       # writes a cycle + registry + queue
python -m scripts.run_reporting --cycle data/orchestration/cycles/<id>.json   # optional framing
```

## Endpoints (all read-only GET)

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/health` | liveness `{status, mode: "read-only", paper_only: true}` |
| GET | `/cycles` | past `CycleResult` summaries, newest first |
| GET | `/cycles/{cycle_id}` | full cycle detail (per-stage cost, counts, duration) |
| GET | `/registry` | champion/challenger entries (params, IS+OOS metrics, critic verdict + concerns) |
| GET | `/approval-queue` | pending entries + the reporting-agent framing where a report exists |
| GET | `/backtests/{config_hash}` | stored `BacktestEvidence` (in-sample + OOS), found by either hash |

### A note on the equity curve

`/backtests/{config_hash}` returns `equity_curve: []` with
`equity_curve_available: false` and a notice. The per-bar curve lives on the
engine's `BacktestResult`, which is **not persisted** — only the slim
`BacktestEvidence` summary is. The API must not re-run the backtest to re-derive
it, so the curve is honestly reported as unavailable and the in-sample-vs-OOS
comparison is the visualization centerpiece instead.

## Running both locally

Two terminals from the repo root.

**1. API** (Python venv with the `api` extra: `pip install -e '.[api]'`):

```bash
uvicorn api.main:app --reload --port 8000
# health check:
curl -s localhost:8000/health
```

**2. Frontend:**

```bash
cd web
npm install
npm run dev        # http://localhost:5173, proxies /api -> :8000
```

CORS is configured for the Vite dev origins only (local dev). For a non-default
API location, set `VITE_API_BASE`.

## Tests & gates

- **Python:** `scripts/check.sh` (unchanged, Python-only) + `pytest` (the core
  suite plus `api/tests`, which assert serialization and that **no mutating
  endpoint exists**).
- **Web:** `cd web && npm run build` (type-check + build) and `npm run test`
  (Vitest render tests for all four tabs incl. empty and error states).
