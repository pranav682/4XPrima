# 4xPrima web dashboard

A **read-only** React + TypeScript + Vite + Tailwind dashboard for the 4xPrima
paper-research system. It talks only to the read-only FastAPI service in `api/`.
It cannot mutate state, run the orchestrator, promote a champion, or trade.

See [`DESIGN.md`](./DESIGN.md) for the design system and the honesty-in-visuals
rules, and [`../docs/webapp.md`](../docs/webapp.md) for the full two-layer setup.

## Run locally

You need the API running first (see `../docs/webapp.md`):

```bash
# from the repo root, in your Python venv:
uvicorn api.main:app --reload --port 8000
```

Then the frontend:

```bash
cd web
npm install
npm run dev      # http://localhost:5173  (proxies /api -> http://localhost:8000)
```

The dev server proxies `/api/*` to the API, so no CORS juggling is needed. To
point at a different API, set `VITE_API_BASE` (e.g. `VITE_API_BASE=http://host:8000`).

## Scripts

| Command | What |
| --- | --- |
| `npm run dev` | Vite dev server with the `/api` proxy. |
| `npm run build` | Type-check (`tsc --noEmit`) then production build. |
| `npm run test` | Vitest render tests (jsdom). |
| `npm run lint` | `tsc --noEmit` type check. |

## Views

- **Cycles** — every slow-loop run; list + detail (proposed / killed / queued,
  cost, duration, per-stage cost).
- **Registry** — champion & challengers by config-hash identity; KILLED vs
  SURVIVED_FOR_NOW kept honestly distinct.
- **Approval queue** — survivors awaiting a human, with the critic's concerns
  shown as prominently as the metrics. No approve action in this read-only slice.
- **Backtests** — the in-sample vs out-of-sample comparison the critic judged.
  (Per-bar equity curves aren't persisted, so that panel says so honestly rather
  than fabricating a line.)
