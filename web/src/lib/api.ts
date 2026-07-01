// Typed client for the read-only dashboard API. Same-origin "/api/*" in dev
// (Vite proxies to the FastAPI service — see vite.config.ts); override with
// VITE_API_BASE for other setups.
import type {
  ApprovalItem,
  BacktestDetail,
  CandidateEconomics,
  CycleDetail,
  CycleSummary,
  Health,
  RegistryEntry,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function get<T>(path: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { headers: { accept: "application/json" } });
  } catch (cause) {
    throw new ApiError(0, `Could not reach the API at ${BASE}. Is uvicorn running?`);
  }
  if (!res.ok) {
    throw new ApiError(res.status, `Request to ${path} failed (${res.status}).`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => get<Health>("/health"),
  cycles: () => get<CycleSummary[]>("/cycles"),
  cycle: (id: string) => get<CycleDetail>(`/cycles/${encodeURIComponent(id)}`),
  registry: () => get<RegistryEntry[]>("/registry"),
  approvalQueue: () => get<ApprovalItem[]>("/approval-queue"),
  backtest: (configHash: string) =>
    get<BacktestDetail>(`/backtests/${encodeURIComponent(configHash)}`),
  economics: () => get<CandidateEconomics[]>("/economics"),
};
