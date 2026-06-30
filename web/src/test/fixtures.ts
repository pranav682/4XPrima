// Shared fixtures for render tests. These mirror the real API shapes (verbatim
// string Decimals); no fabricated/lorem content.
import type {
  ApprovalItem,
  BacktestDetail,
  CycleSummary,
  Metrics,
  RegistryEntry,
} from "@/lib/types";

export function metrics(over: Partial<Metrics> = {}): Metrics {
  return {
    total_return_pct: "0.30",
    annualised_return_pct: "0.30",
    sharpe_ratio: 2.1,
    sortino_ratio: 0.7,
    max_drawdown_pct: "0.08",
    win_rate: 0.55,
    profit_factor: 2.4,
    trade_count: 40,
    avg_trade_pnl: "1.2",
    exposure_pct: 0.3,
    ...over,
  };
}

export const cycleSummary: CycleSummary = {
  cycle_id: "cycle-abc123",
  outcome: "completed",
  started_at: "2026-06-01T12:00:00+00:00",
  ended_at: "2026-06-01T12:00:42+00:00",
  duration_seconds: 42.5,
  total_cost_usd: "0.1734",
  candidates_proposed: 3,
  candidates_killed: 2,
  candidates_queued: 1,
  abort_reason: null,
};

export const survivorEntry: RegistryEntry = {
  identity: "idy-survivor-1234",
  state: "queued_for_approval",
  run_id: "cycle-abc123",
  created_at: "2026-06-01T12:00:00+00:00",
  updated_at: "2026-06-01T12:05:00+00:00",
  candidate: {
    candidate_id: "cand-survivor",
    run_id: "cycle-abc123",
    archetype: "ma_crossover",
    instrument: "USDJPY",
    timeframe: "H1",
    parameters: [
      { name: "fast_period", value: "5" },
      { name: "slow_period", value: "15" },
    ],
    parameter_ranges: [],
    rationale: "trend",
  },
  in_sample_evidence: evidence("is-survivor-hash", metrics()),
  out_of_sample_evidence: evidence(
    "oos-survivor-hash",
    metrics({ sharpe_ratio: 0.3, total_return_pct: "0.02", profit_factor: 1.1, trade_count: 6 }),
    "out_of_sample",
  ),
  critic_verdict: {
    candidate_id: "cand-survivor",
    in_sample_config_hash: "is-survivor-hash",
    oos_config_hash: "oos-survivor-hash",
    in_sample_metrics: metrics(),
    out_of_sample_metrics: metrics({ sharpe_ratio: 0.3, trade_count: 6 }),
    verdict: "survive_for_now",
    concerns: [
      { item: "out_of_sample_decay", finding: "Sharpe falls 2.1 to 0.3 out-of-sample." },
      { item: "trade_count", finding: "Only 6 out-of-sample trades." },
    ],
    assessment: "Survived but the OOS sample is thin.",
    caveats: "survive_for_now is not validation",
  },
};

export const killedEntry: RegistryEntry = {
  identity: "idy-killed-5678",
  state: "killed",
  run_id: "cycle-abc123",
  created_at: "2026-06-01T12:00:00+00:00",
  updated_at: "2026-06-01T12:04:00+00:00",
  candidate: {
    candidate_id: "cand-killed",
    run_id: "cycle-abc123",
    archetype: "ma_crossover",
    instrument: "EURUSD",
    timeframe: "H1",
    parameters: [{ name: "fast_period", value: "10" }],
    parameter_ranges: [],
    rationale: "trend",
  },
  in_sample_evidence: evidence("is-killed-hash", metrics({ sharpe_ratio: -1.4, total_return_pct: "-0.15" })),
  out_of_sample_evidence: evidence(
    "oos-killed-hash",
    metrics({ sharpe_ratio: -1.9, total_return_pct: "-0.20" }),
    "out_of_sample",
  ),
  critic_verdict: {
    candidate_id: "cand-killed",
    in_sample_config_hash: "is-killed-hash",
    oos_config_hash: "oos-killed-hash",
    in_sample_metrics: metrics({ sharpe_ratio: -1.4 }),
    out_of_sample_metrics: metrics({ sharpe_ratio: -1.9 }),
    verdict: "kill",
    concerns: [{ item: "out_of_sample_decay", finding: "OOS collapse." }],
    assessment: "Hard kill.",
    caveats: "kill is the default",
  },
};

export const approvalItem: ApprovalItem = {
  entry_id: "cycle-abc123:idy-survivor-1234",
  cycle_id: "cycle-abc123",
  identity: "idy-survivor-1234",
  candidate: survivorEntry.candidate,
  in_sample_evidence: survivorEntry.in_sample_evidence!,
  out_of_sample_evidence: survivorEntry.out_of_sample_evidence,
  critic_verdict: survivorEntry.critic_verdict!,
  status: "pending",
  created_at: "2026-06-01T12:05:00+00:00",
  report_explanation: "The critic did not kill this; it remains worried about out-of-sample decay.",
};

export const backtestDetail: BacktestDetail = {
  config_hash: "is-survivor-hash",
  identity: "idy-survivor-1234",
  state: "queued_for_approval",
  candidate: survivorEntry.candidate,
  in_sample: survivorEntry.in_sample_evidence,
  out_of_sample: survivorEntry.out_of_sample_evidence,
  critic_verdict: survivorEntry.critic_verdict,
  equity_curve: [],
  equity_curve_available: false,
  equity_curve_notice:
    "Equity-curve points are not persisted in BacktestEvidence — only summary metrics are stored.",
};

function evidence(configHash: string, m: Metrics, segment: "in_sample" | "out_of_sample" = "in_sample") {
  return {
    candidate_id: "cand",
    config_hash: configHash,
    pair: "USDJPY",
    segment,
    window_start: "2026-05-01T00:00:00+00:00",
    window_end: "2026-06-01T00:00:00+00:00",
    bars_total: 100,
    bars_processed: 100,
    halted_due_to_kill_switch: false,
    halt_reason: null,
    n_signals_proposed: 10,
    n_signals_accepted: 8,
    n_signals_rejected: 2,
    starting_balance: "100000",
    ending_equity: "105000",
    cost_total: "5",
    metrics: m,
    gates: [],
    gates_all_passed: true,
  };
}
