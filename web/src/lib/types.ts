// TypeScript mirror of the read-only API contract (see api/serializers.py).
// All money / ratio fields the engine stores as Decimal arrive as STRINGS — they
// pass through verbatim and are never re-floated. The UI may format them for
// display but must not recompute the underlying value.

export type CycleOutcome = "completed" | "aborted_budget" | "aborted_failure";

export interface CycleSummary {
  cycle_id: string;
  outcome: CycleOutcome;
  started_at: string;
  ended_at: string;
  duration_seconds: number;
  total_cost_usd: string;
  candidates_proposed: number;
  candidates_killed: number;
  candidates_queued: number;
  abort_reason: string | null;
}

export interface CycleDetail extends CycleSummary {
  schema_version: string;
  stage_costs_usd: Record<string, string>;
  queued_identities: string[];
}

export interface Metrics {
  total_return_pct: string;
  annualised_return_pct: string;
  sharpe_ratio: number;
  sortino_ratio: number | null;
  max_drawdown_pct: string;
  win_rate: number;
  profit_factor: number | null;
  trade_count: number;
  avg_trade_pnl: string;
  exposure_pct: number;
}

export type EvidenceSegment = "in_sample" | "out_of_sample";

export interface Evidence {
  candidate_id: string;
  config_hash: string;
  pair: string;
  segment: EvidenceSegment;
  window_start: string;
  window_end: string;
  bars_total: number;
  bars_processed: number;
  halted_due_to_kill_switch: boolean;
  halt_reason: string | null;
  n_signals_proposed: number;
  n_signals_accepted: number;
  n_signals_rejected: number;
  starting_balance: string;
  ending_equity: string;
  cost_total: string;
  metrics: Metrics;
  gates: unknown[];
  gates_all_passed: boolean;
}

export interface Concern {
  item: string;
  finding: string;
}

export type CriticVerdictKind = "kill" | "survive_for_now";

export interface CriticVerdict {
  candidate_id: string;
  in_sample_config_hash: string;
  oos_config_hash: string | null;
  in_sample_metrics: Metrics;
  out_of_sample_metrics: Metrics | null;
  verdict: CriticVerdictKind;
  concerns: Concern[];
  assessment: string;
  caveats: string;
}

export interface StrategyParam {
  name: string;
  value: string;
}

export interface Candidate {
  candidate_id: string;
  run_id: string;
  archetype: string;
  instrument: string;
  timeframe: string;
  parameters: StrategyParam[];
  parameter_ranges: unknown[];
  rationale: string;
}

export type RegistryState =
  | "proposed"
  | "backtested"
  | "killed"
  | "survived_for_now"
  | "queued_for_approval"
  | "approved"
  | "champion"
  | "live";

export interface RegistryEntry {
  identity: string;
  candidate: Candidate;
  state: RegistryState;
  run_id: string;
  in_sample_evidence: Evidence | null;
  out_of_sample_evidence: Evidence | null;
  critic_verdict: CriticVerdict | null;
  created_at: string;
  updated_at: string;
}

export interface ApprovalItem {
  entry_id: string;
  cycle_id: string;
  identity: string;
  candidate: Candidate;
  in_sample_evidence: Evidence;
  out_of_sample_evidence: Evidence | null;
  critic_verdict: CriticVerdict;
  status: "pending";
  created_at: string;
  report_explanation: string | null;
}

export interface EquityCurvePoint {
  bar_index: number;
  time: string;
  equity: string;
  drawdown_pct: string;
}

export interface BacktestArtifact {
  config_hash: string;
  candidate_id: string;
  pair: string;
  segment: EvidenceSegment;
  window_start: string;
  window_end: string;
  starting_balance: string;
  ending_balance: string;
  ending_equity: string;
  peak_equity: string;
  net_pnl: string;
  return_pct: string;
  max_drawdown_pct: string;
  trade_count: number;
  cost_total: string;
  bars_processed: number;
  halted_due_to_kill_switch: boolean;
  equity_curve: EquityCurvePoint[];
}

export interface BacktestDetail {
  config_hash: string;
  identity: string;
  state: RegistryState;
  candidate: Candidate;
  in_sample: Evidence | null;
  out_of_sample: Evidence | null;
  critic_verdict: CriticVerdict | null;
  in_sample_artifact: BacktestArtifact | null;
  out_of_sample_artifact: BacktestArtifact | null;
  equity_curve_available: boolean;
  equity_curve_notice: string;
}

export interface Health {
  status: string;
  service: string;
  mode: string;
  paper_only: boolean;
}

export type EconomicFlag = "ok" | "concern" | "retire";

export interface EconomicConcern {
  level: "concern" | "retire";
  reason: string;
}

export interface WindowEconomics {
  segment: EvidenceSegment;
  trade_count: number;
  net_pnl: string;
  cost_total: string;
  gross_pnl: string;
  return_pct: string;
  win_rate: number;
  gross_expectancy_per_trade: string;
  net_expectancy_per_trade: string | null;
  cost_per_trade: string | null;
  avg_win: string | null;
  avg_loss: string | null;
  cost_to_edge: number | null;
  cost_to_edge_label: string;
  costs_exceed_gross: boolean;
}

export interface EconomicDecay {
  in_sample_net_expectancy: string | null;
  out_of_sample_net_expectancy: string | null;
  oos_expectancy_fraction_of_is: number | null;
  oos_return_fraction_of_is: number | null;
  note: string;
}

export interface CandidateEconomics {
  config_hash: string;
  candidate_id: string;
  pair: string;
  in_sample: WindowEconomics;
  out_of_sample: WindowEconomics | null;
  decay: EconomicDecay | null;
  flag: EconomicFlag;
  concerns: EconomicConcern[];
  amortized_research_cost_usd: string | null;
}
