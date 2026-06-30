// In-sample vs out-of-sample comparison — the honest centerpiece. The equity
// curve isn't persisted in BacktestEvidence (and the API must not re-derive it),
// so the decisive visualization here is the IS→OOS decay: a strategy that only
// works in-sample is curve-fit. We never style a flattering OOS number as good;
// a tiny trade count is flagged with equal weight.
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { pct, num, ratioOrNull } from "@/lib/format";
import type { Metrics } from "@/lib/types";

const IS_COLOR = "hsl(217, 30%, 62%)"; // slate — in-sample
const OOS_COLOR = "hsl(245, 80%, 66%)"; // accent — out-of-sample (the decisive slice)

function val(n: number | null | undefined): number | null {
  return n == null || !Number.isFinite(n) ? null : n;
}

export function MetricComparison({
  inSample,
  outOfSample,
}: {
  inSample: Metrics | null;
  outOfSample: Metrics | null;
}) {
  const hasBoth = inSample != null && outOfSample != null;

  // Only chart unitless, comparable risk-adjusted ratios (mixing % with ratios
  // on one axis would be chartjunk). The IS→OOS drop is the story.
  const chartData =
    hasBoth && inSample && outOfSample
      ? [
          { metric: "Sharpe", in_sample: val(inSample.sharpe_ratio), out_of_sample: val(outOfSample.sharpe_ratio) },
          { metric: "Sortino", in_sample: val(inSample.sortino_ratio), out_of_sample: val(outOfSample.sortino_ratio) },
          { metric: "Profit factor", in_sample: val(inSample.profit_factor), out_of_sample: val(outOfSample.profit_factor) },
        ]
      : [];

  const oosTrades = outOfSample?.trade_count ?? null;
  const thinOos = oosTrades != null && oosTrades < 30;

  return (
    <div className="flex flex-col gap-5">
      {hasBoth ? (
        <figure className="rounded-lg border border-border bg-surface p-4">
          <figcaption className="mb-3 flex items-center justify-between">
            <span className="text-sm font-medium text-foreground">
              Risk-adjusted performance · in-sample vs out-of-sample
            </span>
            <span className="text-2xs text-muted-foreground">higher = better; watch the OOS drop</span>
          </figcaption>
          <div className="h-56 w-full" data-testid="is-oos-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 8, right: 8, bottom: 4, left: -16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(240,4%,16%)" vertical={false} />
                <XAxis dataKey="metric" tick={{ fontSize: 11, fill: "hsl(240,5%,62%)" }} tickLine={false} axisLine={{ stroke: "hsl(240,4%,16%)" }} />
                <YAxis tick={{ fontSize: 11, fill: "hsl(240,5%,62%)" }} tickLine={false} axisLine={false} width={44} />
                <Tooltip
                  cursor={{ fill: "hsl(240,5%,11%)" }}
                  contentStyle={{
                    background: "hsl(240,5%,11%)",
                    border: "1px solid hsl(240,4%,16%)",
                    borderRadius: 8,
                    fontSize: 12,
                    color: "hsl(0,0%,95%)",
                  }}
                  formatter={(v: number, name: string) => [
                    v == null ? "n/a" : v.toFixed(2),
                    name === "in_sample" ? "In-sample" : "Out-of-sample",
                  ]}
                />
                <Legend
                  formatter={(v) => (v === "in_sample" ? "In-sample" : "Out-of-sample")}
                  wrapperStyle={{ fontSize: 11, color: "hsl(240,5%,62%)" }}
                />
                <Bar dataKey="in_sample" name="in_sample" fill={IS_COLOR} radius={[3, 3, 0, 0]} maxBarSize={36}>
                  {chartData.map((_, i) => (
                    <Cell key={i} fill={IS_COLOR} />
                  ))}
                </Bar>
                <Bar dataKey="out_of_sample" name="out_of_sample" fill={OOS_COLOR} radius={[3, 3, 0, 0]} maxBarSize={36} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </figure>
      ) : null}

      {thinOos && (
        <p className="rounded-md border-l-2 border-survived/70 bg-survived/[0.06] px-3 py-2 text-xs text-survived">
          Out-of-sample is based on {oosTrades} trade{oosTrades === 1 ? "" : "s"} — limited
          statistical power. A flattering OOS figure on a tiny sample is not validation.
        </p>
      )}

      <Table>
        <THead>
          <TR>
            <TH>Metric</TH>
            <TH className="text-right">In-sample</TH>
            <TH className="text-right">Out-of-sample</TH>
          </TR>
        </THead>
        <TBody>
          <MetricRow label="Total return" is={inSample && pct(inSample.total_return_pct)} oos={outOfSample && pct(outOfSample.total_return_pct)} />
          <MetricRow label="Annualised return" is={inSample && pct(inSample.annualised_return_pct)} oos={outOfSample && pct(outOfSample.annualised_return_pct)} />
          <MetricRow label="Sharpe ratio" is={inSample && num(inSample.sharpe_ratio)} oos={outOfSample && num(outOfSample.sharpe_ratio)} />
          <MetricRow label="Sortino ratio" is={inSample && ratioOrNull(inSample.sortino_ratio)} oos={outOfSample && ratioOrNull(outOfSample.sortino_ratio)} />
          <MetricRow label="Profit factor" is={inSample && ratioOrNull(inSample.profit_factor)} oos={outOfSample && ratioOrNull(outOfSample.profit_factor)} />
          <MetricRow label="Max drawdown" is={inSample && pct(inSample.max_drawdown_pct)} oos={outOfSample && pct(outOfSample.max_drawdown_pct)} />
          <MetricRow label="Win rate" is={inSample && pct(inSample.win_rate)} oos={outOfSample && pct(outOfSample.win_rate)} />
          <MetricRow label="Trades" is={inSample && String(inSample.trade_count)} oos={outOfSample && String(outOfSample.trade_count)} />
          <MetricRow label="Exposure" is={inSample && pct(inSample.exposure_pct)} oos={outOfSample && pct(outOfSample.exposure_pct)} />
        </TBody>
      </Table>
    </div>
  );
}

function MetricRow({
  label,
  is,
  oos,
}: {
  label: string;
  is: string | null;
  oos: string | null;
}) {
  return (
    <TR>
      <TD className="text-muted-foreground">{label}</TD>
      <TD className="text-right tnum text-foreground">{is ?? "—"}</TD>
      <TD className="text-right tnum text-foreground">{oos ?? "—"}</TD>
    </TR>
  );
}
