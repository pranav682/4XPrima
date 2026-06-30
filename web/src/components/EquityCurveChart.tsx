// The forex-trader centerpiece: one continuous equity curve from the start of
// the in-sample window through into the sealed out-of-sample slice, with the OOS
// region clearly demarcated, a break-even reference at the starting balance, and
// the amount earned per window annotated. Curve values are charted as numbers
// (a visual), while every annotation reads from the VERBATIM string the API
// served — nothing is recomputed.
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { usd } from "@/lib/format";
import type { BacktestArtifact } from "@/lib/types";

const IS_COLOR = "hsl(217, 30%, 62%)"; // slate — in-sample
const OOS_COLOR = "hsl(245, 80%, 66%)"; // accent — the sealed out-of-sample slice

interface Row {
  x: number;
  is: number | null;
  oos: number | null;
  equity: number;
  time: string;
  seg: "in-sample" | "out-of-sample";
}

function buildRows(is: BacktestArtifact | null, oos: BacktestArtifact | null) {
  const rows: Row[] = [];
  const isCurve = is?.equity_curve ?? [];
  const oosCurve = oos?.equity_curve ?? [];
  isCurve.forEach((p, i) => {
    const e = Number(p.equity);
    rows.push({ x: i, is: e, oos: null, equity: e, time: p.time, seg: "in-sample" });
  });
  const offset = isCurve.length;
  const lastIs = isCurve.length ? Number(isCurve[isCurve.length - 1].equity) : null;
  oosCurve.forEach((p, i) => {
    const e = Number(p.equity);
    rows.push({
      x: offset + i,
      // bridge the first OOS point to the last IS point for a continuous line
      is: i === 0 ? lastIs : null,
      oos: e,
      equity: e,
      time: p.time,
      seg: "out-of-sample",
    });
  });
  return { rows, boundary: offset };
}

export function EquityCurveChart({
  inSample,
  outOfSample,
}: {
  inSample: BacktestArtifact | null;
  outOfSample: BacktestArtifact | null;
}) {
  const { rows, boundary } = buildRows(inSample, outOfSample);
  if (rows.length === 0) return null;
  const startBalance = Number(
    inSample?.starting_balance ?? outOfSample?.starting_balance ?? "0",
  );

  return (
    <div className="h-72 w-full" data-testid="equity-curve">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
          <defs>
            <linearGradient id="isFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={IS_COLOR} stopOpacity={0.35} />
              <stop offset="100%" stopColor={IS_COLOR} stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="oosFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={OOS_COLOR} stopOpacity={0.4} />
              <stop offset="100%" stopColor={OOS_COLOR} stopOpacity={0.03} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(240,4%,16%)" vertical={false} />
          <XAxis
            dataKey="x"
            tick={{ fontSize: 11, fill: "hsl(240,5%,62%)" }}
            tickLine={false}
            axisLine={{ stroke: "hsl(240,4%,16%)" }}
            label={{ value: "bars", position: "insideBottomRight", offset: -2, fontSize: 10, fill: "hsl(240,5%,50%)" }}
          />
          <YAxis
            tick={{ fontSize: 11, fill: "hsl(240,5%,62%)" }}
            tickLine={false}
            axisLine={false}
            width={64}
            domain={["auto", "auto"]}
            tickFormatter={(v: number) => `$${Math.round(v).toLocaleString("en-US")}`}
          />
          <Tooltip
            cursor={{ stroke: "hsl(240,5%,40%)", strokeDasharray: "3 3" }}
            contentStyle={{
              background: "hsl(240,5%,11%)",
              border: "1px solid hsl(240,4%,16%)",
              borderRadius: 8,
              fontSize: 12,
              color: "hsl(0,0%,95%)",
            }}
            labelFormatter={(x: number) => `bar ${x}`}
            formatter={(value: number, _name, item) => [
              usd(String(value)),
              (item?.payload as Row | undefined)?.seg ?? "equity",
            ]}
          />
          {/* break-even at the starting balance */}
          <ReferenceLine
            y={startBalance}
            stroke="hsl(240,5%,45%)"
            strokeDasharray="4 4"
            label={{ value: "start", position: "left", fontSize: 10, fill: "hsl(240,5%,55%)" }}
          />
          {/* where the sealed out-of-sample slice begins */}
          {outOfSample && boundary > 0 && (
            <ReferenceLine
              x={boundary}
              stroke="hsl(245,80%,66%)"
              strokeDasharray="4 3"
              label={{
                value: "out-of-sample →",
                position: "insideTopRight",
                fontSize: 10,
                fill: "hsl(245,70%,72%)",
              }}
            />
          )}
          <Area
            type="monotone"
            dataKey="is"
            name="in-sample"
            stroke={IS_COLOR}
            strokeWidth={1.5}
            fill="url(#isFill)"
            connectNulls
            dot={false}
            isAnimationActive={false}
          />
          <Area
            type="monotone"
            dataKey="oos"
            name="out-of-sample"
            stroke={OOS_COLOR}
            strokeWidth={1.75}
            fill="url(#oosFill)"
            connectNulls
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
