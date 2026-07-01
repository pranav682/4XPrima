import { Scale, TriangleAlert, Info } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { AsyncBoundary, EmptyState } from "@/components/StateViews";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardContent } from "@/components/ui/card";
import { Stat } from "@/components/ui/misc";
import { EconomicFlagBadge } from "@/components/EconomicFlagBadge";
import { cn } from "@/lib/utils";
import { pct, pnlSign, shortHash, signedUsd, usd } from "@/lib/format";
import type { CandidateEconomics, WindowEconomics } from "@/lib/types";

export function EconomicsView() {
  const state = useApi<CandidateEconomics[]>(() => api.economics(), []);
  return (
    <>
      <PageHeader
        title="Economics"
        subtitle="Net-of-cost economic health per candidate. Expectancy — not win rate — is the measure, and the edge must dwarf the cost."
      />
      <ScopeNote />
      <AsyncBoundary
        state={state}
        empty={(d) => d.length === 0}
        emptyView={
          <EmptyState
            icon={<Scale className="h-5 w-5" aria-hidden />}
            title="No economics yet"
            message="Once candidates have been backtested, their net-of-cost economics and historical (in-sample → out-of-sample) decay read appear here."
          />
        }
        children={(rows) => (
          <div className="flex flex-col gap-4">
            {rows.map((r) => (
              <EconomicsCard key={r.config_hash} econ={r} />
            ))}
          </div>
        )}
      />
    </>
  );
}

function ScopeNote() {
  return (
    <div className="mb-4 flex items-start gap-2 rounded-md border border-border bg-elevated/40 px-3 py-2 text-2xs text-muted-foreground">
      <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
      <span>
        This is <strong className="text-foreground">historical backtest economics</strong>{" "}
        (in-sample vs the sealed out-of-sample slice). True <em>live</em> decay monitoring — comparing
        a running champion's forward-test results to its expected band — needs the Stage-4 approval +
        paper-forward-test, which is not built yet.
      </span>
    </div>
  );
}

function EconomicsCard({ econ }: { econ: CandidateEconomics }) {
  const flagged = econ.flag !== "ok";
  const is = econ.in_sample;
  const oos = econ.out_of_sample;
  return (
    <Card className={cn(flagged && "border-survived/30")}>
      <CardContent className="pt-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div className="flex flex-col gap-0.5">
            <span className="text-sm font-semibold text-foreground">{econ.pair}</span>
            <span className="font-mono text-2xs text-muted-foreground">
              {shortHash(econ.config_hash, 12)}
            </span>
          </div>
          <EconomicFlagBadge flag={econ.flag} />
        </div>

        {/* Net leads; gross is subordinate. */}
        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <PnlStat label="Net P&L (in-sample)" value={is.net_pnl} />
          <GrossNote window={is} />
          <PnlStat label="Net P&L (out-of-sample)" value={oos?.net_pnl ?? null} />
          {oos ? <GrossNote window={oos} /> : <Stat label="Out-of-sample">not opened</Stat>}
        </div>

        {/* Expectancy — the measure — net leads, gross subordinate, IS → OOS. */}
        <div className="mt-4 rounded-md border border-border bg-elevated/50 px-3 py-3">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <PerTradeEdge title="Per-trade edge · in-sample" window={is} />
            {oos ? (
              <PerTradeEdge title="Per-trade edge · out-of-sample" window={oos} thinSampleWarn />
            ) : (
              <div className="text-xs text-muted-foreground">
                Out-of-sample not opened for this candidate.
              </div>
            )}
          </div>
        </div>

        {/* Edge must dwarf cost. */}
        <div className="mt-4">
          <CostToEdge window={is} />
        </div>

        {/* Historical decay. */}
        {econ.decay && oos && (
          <div className="mt-4 rounded-md border-l-2 border-border bg-background/40 px-3 py-2.5">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <span className="text-2xs uppercase tracking-wide text-muted-foreground">
                Historical decay (in-sample → out-of-sample)
              </span>
              <span className="tnum text-sm text-foreground">
                {econ.decay.oos_expectancy_fraction_of_is != null
                  ? `OOS expectancy is ${(econ.decay.oos_expectancy_fraction_of_is * 100).toFixed(0)}% of in-sample`
                  : "not comparable"}
              </span>
            </div>
            {oos.trade_count < 30 && (
              <p className="mt-1.5 flex items-start gap-1 text-2xs text-survived">
                <TriangleAlert className="mt-0.5 h-3 w-3 shrink-0" aria-hidden />
                Rests on {oos.trade_count} out-of-sample trade{oos.trade_count === 1 ? "" : "s"} —
                limited statistical power; treat the comparison as noise, not proof.
              </p>
            )}
          </div>
        )}

        {/* Amortized fixed research overhead. */}
        {econ.amortized_research_cost_usd && (
          <p className="mt-3 text-2xs text-muted-foreground">
            Fixed research overhead:{" "}
            <span className="tnum text-foreground">{usd(econ.amortized_research_cost_usd)}</span> /
            candidate — roughly flat regardless of capital or trade volume (the LLM is in the research
            loop, never the trade path).
          </p>
        )}

        {econ.concerns.length > 0 && (
          <ul className="mt-3 flex flex-col gap-1.5" aria-label="Economic flags">
            {econ.concerns.map((c, i) => (
              <li
                key={i}
                className="flex items-start gap-2 rounded-md border-l-2 border-survived/70 bg-survived/[0.06] px-3 py-1.5 text-xs text-survived"
              >
                <TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
                <span>
                  <span className="font-medium uppercase">{c.level}</span> · {c.reason}
                </span>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function PnlStat({ label, value }: { label: string; value: string | null }) {
  if (value == null) return <Stat label={label}>—</Stat>;
  const sign = pnlSign(value);
  return (
    <Stat label={label} emphasis>
      <span className={cn(sign === "pos" && "text-pnlPos", sign === "neg" && "text-pnlNeg")}>
        {signedUsd(value)}
      </span>
    </Stat>
  );
}

function GrossNote({ window }: { window: WindowEconomics }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-2xs uppercase tracking-wide text-muted-foreground">Gross (before cost)</span>
      <span className="tnum text-sm text-muted-foreground">{usd(window.gross_pnl)}</span>
      <span className="tnum text-2xs text-muted-foreground">cost {usd(window.cost_total)}</span>
    </div>
  );
}

/** Win rate is shown ONLY here — always next to avg win / avg loss / expectancy.
 *  On its own it is misleading, so it never renders alone. */
function PerTradeEdge({
  title,
  window,
  thinSampleWarn,
}: {
  title: string;
  window: WindowEconomics;
  thinSampleWarn?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-2xs uppercase tracking-wide text-muted-foreground">{title}</span>
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 text-xs">
        <span className="text-muted-foreground">
          Net expectancy{" "}
          <span
            className={cn(
              "tnum font-semibold",
              window.net_expectancy_per_trade != null &&
                (pnlSign(window.net_expectancy_per_trade) === "pos"
                  ? "text-pnlPos"
                  : pnlSign(window.net_expectancy_per_trade) === "neg"
                    ? "text-pnlNeg"
                    : "text-foreground"),
            )}
          >
            {window.net_expectancy_per_trade != null
              ? signedUsd(window.net_expectancy_per_trade)
              : "—"}
          </span>
          <span className="text-muted-foreground"> /trade</span>
        </span>
        <span className="text-muted-foreground">
          Win rate <span className="tnum text-foreground">{pct(window.win_rate)}</span>
        </span>
        <span className="text-muted-foreground">
          Avg win{" "}
          <span className="tnum text-foreground">
            {window.avg_win != null ? usd(window.avg_win) : "—"}
          </span>
        </span>
        <span className="text-muted-foreground">
          Avg loss{" "}
          <span className="tnum text-foreground">
            {window.avg_loss != null ? usd(window.avg_loss) : "—"}
          </span>
        </span>
        <span className="text-muted-foreground">
          Gross exp <span className="tnum text-foreground">{usd(window.gross_expectancy_per_trade)}</span>
        </span>
      </div>
      {thinSampleWarn && window.trade_count < 30 && (
        <span className="text-2xs text-survived">
          {window.trade_count} trade{window.trade_count === 1 ? "" : "s"} — limited statistical power.
        </span>
      )}
    </div>
  );
}

function CostToEdge({ window }: { window: WindowEconomics }) {
  const pctVal = window.cost_to_edge != null ? Math.min(1, window.cost_to_edge) : null;
  const heavy = (window.cost_to_edge ?? 0) >= 0.5 || window.costs_exceed_gross;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">Cost vs edge (in-sample)</span>
        <span className={cn("font-medium", heavy ? "text-survived" : "text-foreground")}>
          {window.cost_to_edge_label}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted" aria-hidden>
        <div
          className={cn("h-full rounded-full", heavy ? "bg-survived" : "bg-neutralStatus")}
          style={{ width: pctVal != null ? `${pctVal * 100}%` : "100%" }}
        />
      </div>
    </div>
  );
}
