import { Link, useParams } from "react-router-dom";
import { ArrowLeft, LineChart } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { AsyncBoundary } from "@/components/StateViews";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StateBadge, VerdictBadge } from "@/components/status";
import { MetricComparison } from "@/components/MetricComparison";
import { EquityCurveChart } from "@/components/EquityCurveChart";
import { ConcernList } from "@/components/ConcernList";
import { SampleCaveat } from "@/components/SampleCaveat";
import { Stat } from "@/components/ui/misc";
import { cn } from "@/lib/utils";
import { dateUTC, money, pct, pnlSign, signedUsd, titleCase, usd } from "@/lib/format";
import type { BacktestArtifact, BacktestDetail, Evidence } from "@/lib/types";

export function BacktestDetailView() {
  const { configHash = "" } = useParams();
  const state = useApi<BacktestDetail>(() => api.backtest(configHash), [configHash]);
  return (
    <>
      <Link
        to="/backtests"
        className="mb-3 inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-3.5 w-3.5" aria-hidden /> All backtests
      </Link>
      <AsyncBoundary
        state={state}
        empty={() => false}
        children={(bt) => {
          const cand = bt.candidate;
          return (
            <>
              <PageHeader
                title={`${cand.instrument} · ${cand.timeframe} · ${titleCase(cand.archetype)}`}
                subtitle={`Selected candidate · config ${bt.config_hash}`}
                actions={<StateBadge state={bt.state} />}
              />

              {/* Centerpiece: the equity curve + amount-earned annotations. */}
              <EquitySection bt={bt} />

              <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-[1fr_320px]">
                <Card>
                  <CardHeader>
                    <CardTitle>In-sample vs out-of-sample</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <MetricComparison
                      inSample={bt.in_sample?.metrics ?? null}
                      outOfSample={bt.out_of_sample?.metrics ?? null}
                    />
                  </CardContent>
                </Card>

                <div className="flex flex-col gap-5">
                  <Card>
                    <CardHeader>
                      <CardTitle>Parameters</CardTitle>
                    </CardHeader>
                    <CardContent className="flex flex-col gap-2">
                      {cand.parameters.map((p) => (
                        <div key={p.name} className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">{p.name}</span>
                          <span className="tnum text-foreground">{p.value}</span>
                        </div>
                      ))}
                    </CardContent>
                  </Card>

                  {bt.in_sample && <RunFacts label="In-sample run" ev={bt.in_sample} />}
                  {bt.out_of_sample && <RunFacts label="Out-of-sample run" ev={bt.out_of_sample} />}

                  {bt.critic_verdict && (
                    <Card className="border-survived/30">
                      <CardHeader className="flex-row items-center justify-between">
                        <CardTitle>Critic</CardTitle>
                        <VerdictBadge verdict={bt.critic_verdict.verdict} />
                      </CardHeader>
                      <CardContent>
                        <ConcernList concerns={bt.critic_verdict.concerns} />
                      </CardContent>
                    </Card>
                  )}
                </div>
              </div>
            </>
          );
        }}
      />
    </>
  );
}

function EquitySection({ bt }: { bt: BacktestDetail }) {
  const is = bt.in_sample_artifact;
  const oos = bt.out_of_sample_artifact;
  if (!bt.equity_curve_available || (!is && !oos)) {
    return <EquityCurveUnavailable notice={bt.equity_curve_notice} />;
  }
  const start = is?.starting_balance ?? oos?.starting_balance ?? "0";
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle>Equity curve</CardTitle>
        <span className="text-2xs text-muted-foreground">
          starts at {usd(start)} · in-sample then the sealed out-of-sample slice
        </span>
      </CardHeader>
      <CardContent>
        <EquityCurveChart inSample={is} outOfSample={oos} />
        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <PnlStat label="In-sample earned" artifact={is} />
          <PnlStat label="Out-of-sample earned" artifact={oos} emphasizeCaveat />
          <Stat label="In-sample return">{is ? pct(is.return_pct) : "—"}</Stat>
          <Stat label="Out-of-sample return">{oos ? pct(oos.return_pct) : "—"}</Stat>
          <Stat label="Peak equity">{is ? usd(is.peak_equity) : "—"}</Stat>
          <Stat label="Max drawdown (IS)">{is ? pct(is.max_drawdown_pct) : "—"}</Stat>
          <Stat label="Trades (IS)">{is ? is.trade_count : "—"}</Stat>
          <Stat label="Trades (OOS)">{oos ? oos.trade_count : "—"}</Stat>
        </div>
        {oos && (
          <SampleCaveat metrics={bt.out_of_sample?.metrics ?? null} className="mt-3" />
        )}
      </CardContent>
    </Card>
  );
}

function PnlStat({
  label,
  artifact,
  emphasizeCaveat,
}: {
  label: string;
  artifact: BacktestArtifact | null;
  emphasizeCaveat?: boolean;
}) {
  if (!artifact) {
    return <Stat label={label}>{emphasizeCaveat ? "not opened" : "—"}</Stat>;
  }
  const sign = pnlSign(artifact.net_pnl);
  return (
    <Stat label={label} emphasis>
      <span
        className={cn(
          sign === "pos" && "text-pnlPos",
          sign === "neg" && "text-pnlNeg",
        )}
      >
        {signedUsd(artifact.net_pnl)}
      </span>
    </Stat>
  );
}

function RunFacts({ label, ev }: { label: string; ev: Evidence }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{label}</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-2 gap-3">
        <Stat label="Window start">{dateUTC(ev.window_start)}</Stat>
        <Stat label="Window end">{dateUTC(ev.window_end)}</Stat>
        <Stat label="Bars processed">{ev.bars_processed}</Stat>
        <Stat label="Total cost">{money(ev.cost_total)}</Stat>
        <Stat label="Signals accepted">{ev.n_signals_accepted}</Stat>
        <Stat label="Signals rejected">{ev.n_signals_rejected}</Stat>
      </CardContent>
    </Card>
  );
}

/** Honest fallback when no curve artifact was persisted (e.g. a candidate the
 *  critic never reached). We say so plainly; a real run persists the curve. */
function EquityCurveUnavailable({ notice }: { notice: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Equity curve</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center gap-2.5 rounded-md border border-dashed border-border bg-background/40 py-10 text-center">
          <LineChart className="h-5 w-5 text-muted-foreground" aria-hidden />
          <p className="mx-auto max-w-md text-xs text-muted-foreground">{notice}</p>
          <p className="mx-auto max-w-md text-2xs text-muted-foreground/80">
            A real orchestrator run persists the per-bar curve; this candidate has none stored.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
