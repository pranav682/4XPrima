import { Link, useParams } from "react-router-dom";
import { ArrowLeft, LineChart } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { AsyncBoundary } from "@/components/StateViews";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StateBadge, VerdictBadge } from "@/components/status";
import { MetricComparison } from "@/components/MetricComparison";
import { ConcernList } from "@/components/ConcernList";
import { Stat } from "@/components/ui/misc";
import { dateUTC, money, titleCase } from "@/lib/format";
import type { BacktestDetail, Evidence } from "@/lib/types";

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
          const ins = bt.in_sample;
          const oos = bt.out_of_sample;
          return (
            <>
              <PageHeader
                title={`${cand.instrument} · ${cand.timeframe} · ${titleCase(cand.archetype)}`}
                subtitle={`config ${bt.config_hash}`}
                actions={<StateBadge state={bt.state} />}
              />

              <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_320px]">
                <div className="flex flex-col gap-5">
                  <Card>
                    <CardHeader>
                      <CardTitle>In-sample vs out-of-sample</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <MetricComparison
                        inSample={ins?.metrics ?? null}
                        outOfSample={oos?.metrics ?? null}
                      />
                    </CardContent>
                  </Card>

                  <EquityCurvePanel notice={bt.equity_curve_notice} available={bt.equity_curve_available} />
                </div>

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

                  {ins && <RunFacts label="In-sample run" ev={ins} />}
                  {oos && <RunFacts label="Out-of-sample run" ev={oos} />}

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

/** Honest empty state: the per-bar curve isn't persisted and the API must not
 *  re-derive it. We say so plainly rather than drawing a fabricated line. */
function EquityCurvePanel({ notice, available }: { notice: string; available: boolean }) {
  if (available) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Equity curve</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center gap-2.5 rounded-md border border-dashed border-border bg-background/40 py-10 text-center">
          <LineChart className="h-5 w-5 text-muted-foreground" aria-hidden />
          <p className="mx-auto max-w-md text-xs text-muted-foreground">{notice}</p>
        </div>
      </CardContent>
    </Card>
  );
}
