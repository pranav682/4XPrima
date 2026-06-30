import { Link } from "react-router-dom";
import { FlaskConical, ArrowRight } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { AsyncBoundary, EmptyState } from "@/components/StateViews";
import { PageHeader } from "@/components/PageHeader";
import { Card } from "@/components/ui/card";
import { StateBadge } from "@/components/status";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { num, shortHash, titleCase } from "@/lib/format";
import type { RegistryEntry } from "@/lib/types";

/** The backtests index is derived from the registry — every tracked candidate
 *  has stored in-sample (and usually OOS) evidence. */
export function BacktestsView() {
  const state = useApi<RegistryEntry[]>(() => api.registry(), []);
  return (
    <>
      <PageHeader
        title="Backtests"
        subtitle="Stored evidence per candidate: the in-sample vs out-of-sample comparison the critic judged."
      />
      <AsyncBoundary
        state={state}
        empty={(d) => d.filter((e) => e.in_sample_evidence).length === 0}
        emptyView={
          <EmptyState
            icon={<FlaskConical className="h-5 w-5" aria-hidden />}
            title="No backtests yet"
            message="When the orchestrator backtests a candidate, its evidence is stored and becomes browsable here."
          />
        }
        children={(entries) => <BacktestTable entries={entries.filter((e) => e.in_sample_evidence)} />}
      />
    </>
  );
}

function BacktestTable({ entries }: { entries: RegistryEntry[] }) {
  return (
    <Card>
      <Table>
        <THead>
          <TR>
            <TH>Candidate</TH>
            <TH>Config hash</TH>
            <TH>State</TH>
            <TH className="text-right">IS Sharpe</TH>
            <TH className="text-right">OOS Sharpe</TH>
            <TH />
          </TR>
        </THead>
        <TBody>
          {entries.map((e) => {
            const ins = e.in_sample_evidence!;
            const oos = e.out_of_sample_evidence;
            return (
              <TR key={e.identity} className="hover:bg-elevated/60">
                <TD>
                  <span className="font-medium text-foreground">
                    {e.candidate.instrument} · {e.candidate.timeframe}
                  </span>
                  <span className="ml-2 text-2xs text-muted-foreground">
                    {titleCase(e.candidate.archetype)}
                  </span>
                </TD>
                <TD className="font-mono text-2xs text-muted-foreground">
                  {shortHash(ins.config_hash, 12)}
                </TD>
                <TD>
                  <StateBadge state={e.state} />
                </TD>
                <TD className="text-right tnum">{num(ins.metrics.sharpe_ratio)}</TD>
                <TD className="text-right tnum text-survived">
                  {oos ? num(oos.metrics.sharpe_ratio) : "—"}
                </TD>
                <TD className="text-right">
                  <Link
                    to={`/backtests/${ins.config_hash}`}
                    className="inline-flex items-center gap-1 text-xs text-accent underline-offset-2 hover:underline"
                  >
                    Open <ArrowRight className="h-3 w-3" aria-hidden />
                  </Link>
                </TD>
              </TR>
            );
          })}
        </TBody>
      </Table>
    </Card>
  );
}
