import { Link } from "react-router-dom";
import { Activity } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { AsyncBoundary, EmptyState } from "@/components/StateViews";
import { PageHeader } from "@/components/PageHeader";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { dateUTC, duration, money, titleCase } from "@/lib/format";
import type { CycleOutcome, CycleSummary } from "@/lib/types";

function OutcomeBadge({ outcome }: { outcome: CycleOutcome }) {
  const tone = outcome === "completed" ? "info" : outcome === "aborted_budget" ? "survived" : "killed";
  return <Badge tone={tone}>{titleCase(outcome)}</Badge>;
}

export function CyclesView() {
  const state = useApi<CycleSummary[]>(() => api.cycles(), []);
  return (
    <>
      <PageHeader
        title="Cycles"
        subtitle="Each run of the slow loop: market context → strategy lab → backtest → critic."
      />
      <AsyncBoundary
        state={state}
        empty={(d) => d.length === 0}
        emptyView={
          <EmptyState
            icon={<Activity className="h-5 w-5" aria-hidden />}
            title="No cycles yet"
            message="Run the orchestrator to produce one: python -m scripts.run_orchestrator. Cycles persist under data/orchestration/cycles/ and appear here, newest first."
          />
        }
        children={(cycles) => <CyclesTable cycles={cycles} />}
      />
    </>
  );
}

function CyclesTable({ cycles }: { cycles: CycleSummary[] }) {
  return (
    <Card>
      <Table>
        <THead>
          <TR>
            <TH>Cycle</TH>
            <TH>Started (UTC)</TH>
            <TH>Outcome</TH>
            <TH className="text-right">Proposed</TH>
            <TH className="text-right">Killed</TH>
            <TH className="text-right">Queued</TH>
            <TH className="text-right">Cost</TH>
            <TH className="text-right">Duration</TH>
          </TR>
        </THead>
        <TBody>
          {cycles.map((c) => (
            <TR key={c.cycle_id} className="hover:bg-elevated/60">
              <TD>
                <Link
                  to={`/cycles/${c.cycle_id}`}
                  className="font-medium text-foreground underline-offset-2 hover:text-accent hover:underline"
                >
                  {c.cycle_id}
                </Link>
              </TD>
              <TD className="tnum text-muted-foreground">{dateUTC(c.started_at)}</TD>
              <TD>
                <OutcomeBadge outcome={c.outcome} />
              </TD>
              <TD className="text-right tnum">{c.candidates_proposed}</TD>
              <TD className="text-right tnum text-killed">{c.candidates_killed}</TD>
              <TD className="text-right tnum text-survived">{c.candidates_queued}</TD>
              <TD className="text-right tnum">{money(c.total_cost_usd)}</TD>
              <TD className="text-right tnum text-muted-foreground">{duration(c.duration_seconds)}</TD>
            </TR>
          ))}
        </TBody>
      </Table>
    </Card>
  );
}
