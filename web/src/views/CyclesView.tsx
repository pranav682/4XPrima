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

/** Plain-language summary of a completed cycle. Kills are the system working
 *  (the critic rejecting weak candidates), so this reads matter-of-factly — red
 *  is reserved for genuine failures (aborts), surfaced by the OutcomeBadge. */
function cycleNarrative(c: CycleSummary): string {
  if (c.outcome !== "completed") {
    return c.abort_reason ? `Aborted — ${c.abort_reason}` : "Aborted before completing.";
  }
  const { candidates_proposed: n, candidates_killed: k, candidates_queued: q } = c;
  if (n === 0) return "No candidates proposed.";
  const killed =
    k === 0
      ? "none rejected"
      : k === n
        ? `${n === 2 ? "both" : "all"} rejected by critic`
        : `${k} rejected by critic`;
  const queued = q === 0 ? "none queued" : `${q} queued for review`;
  return `${n} proposed · ${killed} · ${queued}`;
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
                <div className="flex flex-col gap-0.5">
                  <Link
                    to={`/cycles/${c.cycle_id}`}
                    className="font-medium text-foreground underline-offset-2 hover:text-accent hover:underline"
                  >
                    {c.cycle_id}
                  </Link>
                  <span className="text-2xs text-muted-foreground">{cycleNarrative(c)}</span>
                </div>
              </TD>
              <TD className="tnum text-muted-foreground">{dateUTC(c.started_at)}</TD>
              <TD>
                <OutcomeBadge outcome={c.outcome} />
              </TD>
              <TD className="text-right tnum">{c.candidates_proposed}</TD>
              {/* Kills are an expected, healthy outcome — neutral, not alarming. */}
              <TD className="text-right tnum text-foreground">{c.candidates_killed}</TD>
              <TD
                className={
                  c.candidates_queued > 0
                    ? "text-right tnum text-survived"
                    : "text-right tnum text-muted-foreground"
                }
              >
                {c.candidates_queued}
              </TD>
              <TD className="text-right tnum">{money(c.total_cost_usd)}</TD>
              <TD className="text-right tnum text-muted-foreground">{duration(c.duration_seconds)}</TD>
            </TR>
          ))}
        </TBody>
      </Table>
    </Card>
  );
}
