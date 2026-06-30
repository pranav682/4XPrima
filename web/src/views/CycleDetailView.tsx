import { Link, useParams } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { AsyncBoundary } from "@/components/StateViews";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Stat } from "@/components/ui/misc";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { dateUTC, duration, money, titleCase } from "@/lib/format";
import type { CycleDetail, CycleOutcome } from "@/lib/types";

function tone(outcome: CycleOutcome) {
  return outcome === "completed" ? "info" : outcome === "aborted_budget" ? "survived" : "killed";
}

export function CycleDetailView() {
  const { cycleId = "" } = useParams();
  const state = useApi<CycleDetail>(() => api.cycle(cycleId), [cycleId]);
  return (
    <>
      <Link
        to="/cycles"
        className="mb-3 inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-3.5 w-3.5" aria-hidden /> All cycles
      </Link>
      <AsyncBoundary
        state={state}
        empty={() => false}
        children={(c) => (
          <>
            <PageHeader
              title={c.cycle_id}
              subtitle={`Started ${dateUTC(c.started_at)} · ran for ${duration(c.duration_seconds)}`}
              actions={<Badge tone={tone(c.outcome)}>{titleCase(c.outcome)}</Badge>}
            />
            <div className="mb-5 grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Card>
                <CardContent className="pt-5">
                  <Stat label="Proposed" emphasis>
                    {c.candidates_proposed}
                  </Stat>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-5">
                  <Stat label="Killed" emphasis>
                    <span className="text-killed">{c.candidates_killed}</span>
                  </Stat>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-5">
                  <Stat label="Queued" emphasis>
                    <span className="text-survived">{c.candidates_queued}</span>
                  </Stat>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-5">
                  <Stat label="Cost" emphasis>
                    {money(c.total_cost_usd)}
                  </Stat>
                </CardContent>
              </Card>
            </div>

            {c.abort_reason && (
              <p className="mb-5 rounded-md border-l-2 border-killed/70 bg-killed/[0.06] px-3 py-2 text-xs text-killed">
                Aborted: {c.abort_reason}
              </p>
            )}

            <Card>
              <CardHeader>
                <CardTitle>Per-stage cost</CardTitle>
              </CardHeader>
              <CardContent className="px-0 pb-0">
                <Table>
                  <THead>
                    <TR>
                      <TH>Stage</TH>
                      <TH className="text-right">Cost (USD)</TH>
                    </TR>
                  </THead>
                  <TBody>
                    {Object.entries(c.stage_costs_usd).map(([stage, cost]) => (
                      <TR key={stage}>
                        <TD className="text-muted-foreground">{titleCase(stage)}</TD>
                        <TD className="text-right tnum">{money(cost)}</TD>
                      </TR>
                    ))}
                    {Object.keys(c.stage_costs_usd).length === 0 && (
                      <TR>
                        <TD className="text-muted-foreground" colSpan={2}>
                          No stage costs recorded.
                        </TD>
                      </TR>
                    )}
                  </TBody>
                </Table>
              </CardContent>
            </Card>
          </>
        )}
      />
    </>
  );
}
