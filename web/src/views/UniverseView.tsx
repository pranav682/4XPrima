import { Globe, Info, Check, TriangleAlert } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { AsyncBoundary, EmptyState } from "@/components/StateViews";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { cn } from "@/lib/utils";
import type { DroppedPair, UniverseCorrelation, UniverseView as UniverseData } from "@/lib/types";

export function UniverseView() {
  const state = useApi<UniverseData>(() => api.universe(), []);
  return (
    <>
      <PageHeader
        title="Universe"
        subtitle="The pair screener's structural decisions: which pairs are admitted to trade on, and which are dropped — and why."
      />
      <WhyNarrow />
      <AsyncBoundary
        state={state}
        empty={(d) => !d.available}
        emptyView={
          <EmptyState
            icon={<Globe className="h-5 w-5" aria-hidden />}
            title="No screen run yet"
            message="Run the pair screener (via a real cycle, or the demo seed) to see which pairs are admitted and which are dropped, with the structural reason for each."
          />
        }
        children={(u) => (
          <div className="flex flex-col gap-5">
            <Admitted admitted={u.admitted} />
            <Dropped dropped={u.dropped} />
            <Correlation correlation={u.correlation} />
          </div>
        )}
      />
    </>
  );
}

function WhyNarrow() {
  return (
    <div className="mb-4 flex items-start gap-2 rounded-md border border-border bg-elevated/40 px-3 py-2.5 text-2xs leading-relaxed text-muted-foreground">
      <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
      <span>
        The universe is <strong className="text-foreground">deliberately narrow</strong>, and never
        chosen by past return (that would be selection bias). It is picked on{" "}
        <strong className="text-foreground">structure</strong>: (1){" "}
        <strong className="text-foreground">low mutual correlation</strong> — highly correlated pairs
        are one bet, not diversification; (2){" "}
        <strong className="text-foreground">cost-to-move</strong> — a wide spread relative to the
        typical move eats the edge before the strategy can; (3){" "}
        <strong className="text-foreground">enough clean data</strong>. A small, structurally-chosen
        set also limits multiple-testing / overfitting inflation — the more pairs × strategies you
        test, the more "winners" you find by chance.
      </span>
    </div>
  );
}

function Admitted({ admitted }: { admitted: UniverseData["admitted"] }) {
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle>Admitted · {admitted.length}</CardTitle>
        <span className="text-2xs text-muted-foreground">
          structural shortlist (order = selection, not profitability)
        </span>
      </CardHeader>
      <CardContent className="px-0 pb-0">
        <Table>
          <THead>
            <TR>
              <TH>Pair</TH>
              <TH className="text-right">Cost-to-move (spr/ATR)</TH>
              <TH className="text-right">Max corr w/ selected</TH>
              <TH>Reason</TH>
            </TR>
          </THead>
          <TBody>
            {admitted.map((a) => (
              <TR key={a.pair}>
                <TD>
                  <span className="inline-flex items-center gap-1.5">
                    <Check className="h-3.5 w-3.5 text-neutralStatus" aria-hidden />
                    <span className="font-medium text-foreground">{a.pair}</span>
                  </span>
                </TD>
                <TD className="text-right tnum text-muted-foreground">
                  {a.cost_to_move ?? "n/a"}
                </TD>
                <TD className="text-right tnum text-muted-foreground">
                  {a.max_correlation_with_selected.toFixed(2)}
                </TD>
                <TD className="text-xs text-muted-foreground">{a.reason}</TD>
              </TR>
            ))}
          </TBody>
        </Table>
      </CardContent>
    </Card>
  );
}

type DropKind = "correlation" | "cost" | "coverage" | "other";

function classifyDrop(reason: string): DropKind {
  const r = reason.toLowerCase();
  if (r.includes("correlation")) return "correlation";
  if (r.includes("cost-to-move") || r.includes("spread/atr")) return "cost";
  if (r.includes("insufficient data") || r.includes("coverage")) return "coverage";
  return "other";
}

const DROP_LABEL: Record<DropKind, string> = {
  correlation: "correlation",
  cost: "cost-to-move",
  coverage: "data coverage",
  other: "structural",
};

function Dropped({ dropped }: { dropped: DroppedPair[] }) {
  if (dropped.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Dropped · 0</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-muted-foreground">
          No candidate pairs were dropped in this screen.
        </CardContent>
      </Card>
    );
  }
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle>Dropped · {dropped.length}</CardTitle>
        <span className="text-2xs text-muted-foreground">
          dropping a pair is the screener working, not an error
        </span>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {dropped.map((d) => {
          const kind = classifyDrop(d.reason);
          return (
            <div
              key={d.pair}
              className="flex flex-wrap items-center gap-2 rounded-md border-l-2 border-survived/70 bg-survived/[0.06] px-3 py-2"
            >
              <TriangleAlert className="h-3.5 w-3.5 shrink-0 text-survived" aria-hidden />
              <span className="font-medium text-foreground">{d.pair}</span>
              <Badge tone="survived">{DROP_LABEL[kind]}</Badge>
              <span className="text-xs text-muted-foreground">— {d.reason}</span>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

function Correlation({ correlation }: { correlation: UniverseCorrelation }) {
  const { pairs, matrix } = correlation;
  if (pairs.length === 0) {
    return null;
  }
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle>Return correlation</CardTitle>
        <span className="text-2xs text-muted-foreground">
          darker = more correlated (moves together — less diversification)
        </span>
      </CardHeader>
      <CardContent className="px-0 pb-4">
        <div className="w-full overflow-x-auto px-5">
          <table className="border-collapse text-2xs" role="table" aria-label="Return correlation matrix">
            <thead>
              <tr>
                <th className="p-1" />
                {pairs.map((p) => (
                  <th
                    key={p}
                    scope="col"
                    className="p-1 font-medium text-muted-foreground tnum"
                  >
                    {p.slice(0, 6)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {pairs.map((rowPair, i) => (
                <tr key={rowPair}>
                  <th
                    scope="row"
                    className="whitespace-nowrap p-1 text-right font-medium text-muted-foreground"
                  >
                    {rowPair.slice(0, 6)}
                  </th>
                  {matrix[i].map((v, j) => (
                    <td
                      key={j}
                      className="p-0"
                      title={`${rowPair} vs ${pairs[j]}: ${v.toFixed(2)}`}
                    >
                      <div
                        className={cn(
                          "flex h-8 w-12 items-center justify-center tnum",
                          Math.abs(v) >= 0.8 && i !== j ? "text-survived" : "text-foreground/80",
                        )}
                        style={{
                          backgroundColor:
                            i === j
                              ? "hsl(240 4% 14%)"
                              : `hsla(38, 92%, 56%, ${Math.min(0.6, Math.abs(v) * 0.6)})`,
                        }}
                      >
                        {v.toFixed(2)}
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
