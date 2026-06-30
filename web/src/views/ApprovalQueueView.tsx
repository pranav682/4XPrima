import { ListChecks, Lock } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { AsyncBoundary, EmptyState } from "@/components/StateViews";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardContent } from "@/components/ui/card";
import { VerdictBadge } from "@/components/status";
import { ConcernList } from "@/components/ConcernList";
import { Stat } from "@/components/ui/misc";
import { num, pct, titleCase } from "@/lib/format";
import type { ApprovalItem, Metrics } from "@/lib/types";

export function ApprovalQueueView() {
  const state = useApi<ApprovalItem[]>(() => api.approvalQueue(), []);
  return (
    <>
      <PageHeader
        title="Approval queue"
        subtitle="Candidates the critic could not kill, awaiting a human decision. Surviving is not validation — the critic's concerns are shown alongside the metrics, not beneath them."
      />
      <AsyncBoundary
        state={state}
        empty={(d) => d.length === 0}
        emptyView={
          <EmptyState
            icon={<ListChecks className="h-5 w-5" aria-hidden />}
            title="Nothing awaiting review"
            message="When a candidate survives the critic, it lands here with its evidence and the critic's surviving concerns. Nothing is auto-approved."
          />
        }
        children={(items) => (
          <div className="flex flex-col gap-4">
            {items.map((item) => (
              <QueueItem key={item.entry_id} item={item} />
            ))}
          </div>
        )}
      />
    </>
  );
}

function QueueItem({ item }: { item: ApprovalItem }) {
  const c = item.candidate;
  const ins = item.in_sample_evidence.metrics;
  const oos = item.out_of_sample_evidence?.metrics ?? null;
  return (
    <Card className="border-survived/30">
      <CardContent className="pt-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div className="flex flex-col gap-0.5">
            <span className="text-sm font-semibold text-foreground">
              {c.instrument} · {c.timeframe} · {titleCase(c.archetype)}
            </span>
            <span className="font-mono text-2xs text-muted-foreground">{item.identity}</span>
          </div>
          <VerdictBadge verdict={item.critic_verdict.verdict} />
        </div>

        <p className="mt-3 text-sm text-foreground/90">
          {item.report_explanation ??
            "The critic did not kill this candidate. Surviving means only that — here is what it remains worried about."}
        </p>

        <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
          {/* Concerns get the prominent, leading column — equal-or-greater weight. */}
          <section aria-label="Surviving concerns" className="order-1">
            <h3 className="mb-2 text-2xs font-medium uppercase tracking-wide text-survived">
              What the critic still worries about
            </h3>
            <ConcernList concerns={item.critic_verdict.concerns} />
          </section>

          <section aria-label="Metrics" className="order-2">
            <h3 className="mb-2 text-2xs font-medium uppercase tracking-wide text-muted-foreground">
              Metrics (in-sample → out-of-sample)
            </h3>
            <div className="grid grid-cols-2 gap-3 rounded-md border border-border bg-elevated/50 px-3 py-3">
              <MetricPair label="Sharpe" is={num(ins.sharpe_ratio)} oos={oos ? num(oos.sharpe_ratio) : "—"} />
              <MetricPair label="Return" is={pct(ins.total_return_pct)} oos={oos ? pct(oos.total_return_pct) : "—"} />
              <MetricPair label="Profit factor" is={fmtRatio(ins)} oos={oos ? fmtRatio(oos) : "—"} />
              <MetricPair label="Trades" is={String(ins.trade_count)} oos={oos ? String(oos.trade_count) : "—"} />
            </div>
          </section>
        </div>

        <ActionPlaceholder />
      </CardContent>
    </Card>
  );
}

function fmtRatio(m: Metrics): string {
  return m.profit_factor == null ? "n/a" : m.profit_factor.toFixed(2);
}

function MetricPair({ label, is, oos }: { label: string; is: string; oos: string }) {
  return (
    <Stat label={label}>
      <span className="tnum">
        {is} <span className="text-muted-foreground">→</span> {oos}
      </span>
    </Stat>
  );
}

/** The operator's decision belongs in the NEXT slice. We reserve the layout for
 *  it but expose no action — this dashboard is strictly read-only. */
function ActionPlaceholder() {
  return (
    <div className="mt-4 flex flex-wrap items-center justify-between gap-2 rounded-md border border-dashed border-border bg-background/40 px-3 py-2.5">
      <span className="flex items-center gap-1.5 text-2xs text-muted-foreground">
        <Lock className="h-3.5 w-3.5" aria-hidden />
        Read-only. Approval is a human action handled outside this dashboard (a later slice).
      </span>
      <span
        aria-disabled="true"
        className="cursor-not-allowed select-none rounded-md border border-border px-3 py-1 text-2xs font-medium text-muted-foreground opacity-60"
        title="Not available in this read-only slice"
      >
        Operator decision
      </span>
    </div>
  );
}
