import { Link } from "react-router-dom";
import { GitBranch, ArrowRight } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { AsyncBoundary, EmptyState } from "@/components/StateViews";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardContent } from "@/components/ui/card";
import { StateBadge, VerdictBadge } from "@/components/status";
import { cn } from "@/lib/utils";
import { num, pct, shortHash, titleCase } from "@/lib/format";
import type { Candidate, RegistryEntry } from "@/lib/types";

export function RegistryView() {
  const state = useApi<RegistryEntry[]>(() => api.registry(), []);
  return (
    <>
      <PageHeader
        title="Registry"
        subtitle="Champion & challengers, keyed by strategy identity. Killed and survived-for-now are kept honestly distinct — surviving only means the critic could not yet kill it."
      />
      <AsyncBoundary
        state={state}
        empty={(d) => d.length === 0}
        emptyView={
          <EmptyState
            icon={<GitBranch className="h-5 w-5" aria-hidden />}
            title="No candidates yet"
            message="Once the orchestrator runs, every proposed strategy is tracked here by its config-hash identity, with its critic verdict."
          />
        }
        children={(entries) => <RegistryList entries={entries} />}
      />
    </>
  );
}

function RegistryList({ entries }: { entries: RegistryEntry[] }) {
  const champion = entries.find((e) => e.state === "champion");
  const challengers = entries.filter((e) => e.state !== "champion");
  return (
    <div className="flex flex-col gap-6">
      <section aria-label="Champion">
        <h2 className="mb-2 text-2xs font-medium uppercase tracking-wide text-muted-foreground">
          Champion
        </h2>
        {champion ? (
          <EntryCard entry={champion} />
        ) : (
          <Card>
            <CardContent className="py-6 text-center text-xs text-muted-foreground">
              No champion. Promotion is a human-only action and is not part of this read-only
              dashboard.
            </CardContent>
          </Card>
        )}
      </section>

      <section aria-label="Challengers">
        <h2 className="mb-2 text-2xs font-medium uppercase tracking-wide text-muted-foreground">
          Challengers · {challengers.length}
        </h2>
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          {challengers.map((e) => (
            <EntryCard key={e.identity} entry={e} />
          ))}
        </div>
      </section>
    </div>
  );
}

function paramSummary(candidate: Candidate): string {
  return candidate.parameters.map((p) => `${p.name} ${p.value}`).join(" · ");
}

function EntryCard({ entry }: { entry: RegistryEntry }) {
  const killed = entry.state === "killed";
  const ins = entry.in_sample_evidence;
  const oos = entry.out_of_sample_evidence;
  const linkHash = ins?.config_hash ?? oos?.config_hash;
  return (
    <Card
      className={cn(
        "flex flex-col",
        killed && "opacity-75",
        killed ? "border-killed/30" : "border-survived/30",
      )}
    >
      <CardContent className="flex flex-1 flex-col gap-3 pt-4">
        <div className="flex items-start justify-between gap-2">
          <div className="flex flex-col gap-0.5">
            <span className="text-sm font-semibold text-foreground">
              {entry.candidate.instrument} · {entry.candidate.timeframe} ·{" "}
              {titleCase(entry.candidate.archetype)}
            </span>
            <span className="font-mono text-2xs text-muted-foreground">{shortHash(entry.identity)}</span>
          </div>
          <StateBadge state={entry.state} />
        </div>

        <p className={cn("tnum text-2xs", killed ? "text-muted-foreground line-through" : "text-muted-foreground")}>
          {paramSummary(entry.candidate)}
        </p>

        <div className="grid grid-cols-2 gap-3 rounded-md border border-border bg-elevated/50 px-3 py-2">
          <MiniMetric label="In-sample Sharpe" value={ins ? num(ins.metrics.sharpe_ratio) : "—"} sub={ins ? pct(ins.metrics.total_return_pct) : undefined} />
          <MiniMetric
            label="Out-of-sample Sharpe"
            value={oos ? num(oos.metrics.sharpe_ratio) : "—"}
            sub={oos ? pct(oos.metrics.total_return_pct) : "not opened"}
            emphasize
          />
        </div>

        <div className="mt-auto flex items-center justify-between gap-2 pt-1">
          {entry.critic_verdict ? (
            <VerdictBadge verdict={entry.critic_verdict.verdict} />
          ) : (
            <span className="text-2xs text-muted-foreground">Not yet critiqued</span>
          )}
          {linkHash && (
            <Link
              to={`/backtests/${linkHash}`}
              className="inline-flex items-center gap-1 text-xs text-accent underline-offset-2 hover:underline"
            >
              Backtest <ArrowRight className="h-3 w-3" aria-hidden />
            </Link>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function MiniMetric({
  label,
  value,
  sub,
  emphasize,
}: {
  label: string;
  value: string;
  sub?: string;
  emphasize?: boolean;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-2xs uppercase tracking-wide text-muted-foreground">{label}</span>
      <span className={cn("tnum text-base font-semibold", emphasize ? "text-foreground" : "text-foreground")}>
        {value}
      </span>
      {sub && <span className="tnum text-2xs text-muted-foreground">{sub}</span>}
    </div>
  );
}
