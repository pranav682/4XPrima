// The honesty layer. KILLED vs SURVIVED_FOR_NOW must be instantly distinct, and
// SURVIVED must NEVER be styled as success — it only means "the critic could not
// yet kill it." Amber = caution; red = killed; green is never used for a verdict.
import { Skull, TriangleAlert, CircleDot, Crown } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { CriticVerdictKind, RegistryState } from "@/lib/types";

export function VerdictBadge({ verdict }: { verdict: CriticVerdictKind }) {
  if (verdict === "kill") {
    return (
      <Badge tone="killed" aria-label="Critic verdict: killed">
        <Skull className="h-3 w-3" aria-hidden /> Killed
      </Badge>
    );
  }
  return (
    <Badge tone="survived" aria-label="Critic verdict: survived for now, not validated">
      <TriangleAlert className="h-3 w-3" aria-hidden /> Survived · not validated
    </Badge>
  );
}

const STATE_META: Record<
  RegistryState,
  { tone: "killed" | "survived" | "info" | "neutral" | "accent"; label: string }
> = {
  proposed: { tone: "neutral", label: "Proposed" },
  backtested: { tone: "neutral", label: "Backtested" },
  killed: { tone: "killed", label: "Killed" },
  survived_for_now: { tone: "survived", label: "Survived · not validated" },
  queued_for_approval: { tone: "survived", label: "Queued · awaiting operator" },
  approved: { tone: "info", label: "Approved (human)" },
  champion: { tone: "info", label: "Champion (human)" },
  live: { tone: "info", label: "Live (human)" },
};

export function StateBadge({ state }: { state: RegistryState }) {
  const meta = STATE_META[state] ?? { tone: "neutral" as const, label: state };
  const Icon = state === "killed" ? Skull : state === "champion" ? Crown : TriangleAlert;
  const showIcon = state === "killed" || state === "survived_for_now" || state === "queued_for_approval" || state === "champion";
  return (
    <Badge tone={meta.tone}>
      {showIcon ? <Icon className="h-3 w-3" aria-hidden /> : <CircleDot className="h-3 w-3" aria-hidden />}
      {meta.label}
    </Badge>
  );
}

export function isSurvivor(state: RegistryState): boolean {
  return state === "survived_for_now" || state === "queued_for_approval";
}
