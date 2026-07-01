// The economic-health flag. A "retire" flag is the SYSTEM WORKING, not an alarm
// — an edge that decayed or never cleared its costs. So concern and retire are
// both amber (caution), never alarming red; retire is just the stronger amber.
import { CheckCircle2, TriangleAlert, Archive } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { EconomicFlag } from "@/lib/types";

export function EconomicFlagBadge({ flag }: { flag: EconomicFlag }) {
  if (flag === "ok") {
    return (
      <Badge tone="info">
        <CheckCircle2 className="h-3 w-3" aria-hidden /> Economics OK
      </Badge>
    );
  }
  if (flag === "concern") {
    return (
      <Badge tone="survived">
        <TriangleAlert className="h-3 w-3" aria-hidden /> Concern
      </Badge>
    );
  }
  return (
    <Badge tone="survived" className="font-semibold ring-1 ring-survived/40">
      <Archive className="h-3 w-3" aria-hidden /> Retire candidate
    </Badge>
  );
}
