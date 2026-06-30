// Renders the statistical-power caveat that travels with a (thin-sample) metric.
// Amber caution, never alarming-red and never green — consistent with the
// verdict semantics. Renders nothing when the sample is adequate.
import { TriangleAlert } from "lucide-react";
import { sampleCaveat } from "@/lib/caveats";
import { cn } from "@/lib/utils";
import type { Metrics } from "@/lib/types";

export function SampleCaveat({
  metrics,
  className,
}: {
  metrics: Metrics | null | undefined;
  className?: string;
}) {
  const note = sampleCaveat(metrics);
  if (!note) return null;
  return (
    <span
      role="note"
      className={cn("flex items-start gap-1 text-2xs leading-snug text-survived", className)}
    >
      <TriangleAlert className="mt-0.5 h-3 w-3 shrink-0" aria-hidden />
      <span>{note}</span>
    </span>
  );
}
