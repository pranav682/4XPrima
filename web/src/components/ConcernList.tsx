// The critic's surviving concerns, shown with at-least-equal weight to the
// headline metrics. This is the heart of the honesty ethos: a survivor is only
// "not yet killed", so its caveats are first-class, not footnotes.
import { TriangleAlert } from "lucide-react";
import { titleCase } from "@/lib/format";
import type { Concern } from "@/lib/types";

export function ConcernList({ concerns }: { concerns: Concern[] }) {
  if (concerns.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        No specific concerns were recorded for this candidate.
      </p>
    );
  }
  return (
    <ul className="flex flex-col gap-2" aria-label="Critic's surviving concerns">
      {concerns.map((c, i) => (
        <li
          key={`${c.item}-${i}`}
          className="flex gap-2.5 rounded-md border-l-2 border-survived/70 bg-survived/[0.06] px-3 py-2"
        >
          <TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0 text-survived" aria-hidden />
          <div className="flex flex-col gap-0.5">
            <span className="text-2xs font-medium uppercase tracking-wide text-survived">
              {titleCase(c.item)}
            </span>
            <span className="text-sm text-foreground/90">{c.finding}</span>
          </div>
        </li>
      ))}
    </ul>
  );
}
