import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-sm border px-2 py-0.5 text-2xs font-medium uppercase tracking-wide whitespace-nowrap",
  {
    variants: {
      tone: {
        neutral: "border-border bg-muted text-muted-foreground",
        accent: "border-accent/40 bg-accent/10 text-accent",
        // Status tones — see DESIGN.md. "survived" is amber (caution), never green.
        killed: "border-killed/40 bg-killed/10 text-killed",
        survived: "border-survived/40 bg-survived/10 text-survived",
        info: "border-neutralStatus/40 bg-neutralStatus/10 text-neutralStatus",
        outline: "border-border bg-transparent text-foreground",
      },
    },
    defaultVariants: { tone: "neutral" },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, tone, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ tone }), className)} {...props} />;
}
