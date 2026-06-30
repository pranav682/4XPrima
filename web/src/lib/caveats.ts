// The "caveat travels with the metric" rule (see web/DESIGN.md). A metric is
// only as trustworthy as the sample behind it, so whenever we SHOW a metric we
// can also show its statistical-power caveat — derived from the metric itself
// (trade count), not from any recomputation.
import type { Metrics } from "./types";

// Below this many trades, a figure (especially out-of-sample) has weak
// statistical power — a jump is more likely noise than signal.
export const THIN_SAMPLE_TRADES = 30;

export function isThinSample(m: Metrics | null | undefined, threshold = THIN_SAMPLE_TRADES): boolean {
  return m != null && m.trade_count < threshold;
}

/** A short, honest note for a thin-sample metric, or null when the sample is
 *  large enough not to warrant one. Never implies the figure is good. */
export function sampleCaveat(m: Metrics | null | undefined): string | null {
  if (m == null || !isThinSample(m)) return null;
  const n = m.trade_count;
  return `Rests on ${n} trade${n === 1 ? "" : "s"} — limited statistical power; a jump on so few trades is noise, not validation.`;
}
