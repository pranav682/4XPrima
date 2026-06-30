// Display formatters. These format VERBATIM values for reading; they never
// recompute or re-derive an engine number. A fraction like "0.30" is shown as
// "30.00%", but the underlying string came straight from the API.

export function pct(value: string | number, digits = 2): string {
  const n = typeof value === "string" ? Number(value) : value;
  if (!Number.isFinite(n)) return "—";
  return `${(n * 100).toFixed(digits)}%`;
}

export function num(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

/** Profit factor / sortino can be null = "undefined / no losing side". We show
 *  an explicit marker, never a flattering large number. */
export function ratioOrNull(value: number | null, digits = 2): string {
  if (value == null) return "n/a";
  if (!Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

export function money(value: string): string {
  // Keep the verbatim string; just prefix. Trim only trailing-zero noise beyond
  // 4dp for readability without changing magnitude.
  return `$${value}`;
}

/** Format a verbatim money string for reading: 2dp + thousands separators. The
 *  underlying value is unchanged; this is display only. */
export function usd(value: string): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return `$${value}`;
  return n.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

/** Signed money for a realized P&L figure (e.g. "+$2,880.34"). */
export function signedUsd(value: string): string {
  const n = Number(value);
  const body = usd(value).replace("-", "");
  return n >= 0 ? `+${body}` : `-${body}`;
}

export function pnlSign(value: string): "pos" | "neg" | "flat" {
  const n = Number(value);
  if (!Number.isFinite(n) || n === 0) return "flat";
  return n > 0 ? "pos" : "neg";
}

export function shortHash(hash: string, head = 8): string {
  return hash.length > head ? hash.slice(0, head) : hash;
}

export function dateUTC(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const p = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())} ` +
    `${p(d.getUTCHours())}:${p(d.getUTCMinutes())} UTC`
  );
}

export function duration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

export function titleCase(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
