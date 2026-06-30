# 4xPrima dashboard — design system

This is a **dense, data-heavy decision tool**, not a marketing site. The look is
calm, legible, and trustworthy. It must stay consistent across later slices, so
the tokens and rules below are the source of truth.

## Influences

- **Linear** (primary) — restrained near-monochrome surface ramp, one functional
  accent, tight spacing, crisp type, color = status not decoration.
- **Stripe Dashboard** — financial figures that breathe; calm hierarchy;
  restrained chart styling.
- **Vercel Dashboard** — clean layout and honest information density.

Primitives are **shadcn-style** (we own the source: `cva` + `cn()` +
CSS-variable tokens) rather than installed via the shadcn CLI — see the note at
the bottom. Charts use **Recharts** with custom restrained styling.

## Tokens

All tokens are CSS variables in `src/index.css` (HSL triplets) and surfaced
through `tailwind.config.js`. Dark, near-monochrome.

### Surface ramp (near-monochrome)

| Token | Use |
| --- | --- |
| `--background` | app background |
| `--surface` | cards, sidebar, header |
| `--elevated` | insets, hover, nested panels |
| `--border` | hairlines, dividers |
| `--foreground` | primary text |
| `--muted-foreground` | secondary text, labels |

### The ONE accent

`--accent` (indigo) — **interactive only**: links, active nav, focus ring. Never
used to signal that data is good or bad.

### Status palette (the honest part — see below)

| Token | Meaning | Color |
| --- | --- | --- |
| `--killed` | killed / terminated / negative | red |
| `--survived` | survived-for-now / caution / "not yet killed" | **amber** |
| `--neutral-status` | process facts (e.g. cycle completed) | calm slate-blue |

### Type & spacing

- Type scale: `2xs / xs / sm / base / lg / xl / 2xl` (defined in
  `tailwind.config.js`). Base body is 14px; tight line-heights.
- Numbers use `.tnum` (tabular, monospaced) so columns of figures align — vital
  for financial data.
- Spacing: Tailwind's 4px scale; cards use `px-5 py-4`, tables `px-3 py-2.5`.
- Radius: `sm 0.3 / md 0.45 / lg 0.625` rem.

## Honesty in visuals (this product's core ethos)

The UI must editorialize **no more than the reporting agent does**.

1. **KILLED vs SURVIVED_FOR_NOW are instantly, unmistakably distinct.** Killed is
   red with a skull; survived is amber with a caution triangle.
2. **SURVIVED is never styled as success.** It is amber (caution), **never
   green**. Green is not used for any verdict — it would imply validation the
   critic never gave. Every survived/queued label literally reads
   "**Survived · not validated**".
3. **Caveats carry equal-or-greater weight than the headline number.** On the
   approval queue, the critic's surviving concerns lead (first column, amber
   left-border cards); the metrics sit beside them, not above them.
4. **No flattering number is dressed up.** A high OOS profit factor on a tiny
   trade count is shown next to an explicit "limited statistical power" caution.
   We never recompute or smooth a value — money/ratio figures arrive as verbatim
   strings from the API and are only formatted for reading.
5. **The caveat travels with the metric — everywhere.** A metric's
   trustworthiness caveat (e.g. a thin trade count → weak statistical power) must
   appear *wherever that metric is shown*, not only on one screen. The shared
   `sampleCaveat()` / `<SampleCaveat>` derive the caveat from the metric itself
   (its trade count) and render the same amber caution on the registry cards, the
   approval queue, and the backtest comparison alike. A survivor's OOS Sharpe is
   never shown bare. (Implemented in `src/lib/caveats.ts`.)
6. **A metric increase is not "good".** An in-sample→out-of-sample jump is shown
   with a NEUTRAL arrow, never green/positive — on a small sample a jump is noise,
   not improvement. Color stays reserved for verdict/status, never for deltas.
7. **Kills are the system working, not errors.** The critic rejecting weak
   candidates is the expected, healthy path, so kill *counts* read neutral; red is
   reserved for genuine failures (cycle aborted, budget breach, worker error).
   Each cycle row carries a plain-language summary ("2 proposed · both rejected by
   critic · none queued") so the numbers aren't mistaken for alarms.
8. **Realized P&L sign ≠ verdict.** The equity curve annotates the amount earned
   per window; a profit is tinted with a *muted* `pnlPos`/`pnlNeg` token
   (`--pnl-pos` / `--pnl-neg`), used ONLY for a realized financial figure in a
   single window. It is never the verdict/registry green — a green in-sample P&L
   means "this window made money", not "validated", and it always sits next to
   the out-of-sample figure and the critic's caveats.
9. **The equity curve is real or absent — never faked.** Per-bar curves are now
   persisted as a separate `BacktestArtifact` (captured verbatim from the engine's
   `BacktestResult`, kept OUT of the slim LLM-facing evidence). The dashboard
   charts the actual curve — in-sample then the sealed out-of-sample slice, with
   the OOS region demarcated and a break-even line at the starting balance. When a
   candidate has no persisted artifact, we say so plainly rather than drawing a
   line. Curve points are charted as numbers (a visual); every annotation reads
   from the verbatim string the API served.
10. **Honest empty/loading/error states**, never a void or a fabricated chart.

## Accessibility

- WCAG AA contrast on text and status colors against the dark surfaces.
- Full keyboard nav; a visible 2px accent focus ring (`:focus-visible`).
- Semantic HTML (`nav`, `main`, `table`/`th[scope]`, `figure`/`figcaption`,
  `ul`/`li`); `aria-label`s on nav, charts, and concern lists.
- `prefers-reduced-motion` disables the skeleton shimmer and transitions.
- Responsive down to laptop width; the sidebar collapses below `sm`.

## Note on tooling

We did **not** run the `shadcn/ui` CLI (it is interactive and rewrites Tailwind
config). Instead the primitives in `src/components/ui/*` are written by hand in
the shadcn idiom — `class-variance-authority` variants, a `cn()` merge helper,
and CSS-variable tokens — so they are fully owned, dependency-light, and
consistent with the token set above. Charts use Recharts (the library Tremor is
built on) styled to match these tokens.
