/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Near-monochrome surface ramp (Linear-like). Color = status, not decor.
        background: "hsl(var(--background))",
        surface: "hsl(var(--surface))",
        elevated: "hsl(var(--elevated))",
        border: "hsl(var(--border))",
        input: "hsl(var(--border))",
        ring: "hsl(var(--ring))",
        foreground: "hsl(var(--foreground))",
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        // The ONE functional accent — interactive / focus / nav only.
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        // Status semantics (see DESIGN.md). SURVIVED is amber (caution, "not yet
        // killed") and is deliberately NOT green — green would imply validation.
        killed: "hsl(var(--killed))",
        survived: "hsl(var(--survived))",
        caution: "hsl(var(--survived))",
        neutralStatus: "hsl(var(--neutral-status))",
        danger: "hsl(var(--killed))",
        // Realized P&L sign only — see index.css / DESIGN.md.
        pnlPos: "hsl(var(--pnl-pos))",
        pnlNeg: "hsl(var(--pnl-neg))",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      fontSize: {
        // Tight type scale.
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
        xs: ["0.75rem", { lineHeight: "1.1rem" }],
        sm: ["0.8125rem", { lineHeight: "1.25rem" }],
        base: ["0.875rem", { lineHeight: "1.4rem" }],
        lg: ["1rem", { lineHeight: "1.5rem" }],
        xl: ["1.25rem", { lineHeight: "1.75rem" }],
        "2xl": ["1.625rem", { lineHeight: "2rem" }],
      },
      borderRadius: {
        lg: "0.625rem",
        md: "0.45rem",
        sm: "0.3rem",
      },
      maxWidth: {
        screen: "1440px",
      },
    },
  },
  plugins: [],
};
