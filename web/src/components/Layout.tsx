import { NavLink, Outlet } from "react-router-dom";
import {
  Activity,
  ListChecks,
  FlaskConical,
  GitBranch,
  Globe,
  Scale,
  ShieldAlert,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/cycles", label: "Cycles", icon: Activity },
  { to: "/universe", label: "Universe", icon: Globe },
  { to: "/registry", label: "Registry", icon: GitBranch },
  { to: "/approval-queue", label: "Approval queue", icon: ListChecks },
  { to: "/backtests", label: "Backtests", icon: FlaskConical },
  { to: "/economics", label: "Economics", icon: Scale },
];

export function Layout() {
  return (
    <div className="mx-auto flex min-h-screen max-w-screen flex-col">
      <PaperBanner />
      <div className="flex flex-1">
        <Sidebar />
        <main className="min-w-0 flex-1 px-6 py-6 lg:px-8" id="main">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

function PaperBanner() {
  return (
    <header
      className="flex flex-wrap items-center justify-between gap-2 border-b border-border bg-surface px-6 py-2.5"
      role="banner"
    >
      <div className="flex items-center gap-2.5">
        <span className="text-sm font-semibold tracking-tight text-foreground">4xPrima</span>
        <span className="text-2xs text-muted-foreground">Research dashboard</span>
      </div>
      <p className="flex items-center gap-1.5 text-2xs font-medium text-survived">
        <ShieldAlert className="h-3.5 w-3.5" aria-hidden />
        PAPER RESEARCH SYSTEM · read-only · nothing here authorizes a trade
      </p>
    </header>
  );
}

function Sidebar() {
  return (
    <nav
      className="hidden w-52 shrink-0 border-r border-border bg-surface px-3 py-5 sm:block"
      aria-label="Primary"
    >
      <ul className="flex flex-col gap-0.5">
        {NAV.map(({ to, label, icon: Icon }) => (
          <li key={to}>
            <NavLink
              to={to}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-accent/10 text-accent"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )
              }
            >
              <Icon className="h-4 w-4" aria-hidden />
              {label}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}
