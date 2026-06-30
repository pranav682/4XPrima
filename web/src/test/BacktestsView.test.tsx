import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";

vi.mock("@/lib/api", () => ({
  ApiError: class ApiError extends Error {},
  api: { registry: vi.fn(), backtest: vi.fn() },
}));

import { api } from "@/lib/api";
import { BacktestsView } from "@/views/BacktestsView";
import { BacktestDetailView } from "@/views/BacktestDetailView";
import { renderRoute } from "./utils";
import { backtestDetail, survivorEntry } from "./fixtures";

const registry = vi.mocked(api.registry);
const backtest = vi.mocked(api.backtest);

describe("BacktestsView (index)", () => {
  it("lists candidates with stored evidence", async () => {
    registry.mockResolvedValue([survivorEntry]);
    renderRoute(<BacktestsView />);
    expect(await screen.findByText(/USDJPY · H1/)).toBeInTheDocument();
    expect(screen.getByText("Open")).toBeInTheDocument();
  });

  it("shows the empty state with no evidence", async () => {
    registry.mockResolvedValue([]);
    renderRoute(<BacktestsView />);
    expect(await screen.findByText("No backtests yet")).toBeInTheDocument();
  });
});

describe("BacktestDetailView", () => {
  it("renders the in-sample vs OOS comparison and an honest equity-curve notice", async () => {
    backtest.mockResolvedValue(backtestDetail);
    renderRoute(<BacktestDetailView />, {
      path: "/backtests/:configHash",
      route: "/backtests/is-survivor-hash",
    });

    expect(await screen.findByText("In-sample vs out-of-sample")).toBeInTheDocument();
    // both segments present in the comparison table
    expect(screen.getAllByText("Sharpe ratio").length).toBeGreaterThan(0);
    // the equity curve is honestly reported as unavailable, not fabricated
    expect(screen.getByText("Equity curve")).toBeInTheDocument();
    expect(screen.getByText(/not persisted in BacktestEvidence/)).toBeInTheDocument();
  });

  it("shows an error state when the backtest cannot be loaded", async () => {
    backtest.mockRejectedValue(new Error("Request failed (404)."));
    renderRoute(<BacktestDetailView />, {
      path: "/backtests/:configHash",
      route: "/backtests/missing",
    });
    expect(await screen.findByText("Couldn’t load this view")).toBeInTheDocument();
  });
});
