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
  it("renders the equity curve, amount earned, and the IS-vs-OOS comparison", async () => {
    backtest.mockResolvedValue(backtestDetail);
    renderRoute(<BacktestDetailView />, {
      path: "/backtests/:configHash",
      route: "/backtests/is-survivor-hash",
    });

    expect(await screen.findByText("Equity curve")).toBeInTheDocument();
    expect(screen.getByTestId("equity-curve")).toBeInTheDocument();
    // amount earned annotated per window, verbatim, with sign
    expect(screen.getByText("In-sample earned")).toBeInTheDocument();
    expect(screen.getByText("+$8,200.00")).toBeInTheDocument();
    expect(screen.getByText("+$360.00")).toBeInTheDocument();
    // the IS-vs-OOS comparison table is still present
    expect(screen.getByText("In-sample vs out-of-sample")).toBeInTheDocument();
    expect(screen.getAllByText("Sharpe ratio").length).toBeGreaterThan(0);
  });

  it("falls back to an honest notice when no curve artifact exists", async () => {
    backtest.mockResolvedValue({
      ...backtestDetail,
      in_sample_artifact: null,
      out_of_sample_artifact: null,
      equity_curve_available: false,
      equity_curve_notice: "Equity-curve points are not persisted for this candidate.",
    });
    renderRoute(<BacktestDetailView />, {
      path: "/backtests/:configHash",
      route: "/backtests/is-survivor-hash",
    });
    expect(await screen.findByText(/not persisted/)).toBeInTheDocument();
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
