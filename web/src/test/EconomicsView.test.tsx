import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";

vi.mock("@/lib/api", () => ({
  ApiError: class ApiError extends Error {},
  api: { economics: vi.fn() },
}));

import { api } from "@/lib/api";
import { EconomicsView } from "@/views/EconomicsView";
import { renderRoute } from "./utils";
import { economicsHealthy, economicsRetire } from "./fixtures";

const economics = vi.mocked(api.economics);

describe("EconomicsView", () => {
  it("flags a retire candidate honestly (amber, with the specific reason)", async () => {
    economics.mockResolvedValue([economicsRetire]);
    renderRoute(<EconomicsView />);
    expect(await screen.findByText("Retire candidate")).toBeInTheDocument();
    expect(screen.getByText(/Net expectancy negative after costs/)).toBeInTheDocument();
    // the thin-sample caveat travels alongside the OOS decay read
    expect(screen.getByText(/statistical-power floor/)).toBeInTheDocument();
  });

  it("shows a healthy candidate as economically OK", async () => {
    economics.mockResolvedValue([economicsHealthy]);
    renderRoute(<EconomicsView />);
    expect(await screen.findByText("Economics OK")).toBeInTheDocument();
    expect(screen.getByText(/Broker takes 4.8% of gross/)).toBeInTheDocument();
  });

  it("never shows win rate without avg win / avg loss / expectancy context", async () => {
    economics.mockResolvedValue([economicsHealthy]);
    renderRoute(<EconomicsView />);
    await screen.findByText("Economics OK");
    // win rate appears...
    expect(screen.getAllByText(/Win rate/).length).toBeGreaterThan(0);
    // ...always alongside avg win, avg loss, and net expectancy
    expect(screen.getAllByText(/Avg win/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Avg loss/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Net expectancy/).length).toBeGreaterThan(0);
  });

  it("leads with net P&L and shows gross as subordinate", async () => {
    economics.mockResolvedValue([economicsHealthy]);
    renderRoute(<EconomicsView />);
    await screen.findByText("Economics OK");
    expect(screen.getByText("Net P&L (in-sample)")).toBeInTheDocument();
    expect(screen.getAllByText(/Gross \(before cost\)/).length).toBeGreaterThan(0);
  });

  it("states the scope is historical, not live", async () => {
    economics.mockResolvedValue([economicsHealthy]);
    renderRoute(<EconomicsView />);
    expect(await screen.findByText(/historical backtest economics/i)).toBeInTheDocument();
  });

  it("shows the honest empty state with no data", async () => {
    economics.mockResolvedValue([]);
    renderRoute(<EconomicsView />);
    expect(await screen.findByText("No economics yet")).toBeInTheDocument();
  });

  it("shows an error state when the API fails", async () => {
    economics.mockRejectedValue(new Error("Request failed (500)."));
    renderRoute(<EconomicsView />);
    expect(await screen.findByText("Couldn’t load this view")).toBeInTheDocument();
  });
});
