import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";

vi.mock("@/lib/api", () => ({
  ApiError: class ApiError extends Error {},
  api: { universe: vi.fn() },
}));

import { api } from "@/lib/api";
import { UniverseView } from "@/views/UniverseView";
import { renderRoute } from "./utils";
import { universeData, universeEmpty } from "./fixtures";

const universe = vi.mocked(api.universe);

describe("UniverseView", () => {
  it("shows admitted pairs and the honest 'why narrow' rationale", async () => {
    universe.mockResolvedValue(universeData);
    renderRoute(<UniverseView />);
    expect(await screen.findByText("Admitted · 2")).toBeInTheDocument();
    // pairs appear in both the table and the correlation matrix
    expect(screen.getAllByText("AUDUSD").length).toBeGreaterThan(0);
    // rationale explains structure, and that it is never chosen by return
    expect(document.body.textContent).toMatch(/selection bias/i);
    expect(document.body.textContent).toMatch(/multiple-testing/i);
  });

  it("shows a pair dropped for CORRELATION with its reason", async () => {
    universe.mockResolvedValue(universeData);
    renderRoute(<UniverseView />);
    await screen.findByText("Admitted · 2");
    expect(screen.getAllByText("NZDUSD").length).toBeGreaterThan(0);
    expect(screen.getByText(/1.00 with AUDUSD exceeds 0.80/)).toBeInTheDocument();
    // tagged as a correlation drop
    expect(screen.getAllByText("correlation").length).toBeGreaterThan(0);
  });

  it("shows a pair dropped for COST-TO-MOVE with its reason", async () => {
    universe.mockResolvedValue(universeData);
    renderRoute(<UniverseView />);
    await screen.findByText("Admitted · 2");
    expect(screen.getAllByText("USDCHF").length).toBeGreaterThan(0);
    expect(screen.getByText(/spread\/ATR 3.98 > 0.25/)).toBeInTheDocument();
    expect(screen.getAllByText("cost-to-move").length).toBeGreaterThan(0);
  });

  it("renders the correlation matrix", async () => {
    universe.mockResolvedValue(universeData);
    renderRoute(<UniverseView />);
    await screen.findByText("Admitted · 2");
    expect(screen.getByLabelText("Return correlation matrix")).toBeInTheDocument();
  });

  it("shows the honest empty state when no screen has run", async () => {
    universe.mockResolvedValue(universeEmpty);
    renderRoute(<UniverseView />);
    expect(await screen.findByText("No screen run yet")).toBeInTheDocument();
  });

  it("shows an error state when the API fails", async () => {
    universe.mockRejectedValue(new Error("Request failed (500)."));
    renderRoute(<UniverseView />);
    expect(await screen.findByText("Couldn’t load this view")).toBeInTheDocument();
  });
});
