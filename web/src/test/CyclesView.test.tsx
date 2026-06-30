import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";

vi.mock("@/lib/api", () => ({
  ApiError: class ApiError extends Error {},
  api: { cycles: vi.fn() },
}));

import { api } from "@/lib/api";
import { CyclesView } from "@/views/CyclesView";
import { renderRoute } from "./utils";
import { cycleSummary } from "./fixtures";

const cycles = vi.mocked(api.cycles);

describe("CyclesView", () => {
  it("renders cycles with verbatim cost and counts", async () => {
    cycles.mockResolvedValue([cycleSummary]);
    renderRoute(<CyclesView />, { path: "/", route: "/" });
    expect(await screen.findByText("cycle-abc123")).toBeInTheDocument();
    // Decimal cost passes through verbatim as a string, not floated.
    expect(screen.getByText("$0.1734")).toBeInTheDocument();
    expect(screen.getByText("Completed")).toBeInTheDocument();
    // Plain-language narrative de-alarms the kill count (kills = system working).
    expect(screen.getByText(/2 rejected by critic/)).toBeInTheDocument();
    expect(screen.getByText(/1 queued for review/)).toBeInTheDocument();
  });

  it("shows a designed empty state on day one", async () => {
    cycles.mockResolvedValue([]);
    renderRoute(<CyclesView />);
    expect(await screen.findByText("No cycles yet")).toBeInTheDocument();
    expect(screen.getByText(/run_orchestrator/)).toBeInTheDocument();
  });

  it("shows an error state when the API is unreachable", async () => {
    cycles.mockRejectedValue(new Error("Could not reach the API. Is uvicorn running?"));
    renderRoute(<CyclesView />);
    expect(await screen.findByText("Couldn’t load this view")).toBeInTheDocument();
    expect(screen.getByText(/uvicorn/)).toBeInTheDocument();
  });

  it("shows a loading skeleton before data resolves", () => {
    cycles.mockReturnValue(new Promise(() => {}));
    renderRoute(<CyclesView />);
    expect(screen.getByLabelText("Loading")).toBeInTheDocument();
  });
});
