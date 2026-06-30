import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";

vi.mock("@/lib/api", () => ({
  ApiError: class ApiError extends Error {},
  api: { registry: vi.fn() },
}));

import { api } from "@/lib/api";
import { RegistryView } from "@/views/RegistryView";
import { renderRoute } from "./utils";
import { killedEntry, survivorEntry } from "./fixtures";

const registry = vi.mocked(api.registry);

describe("RegistryView", () => {
  it("renders killed and survived candidates with honest, distinct labels", async () => {
    registry.mockResolvedValue([survivorEntry, killedEntry]);
    renderRoute(<RegistryView />);

    // Survivor is labelled "not validated" — never as good/validated.
    expect(await screen.findAllByText(/not validated/i)).not.toHaveLength(0);
    // Killed is unmistakably killed.
    expect(screen.getAllByText(/^Killed$/).length).toBeGreaterThan(0);

    // No cheerleading language that would imply a survivor is good/endorsed.
    // (Its honest "not validated" label is asserted above.)
    expect(screen.queryByText(/recommended|good strategy|promising/i)).toBeNull();

    expect(screen.getByText(/USDJPY · H1/)).toBeInTheDocument();
    expect(screen.getByText(/EURUSD · H1/)).toBeInTheDocument();

    // The statistical-power caveat travels onto the registry card too: the
    // survivor's OOS Sharpe rests on 6 trades and must not be shown bare.
    expect(screen.getByText(/limited statistical power/i)).toBeInTheDocument();
  });

  it("shows the empty state when there are no candidates", async () => {
    registry.mockResolvedValue([]);
    renderRoute(<RegistryView />);
    expect(await screen.findByText("No candidates yet")).toBeInTheDocument();
  });
});
