import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";

vi.mock("@/lib/api", () => ({
  ApiError: class ApiError extends Error {},
  api: { approvalQueue: vi.fn() },
}));

import { api } from "@/lib/api";
import { ApprovalQueueView } from "@/views/ApprovalQueueView";
import { renderRoute } from "./utils";
import { approvalItem } from "./fixtures";

const approvalQueue = vi.mocked(api.approvalQueue);

describe("ApprovalQueueView", () => {
  it("shows the critic's concerns prominently and the reporting framing", async () => {
    approvalQueue.mockResolvedValue([approvalItem]);
    renderRoute(<ApprovalQueueView />);

    expect(await screen.findByText("What the critic still worries about")).toBeInTheDocument();
    expect(screen.getByText("Only 6 out-of-sample trades.")).toBeInTheDocument();
    expect(screen.getByText(/did not kill this/)).toBeInTheDocument();
    expect(screen.getByText(/Survived · not validated/)).toBeInTheDocument();
  });

  it("exposes NO approve action in this read-only slice", async () => {
    approvalQueue.mockResolvedValue([approvalItem]);
    renderRoute(<ApprovalQueueView />);
    await screen.findByText("What the critic still worries about");

    // No actionable approve/reject button exists yet.
    expect(screen.queryByRole("button", { name: /approve|reject/i })).toBeNull();
    // The action slot is present but inert (reserved for a later slice).
    const slot = screen.getByText("Operator decision");
    expect(slot).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByText(/Read-only\./)).toBeInTheDocument();
  });

  it("shows the empty state when nothing is queued", async () => {
    approvalQueue.mockResolvedValue([]);
    renderRoute(<ApprovalQueueView />);
    expect(await screen.findByText("Nothing awaiting review")).toBeInTheDocument();
  });
});
