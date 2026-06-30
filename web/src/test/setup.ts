import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// recharts' ResponsiveContainer needs a measured size; jsdom reports 0. Stub
// ResizeObserver and element sizes so chart components render without warnings.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = globalThis.ResizeObserver ?? (ResizeObserverStub as never);

Object.defineProperty(globalThis.HTMLElement.prototype, "offsetWidth", {
  configurable: true,
  value: 640,
});
Object.defineProperty(globalThis.HTMLElement.prototype, "offsetHeight", {
  configurable: true,
  value: 240,
});
