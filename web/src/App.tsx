import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "@/components/Layout";
import { CyclesView } from "@/views/CyclesView";
import { CycleDetailView } from "@/views/CycleDetailView";
import { RegistryView } from "@/views/RegistryView";
import { ApprovalQueueView } from "@/views/ApprovalQueueView";
import { BacktestsView } from "@/views/BacktestsView";
import { BacktestDetailView } from "@/views/BacktestDetailView";
import { EconomicsView } from "@/views/EconomicsView";

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Navigate to="/cycles" replace />} />
        <Route path="cycles" element={<CyclesView />} />
        <Route path="cycles/:cycleId" element={<CycleDetailView />} />
        <Route path="registry" element={<RegistryView />} />
        <Route path="approval-queue" element={<ApprovalQueueView />} />
        <Route path="backtests" element={<BacktestsView />} />
        <Route path="backtests/:configHash" element={<BacktestDetailView />} />
        <Route path="economics" element={<EconomicsView />} />
        <Route path="*" element={<Navigate to="/cycles" replace />} />
      </Route>
    </Routes>
  );
}

export function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}
