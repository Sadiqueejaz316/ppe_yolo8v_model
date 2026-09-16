import { useCallback, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { DashboardHeader } from "./components/DashboardHeader";
import { usePoll } from "./hooks/usePoll";
import { CamerasPage } from "./pages/CamerasPage";
import { DashboardPage } from "./pages/DashboardPage";
import { EventDetailPage } from "./pages/EventDetailPage";
import { EventsPage } from "./pages/EventsPage";
import { api, formatTime } from "./services/api";

export function App() {
  const [lastUpdated, setLastUpdated] = useState<string>(() => formatTime(new Date().toISOString()));

  const load = useCallback(async () => {
    const res = await api.health();
    if (res.timestamp) {
      setLastUpdated(formatTime(res.timestamp));
    }
    return res;
  }, []);

  const { data, error } = usePoll(load, 3000);
  const systemOk = error ? false : Boolean(data?.ok);
  const mock = Boolean(data?.mock);
  const statusLabel = error
    ? "Connecting..."
    : data?.ok
      ? "Monitoring active"
      : "Standby";

  return (
    <div className="app-shell">
      <DashboardHeader
        systemOk={systemOk}
        mock={mock}
        statusLabel={statusLabel}
        cameraName="Plant Entrance"
      />

      <main className="main-content">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/events" element={<EventsPage />} />
          <Route path="/events/:id" element={<EventDetailPage />} />
          <Route path="/cameras" element={<CamerasPage />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </main>

      <footer className="operator-footer">
        <div className="footer-left">
          <span className="footer-plant">Tata Steel · Industrial PPE Monitoring</span>
          <span className="footer-sep">•</span>
          <span className="footer-zone">Zone: General Safety Area</span>
        </div>
        <div className="footer-right">
          <span className="footer-status">
            {systemOk ? `Monitoring active · Updated ${lastUpdated}` : "Reconnecting to safety monitor…"}
          </span>
        </div>
      </footer>
    </div>
  );
}
