import { NavLink } from "react-router-dom";

type Props = {
  systemOk: boolean;
  mock: boolean;
  statusLabel?: string;
  cameraName?: string;
};

export function DashboardHeader({ systemOk, mock, statusLabel, cameraName = "Plant Entrance" }: Props) {
  return (
    <header className="header">
      <div className="header-left">
        <div className="brand-group">
          <span className="brand-title">PPE SAFETY</span>
          <span className="brand-subtitle">INDUSTRIAL MONITOR</span>
        </div>
        <div className="header-location">
          <span className="location-pin">📍</span>
          <span className="location-name">{cameraName}</span>
        </div>
      </div>

      <nav className="nav" aria-label="Main Navigation">
        <NavLink to="/dashboard" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
          LIVE VIEW
        </NavLink>
        <NavLink to="/events" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
          ALERT HISTORY
        </NavLink>
        <NavLink to="/cameras" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
          CAMERAS
        </NavLink>
      </nav>

      <div className="header-right">
        {mock ? (
          <span className="demo-badge">
            <span className="demo-icon">▶</span> DEMO MODE · Recorded video
          </span>
        ) : null}
        <div className={`status-pill ${systemOk ? "online" : "offline"}`}>
          <span className={`status-dot ${systemOk ? "pulse" : ""}`} />
          <span className="status-text">{systemOk ? "MONITORING ACTIVE" : (statusLabel || "OFFLINE")}</span>
        </div>
      </div>
    </header>
  );
}
