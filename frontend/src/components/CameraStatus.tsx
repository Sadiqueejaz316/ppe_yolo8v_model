import { useState } from "react";
import { getCameraDisplayName } from "../services/api";
import type { CameraRow } from "../types";

export function CameraStatus({ camera }: { camera: CameraRow }) {
  const [showTechDetails, setShowTechDetails] = useState(false);
  const friendlyName = getCameraDisplayName(camera);

  const getStatusInfo = (status: string) => {
    switch (status.toUpperCase()) {
      case "ONLINE":
        return { label: "Monitoring Active", tone: "safe", dot: "ok pulse" };
      case "MOCK":
        return { label: "Demo Mode", tone: "warn", dot: "warn" };
      case "OFFLINE":
        return { label: "Offline", tone: "attention", dot: "bad" };
      default:
        return { label: "No Live Feed", tone: "neutral", dot: "muted" };
    }
  };

  const statusInfo = getStatusInfo(camera.status);

  return (
    <article className="camera-card">
      <div className="camera-card-top">
        <div className="camera-name-group">
          <h2 className="camera-friendly-name">{friendlyName}</h2>
          <span className="camera-zone-tag">{camera.zone ? `${camera.zone} Zone` : "General Zone"}</span>
        </div>
        <div className={`camera-status-pill tone-${statusInfo.tone}`}>
          <span className={`pill-dot ${statusInfo.dot}`} />
          <span className="pill-text">{statusInfo.label}</span>
        </div>
      </div>

      <div className="camera-card-meta">
        <div className="meta-row">
          <span className="meta-k">Location:</span>
          <span className="meta-v">{camera.location || friendlyName}</span>
        </div>
        <div className="meta-row">
          <span className="meta-k">Live Snapshot Feed:</span>
          <span className="meta-v">{camera.has_snapshot ? "Available ✓" : "Standby"}</span>
        </div>
      </div>

      <div className="camera-card-actions">
        <button
          type="button"
          className="tech-toggle-btn"
          onClick={() => setShowTechDetails((prev) => !prev)}
        >
          {showTechDetails ? "Hide technical details ▴" : "Technical details ▾"}
        </button>
      </div>

      {showTechDetails && (
        <div className="camera-tech-tray">
          <div className="tech-item">
            <span className="tech-k">Hardware / Camera ID:</span>
            <code>{camera.id}</code>
          </div>
          <div className="tech-item">
            <span className="tech-k">Target Ingestion Rate:</span>
            <span>{camera.target_fps ? `${camera.target_fps} FPS` : "10 FPS"}</span>
          </div>
          {camera.camera_fps != null && (
            <div className="tech-item">
              <span className="tech-k">Camera Hardware FPS:</span>
              <span>{camera.camera_fps.toFixed(1)} FPS</span>
            </div>
          )}
          {camera.inference_fps != null && (
            <div className="tech-item">
              <span className="tech-k">Vision Inference FPS:</span>
              <span>{camera.inference_fps.toFixed(1)} FPS</span>
            </div>
          )}
        </div>
      )}
    </article>
  );
}
