import { useState } from "react";
import { useSnapshot } from "../hooks/useSnapshot";
import { getCameraDisplayName } from "../services/api";
import type { CameraRow } from "../types";

type Props = {
  camera?: CameraRow;
  snapshotIntervalMs?: number;
  mock?: boolean;
};

export function CameraView({ camera, snapshotIntervalMs = 120, mock = false }: Props) {
  const [showTechDetails, setShowTechDetails] = useState(false);
  const snapshotUrl = useSnapshot(camera?.id, snapshotIntervalMs, Boolean(camera?.has_snapshot));
  const friendlyName = getCameraDisplayName(camera);

  if (!camera) {
    return (
      <section className="camera-panel">
        <div className="camera-header">
          <div className="camera-title-group">
            <span className="camera-title">Live Camera View</span>
          </div>
        </div>
        <div className="camera-viewport empty">
          <div className="camera-empty-state">
            <div className="empty-icon">📷</div>
            <h3>No camera configured</h3>
            <p>Please configure a camera source in settings.</p>
          </div>
        </div>
      </section>
    );
  }

  if (!camera.has_snapshot) {
    return (
      <section className="camera-panel">
        <div className="camera-header">
          <div className="camera-title-group">
            <span className="camera-title">{friendlyName}</span>
            <span className="camera-zone-badge">{camera.zone ? `${camera.zone} zone` : "General zone"}</span>
          </div>
          <div className="camera-badges">
            <span className="feed-badge standby">STANDBY</span>
          </div>
        </div>
        <div className="camera-viewport waiting">
          <div className="camera-empty-state">
            <div className="spinner" />
            <h3>Live view isn't available yet</h3>
            <p>Check that camera monitoring is running.</p>
            <span className="retrying-text">Retrying feed connection…</span>
          </div>
        </div>
        <div className="camera-footer">
          <button
            type="button"
            className="tech-toggle"
            onClick={() => setShowTechDetails((prev) => !prev)}
          >
            {showTechDetails ? "Hide technical details ▴" : "Technical details ▾"}
          </button>
          {showTechDetails && (
            <div className="tech-details-tray">
              <div><span>Camera ID:</span> <code>{camera.id}</code></div>
              <div><span>Status:</span> <span>{camera.status}</span></div>
              <div><span>Target Rate:</span> <span>{camera.target_fps || 10} FPS</span></div>
            </div>
          )}
        </div>
      </section>
    );
  }

  return (
    <section className="camera-panel">
      <div className="camera-header">
        <div className="camera-title-group">
          <span className="camera-title">{friendlyName}</span>
          <span className="camera-zone-badge">{camera.zone ? `${camera.zone} zone` : "General zone"}</span>
        </div>
        <div className="camera-badges">
          {mock ? (
            <span className="feed-badge demo">DEMO</span>
          ) : (
            <span className="feed-badge live">
              <span className="live-dot" /> LIVE
            </span>
          )}
        </div>
      </div>

      <div className="camera-viewport">
        {snapshotUrl ? (
          <img
            src={snapshotUrl}
            alt={`${friendlyName} live monitoring view`}
            className="camera-stream-image"
          />
        ) : (
          <div className="camera-empty-state">
            <div className="spinner" />
            <p>Connecting to live camera feed…</p>
          </div>
        )}
      </div>

      <div className="camera-footer">
        <span className="feed-caption">
          {friendlyName} · Continuous PPE detection active
        </span>
        <button
          type="button"
          className="tech-toggle"
          onClick={() => setShowTechDetails((prev) => !prev)}
        >
          {showTechDetails ? "Hide diagnostics ▴" : "Diagnostics ▾"}
        </button>
      </div>

      {showTechDetails && (
        <div className="tech-details-tray">
          <div><span>Camera ID:</span> <code>{camera.id}</code></div>
          <div><span>Status:</span> <span>{camera.status}</span></div>
          <div>
            <span>Camera FPS:</span>{" "}
            <span>{camera.camera_fps != null ? `${camera.camera_fps.toFixed(1)} FPS` : "—"}</span>
          </div>
          <div>
            <span>Inference FPS:</span>{" "}
            <span>{camera.inference_fps != null ? `${camera.inference_fps.toFixed(1)} FPS` : "—"}</span>
          </div>
          <div><span>Location:</span> <span>{camera.location || camera.name}</span></div>
        </div>
      )}
    </section>
  );
}
