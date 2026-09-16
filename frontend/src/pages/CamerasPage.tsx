import { useCallback } from "react";
import { CameraStatus } from "../components/CameraStatus";
import { usePoll } from "../hooks/usePoll";
import { api } from "../services/api";

export function CamerasPage() {
  const load = useCallback(() => api.cameras(), []);
  const { data, error } = usePoll(load, 3000);

  if (error && !data) {
    return (
      <div className="page cameras-page">
        <div className="error-card">
          <div className="error-icon">⚠️</div>
          <h3>Camera Status Unavailable</h3>
          <p>Can't reach the safety monitor to fetch camera feeds. Retrying…</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="page cameras-page loading-view">
        <div className="loading-spinner" />
        <p className="loading-text">Loading monitored camera feeds…</p>
      </div>
    );
  }

  return (
    <div className="page cameras-page">
      <div className="page-title-strip">
        <div>
          <h1 className="page-heading">CAMERAS</h1>
          <p className="page-subheading">
            Live industrial safety monitoring points and hardware feeds
          </p>
        </div>
        <div className="cameras-summary-pill">
          {data.filter((c) => c.status === "ONLINE" || c.status === "MOCK").length} of {data.length} Online
        </div>
      </div>

      {!data.length ? (
        <div className="empty-cameras-card">
          <div className="empty-icon">📷</div>
          <h3>No cameras configured</h3>
          <p>No video capture sources or cameras were detected in the system configuration.</p>
        </div>
      ) : (
        <div className="cameras-grid">
          {data.map((camera) => (
            <CameraStatus key={camera.id} camera={camera} />
          ))}
        </div>
      )}
    </div>
  );
}
