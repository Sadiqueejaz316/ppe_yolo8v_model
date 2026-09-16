import { useCallback, useState } from "react";
import { EventTable } from "../components/EventTable";
import { usePoll } from "../hooks/usePoll";
import { api, getCameraDisplayName } from "../services/api";

export function EventsPage() {
  const [cameraId, setCameraId] = useState("");
  const [violationType, setViolationType] = useState("");

  const loadCameras = useCallback(() => api.cameras(), []);
  const loadEvents = useCallback(
    () =>
      api.events({
        camera_id: cameraId || undefined,
        violation_type: violationType || undefined,
      }),
    [cameraId, violationType],
  );

  const { data: cameras } = usePoll(loadCameras, 5000);
  const { data, error } = usePoll(loadEvents, 2000);

  return (
    <div className="page events-page">
      <div className="page-title-strip">
        <div>
          <h1 className="page-heading">ALERT HISTORY</h1>
          <p className="page-subheading">
            Review past safety alerts, missing PPE incidents, and photo evidence
          </p>
        </div>
      </div>

      <div className="filter-toolbar">
        <div className="filter-group">
          <label htmlFor="camera-filter" className="filter-label">Location / Camera:</label>
          <select
            id="camera-filter"
            className="operator-select"
            value={cameraId}
            onChange={(e) => setCameraId(e.target.value)}
          >
            <option value="">All Locations</option>
            {(cameras || []).map((cam) => (
              <option key={cam.id} value={cam.id}>
                {getCameraDisplayName(cam)} ({cam.id})
              </option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label htmlFor="violation-filter" className="filter-label">PPE Requirement:</label>
          <select
            id="violation-filter"
            className="operator-select"
            value={violationType}
            onChange={(e) => setViolationType(e.target.value)}
          >
            <option value="">All Violations</option>
            <option value="HELMET_MISSING">Missing Helmet</option>
            <option value="SAFETY_VEST_MISSING">Missing Safety Vest</option>
            <option value="MASK_MISSING">Missing Mask</option>
          </select>
        </div>

        {error && (
          <div className="filter-error-pill">
            ⚠️ Notice: Alert history sync delayed. Retrying…
          </div>
        )}
      </div>

      <section className="events-table-panel">
        <EventTable events={data || []} />
      </section>
    </div>
  );
}
