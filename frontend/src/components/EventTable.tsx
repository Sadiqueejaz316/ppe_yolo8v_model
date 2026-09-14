import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, formatDateTime, formatMissingPpe, getCameraDisplayName } from "../services/api";
import type { EventRow } from "../types";

type Props = {
  events: EventRow[];
};

export function EventTable({ events }: Props) {
  const navigate = useNavigate();
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (!events.length) {
    return (
      <div className="empty-table-state">
        <div className="empty-table-icon">🛡️</div>
        <h3>No safety alerts found</h3>
        <p>No safety violations match the selected filters.</p>
      </div>
    );
  }

  return (
    <div className="alert-history-table-container">
      <table className="alert-history-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Location</th>
            <th>Worker</th>
            <th>Missing PPE</th>
            <th>Evidence</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {events.map((event) => {
            const location = getCameraDisplayName(event.camera_name || event.location || event.camera_id);
            const timeStr = formatDateTime(event.timestamp);
            const missing = event.ppe?.missing_ppe || (event.violation_type ? [event.violation_type] : []);
            const missingText = formatMissingPpe(missing);
            const isExpanded = expandedId === event.event_id;

            return (
              <tr
                key={event.event_id}
                className={`alert-table-row ${isExpanded ? "expanded" : ""}`}
              >
                <td className="time-cell">
                  <span className="time-primary">{timeStr}</span>
                </td>
                <td className="location-cell">
                  <span className="location-badge">📍 {location}</span>
                </td>
                <td className="worker-cell">
                  <span className="worker-pill">Worker #{event.person_id}</span>
                </td>
                <td className="violation-cell">
                  <span className="missing-ppe-tag">
                    <span className="alert-dot-mini">●</span> {missingText}
                  </span>
                </td>
                <td className="evidence-cell">
                  {event.has_evidence ? (
                    <button
                      type="button"
                      className="thumb-btn"
                      onClick={() => navigate(`/events/${event.event_id}`)}
                      title="View full evidence image"
                    >
                      <img
                        src={api.evidenceUrl(event.event_id)}
                        alt="Evidence thumbnail"
                        className="table-thumb"
                        onError={(e) => {
                          (e.target as HTMLElement).style.display = "none";
                        }}
                      />
                      <span className="thumb-label">Photo</span>
                    </button>
                  ) : (
                    <span className="no-evidence-text">No photo</span>
                  )}
                </td>
                <td className="action-cell">
                  <button
                    type="button"
                    className="view-details-btn"
                    onClick={() => navigate(`/events/${event.event_id}`)}
                  >
                    View Details →
                  </button>
                  <button
                    type="button"
                    className="tech-sublink"
                    onClick={(e) => {
                      e.stopPropagation();
                      setExpandedId(isExpanded ? null : event.event_id);
                    }}
                  >
                    {isExpanded ? "Less ▴" : "More ▾"}
                  </button>

                  {isExpanded && (
                    <div className="row-tech-tray">
                      <div><span>Event ID:</span> <code>{event.event_id}</code></div>
                      <div><span>Camera ID:</span> <code>{event.camera_id}</code></div>
                      <div>
                        <span>Confidence:</span>{" "}
                        <span>{event.confidence ? `${Math.round(event.confidence * 100)}%` : "—"}</span>
                      </div>
                      <div><span>Zone:</span> <span>{event.zone || "general"}</span></div>
                    </div>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
