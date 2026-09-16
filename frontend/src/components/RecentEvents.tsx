import { Link } from "react-router-dom";
import { api, formatMissingPpe, formatTime, getCameraDisplayName } from "../services/api";
import type { EventRow } from "../types";

type Props = {
  events: EventRow[];
};

export function RecentEvents({ events }: Props) {
  return (
    <section className="recent-alerts-section">
      <div className="section-header">
        <div className="section-title-group">
          <h2 className="section-title">RECENT SAFETY ALERTS</h2>
          <span className="section-subtitle">Confirmed PPE compliance events</span>
        </div>
        <Link to="/events" className="all-alerts-link">
          View full alert history ({events.length}) →
        </Link>
      </div>

      {!events.length ? (
        <div className="empty-alerts-card">
          <div className="empty-icon-shield">✓</div>
          <h3>No recent alerts</h3>
          <p>Everyone has been compliant so far.</p>
        </div>
      ) : (
        <div className="alerts-card-grid">
          {events.slice(0, 6).map((event) => {
            const friendlyLocation = getCameraDisplayName(event.camera_name || event.location || event.camera_id);
            const friendlyTime = formatTime(event.timestamp);
            const missingItems = event.ppe?.missing_ppe || (event.violation_type ? [event.violation_type] : []);
            const missingText = formatMissingPpe(missingItems);

            return (
              <article key={event.event_id} className="recent-alert-card">
                <div className="alert-card-thumbnail">
                  {event.has_evidence ? (
                    <img
                      src={api.evidenceUrl(event.event_id)}
                      alt={`Evidence for Worker #${event.person_id}`}
                      loading="lazy"
                      onError={(e) => {
                        (e.target as HTMLElement).parentElement!.innerHTML =
                          '<div class="no-thumb-placeholder">📷</div>';
                      }}
                    />
                  ) : (
                    <div className="no-thumb-placeholder">
                      <span className="no-thumb-icon">🛡️</span>
                    </div>
                  )}
                </div>

                <div className="alert-card-content">
                  <div className="alert-card-top">
                    <span className="alert-worker-badge">Worker #{event.person_id}</span>
                    <span className="alert-timestamp">{friendlyTime}</span>
                  </div>

                  <div className="alert-missing-summary">
                    <span className="missing-label">Missing:</span>{" "}
                    <strong className="missing-bold">{missingText}</strong>
                  </div>

                  <div className="alert-card-bottom">
                    <span className="alert-location-tag">📍 {friendlyLocation}</span>
                    <Link to={`/events/${event.event_id}`} className="view-alert-btn">
                      View Details →
                    </Link>
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
