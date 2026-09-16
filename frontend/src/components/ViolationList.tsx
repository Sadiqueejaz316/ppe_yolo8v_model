import { Link } from "react-router-dom";
import { api, formatMissingPpe, formatTime, getCameraDisplayName } from "../services/api";
import type { ActiveViolation, EventRow } from "../types";

type Props = {
  violations: ActiveViolation[];
  events: EventRow[];
  timestamp?: string;
};

export function ViolationList({ violations, events, timestamp }: Props) {
  if (!violations.length) {
    return (
      <section className="attention-panel safe">
        <div className="attention-header">
          <div className="attention-title-group">
            <span className="attention-icon ok">✓</span>
            <h2 className="attention-title">NEEDS ATTENTION NOW</h2>
          </div>
          <span className="count-pill zero">0</span>
        </div>
        <div className="attention-empty">
          <div className="safe-badge">ALL CLEAR</div>
          <p className="safe-msg">All workers currently in view are wearing required PPE.</p>
          <span className="safe-subtext">Continuous compliance monitoring active</span>
        </div>
      </section>
    );
  }

  return (
    <section className="attention-panel active">
      <div className="attention-header">
        <div className="attention-title-group">
          <span className="attention-icon bad pulse">🔴</span>
          <h2 className="attention-title">NEEDS ATTENTION NOW</h2>
        </div>
        <span className="count-pill active">{violations.length}</span>
      </div>

      <div className="attention-list">
        {violations.map((item) => {
          const key = `${item.camera_id}-${item.person_id}`;
          const match = events.find(
            (event) => event.person_id === item.person_id && event.camera_id === item.camera_id,
          );
          const locationName = getCameraDisplayName(item.camera_id);
          const timeString = match?.timestamp ? formatTime(match.timestamp) : formatTime(timestamp);
          const missingText = formatMissingPpe(item.missing_ppe);

          return (
            <article className="attention-card" key={key}>
              <div className="attention-card-header">
                <div className="worker-id-badge">
                  <span className="alert-marker">●</span>
                  <span className="worker-name">Worker #{item.person_id}</span>
                </div>
                <span className="attention-time">{timeString}</span>
              </div>

              <div className="attention-card-body">
                <div className="missing-ppe-banner">
                  <span className="missing-label">Missing:</span>
                  <strong className="missing-items">{missingText}</strong>
                </div>

                <div className="ppe-item-status-row">
                  <span className={`ppe-check-chip ${item.helmet === "COMPLIANT" ? "ok" : item.helmet === "MISSING" ? "bad" : "neutral"}`}>
                    Helmet {item.helmet === "COMPLIANT" ? "✓" : item.helmet === "MISSING" ? "✗" : "—"}
                  </span>
                  <span className={`ppe-check-chip ${item.vest === "COMPLIANT" ? "ok" : item.vest === "MISSING" ? "bad" : "neutral"}`}>
                    Vest {item.vest === "COMPLIANT" ? "✓" : item.vest === "MISSING" ? "✗" : "—"}
                  </span>
                  {item.mask && item.mask !== "UNKNOWN" && (
                    <span className={`ppe-check-chip ${item.mask === "COMPLIANT" ? "ok" : item.mask === "MISSING" ? "bad" : "neutral"}`}>
                      Mask {item.mask === "COMPLIANT" ? "✓" : item.mask === "MISSING" ? "✗" : "—"}
                    </span>
                  )}
                </div>

                <div className="attention-card-meta">
                  <span className="attention-location">📍 {locationName}</span>
                  {match?.has_evidence ? (
                    <Link to={`/events/${match.event_id}`} className="evidence-link">
                      View Evidence →
                    </Link>
                  ) : match ? (
                    <Link to={`/events/${match.event_id}`} className="evidence-link subtle">
                      View Alert →
                    </Link>
                  ) : null}
                </div>
              </div>

              {match?.has_evidence && (
                <div className="attention-card-thumbnail">
                  <Link to={`/events/${match.event_id}`} title="Click to view full evidence">
                    <img
                      src={api.evidenceUrl(match.event_id)}
                      alt={`Evidence for Worker #${item.person_id}`}
                      onError={(e) => {
                        (e.target as HTMLElement).style.display = "none";
                      }}
                    />
                  </Link>
                </div>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}
