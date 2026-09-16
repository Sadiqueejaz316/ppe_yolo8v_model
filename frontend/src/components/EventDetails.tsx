import { useState } from "react";
import { Link } from "react-router-dom";
import { api, formatDateTime, formatMissingPpe, getCameraDisplayName } from "../services/api";
import type { EventRow } from "../types";
import { PPEStatus } from "./PPEStatus";

export function EventDetails({ event }: { event: EventRow }) {
  const [showTechDetails, setShowTechDetails] = useState(false);
  const locationName = getCameraDisplayName(event.camera_name || event.location || event.camera_id);
  const formattedTime = formatDateTime(event.timestamp);
  const missingItems = event.ppe?.missing_ppe || (event.violation_type ? [event.violation_type] : []);
  const missingText = formatMissingPpe(missingItems);

  return (
    <div className="alert-detail-wrapper">
      <div className="alert-detail-top-nav">
        <Link to="/events" className="back-link">
          ← Back to Alert History
        </Link>
        <Link to="/dashboard" className="live-view-btn">
          Go to Live View →
        </Link>
      </div>

      <div className="alert-detail-hero">
        <div className="alert-hero-badge">
          <span className="hero-alert-dot">🔴</span>
          SAFETY VIOLATION DETECTED
        </div>
        <h1 className="alert-hero-title">Worker #{event.person_id} — {missingText}</h1>
        <div className="alert-hero-meta">
          <span>📍 {locationName}</span>
          <span>•</span>
          <span>🕒 {formattedTime}</span>
        </div>
      </div>

      <div className="alert-detail-grid">
        {/* LEFT COLUMN: EVIDENCE IMAGE (PRIMARY FOCUS) */}
        <section className="evidence-panel">
          <div className="panel-header">
            <h2 className="panel-title">PHOTO EVIDENCE</h2>
            {event.has_evidence && (
              <span className="evidence-pill ok">Captured & Confirmed</span>
            )}
          </div>

          <div className="evidence-frame">
            {event.has_evidence ? (
              <img
                src={api.evidenceUrl(event.event_id)}
                alt={`Photo evidence for Worker #${event.person_id}`}
                className="evidence-large-img"
                onError={(e) => {
                  (e.target as HTMLElement).style.display = "none";
                  (e.target as HTMLElement).parentElement!.innerHTML =
                    '<div class="evidence-error-state"><p>Photo evidence could not be loaded from local storage.</p></div>';
                }}
              />
            ) : (
              <div className="evidence-empty-state">
                <div className="empty-evidence-icon">📷</div>
                <h3>No photo evidence available</h3>
                <p>An evidence snapshot was not captured or has expired for this event.</p>
              </div>
            )}
          </div>
        </section>

        {/* RIGHT COLUMN: PPE STATE & INCIDENT SUMMARY */}
        <div className="incident-summary-column">
          <section className="incident-panel">
            <h2 className="panel-title">INCIDENT SUMMARY</h2>
            <div className="summary-list">
              <div className="summary-row">
                <span className="summary-label">Worker ID</span>
                <strong className="summary-val highlight">Worker #{event.person_id}</strong>
              </div>
              <div className="summary-row">
                <span className="summary-label">Location</span>
                <span className="summary-val">{locationName}</span>
              </div>
              <div className="summary-row">
                <span className="summary-label">Timestamp</span>
                <span className="summary-val">{formattedTime}</span>
              </div>
              <div className="summary-row">
                <span className="summary-label">Violation</span>
                <strong className="summary-val alert-text">{missingText}</strong>
              </div>
            </div>

            <div className="ppe-breakdown-box">
              <h3 className="sub-title">Detailed PPE Compliance</h3>
              <PPEStatus person={event.ppe} />
            </div>
          </section>

          {/* COLLAPSIBLE TECHNICAL DETAILS */}
          <section className="technical-collapsible-panel">
            <button
              type="button"
              className="tech-details-btn"
              onClick={() => setShowTechDetails((prev) => !prev)}
            >
              <span>⚙️ {showTechDetails ? "Hide technical diagnostics" : "Show technical diagnostics"}</span>
              <span>{showTechDetails ? "▴" : "▾"}</span>
            </button>

            {showTechDetails && (
              <div className="tech-fields-grid">
                <div className="tech-row">
                  <span className="tech-k">Event ID:</span>
                  <code className="tech-v">{event.event_id}</code>
                </div>
                <div className="tech-row">
                  <span className="tech-k">Camera ID:</span>
                  <code className="tech-v">{event.camera_id}</code>
                </div>
                <div className="tech-row">
                  <span className="tech-k">Raw Violation Key:</span>
                  <code className="tech-v">{event.violation_type}</code>
                </div>
                <div className="tech-row">
                  <span className="tech-k">Detection Confidence:</span>
                  <span className="tech-v">
                    {event.confidence != null ? `${(event.confidence * 100).toFixed(1)}%` : "—"}
                  </span>
                </div>
                <div className="tech-row">
                  <span className="tech-k">Zone:</span>
                  <span className="tech-v">{event.zone || "general"}</span>
                </div>
                <div className="tech-row">
                  <span className="tech-k">Raw Status:</span>
                  <span className="tech-v">{event.status}</span>
                </div>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
