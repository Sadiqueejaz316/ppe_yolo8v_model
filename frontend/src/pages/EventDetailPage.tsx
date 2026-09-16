import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { EventDetails } from "../components/EventDetails";
import { api } from "../services/api";
import type { EventRow } from "../types";

export function EventDetailPage() {
  const { id } = useParams();
  const [event, setEvent] = useState<EventRow | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    api
      .event(id)
      .then((row) => {
        if (!cancelled) setEvent(row);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message || "Safety alert could not be found.");
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  if (error) {
    return (
      <div className="page event-detail-page">
        <p>
          <Link to="/events" className="back-link">← Back to Alert History</Link>
        </p>
        <div className="error-card">
          <div className="error-icon">⚠️</div>
          <h3>Alert Not Found</h3>
          <p>{error}</p>
          <Link to="/events" className="view-details-btn">
            Return to Alert History
          </Link>
        </div>
      </div>
    );
  }

  if (!event) {
    return (
      <div className="page event-detail-page loading-view">
        <div className="loading-spinner" />
        <p className="loading-text">Loading safety alert details…</p>
      </div>
    );
  }

  return (
    <div className="page event-detail-page">
      <EventDetails event={event} />
    </div>
  );
}
