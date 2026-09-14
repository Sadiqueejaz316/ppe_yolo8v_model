import { useCallback, useState } from "react";
import { CameraView } from "../components/CameraView";
import { PPEStatus } from "../components/PPEStatus";
import { RecentEvents } from "../components/RecentEvents";
import { SafetyStatusBanner } from "../components/SafetyStatusBanner";
import { StatusCard } from "../components/StatusCard";
import { ViolationList } from "../components/ViolationList";
import { usePoll } from "../hooks/usePoll";
import { api, getCameraDisplayName } from "../services/api";

export function DashboardPage() {
  const [intervalMs, setIntervalMs] = useState(1000);
  const [snapshotIntervalMs, setSnapshotIntervalMs] = useState(120);

  const load = useCallback(async () => {
    const summary = await api.summary();
    if (summary.poll_interval_ms && summary.poll_interval_ms >= 200) {
      setIntervalMs(summary.poll_interval_ms);
    }
    if (summary.snapshot_poll_interval_ms && summary.snapshot_poll_interval_ms >= 50) {
      setSnapshotIntervalMs(summary.snapshot_poll_interval_ms);
    }
    return summary;
  }, []);

  const { data, error } = usePoll(load, intervalMs);

  if (error && !data) {
    return (
      <div className="page error-view">
        <SafetyStatusBanner
          systemOk={false}
          liveAvailable={false}
          totalPeople={0}
          violations={0}
          mock={false}
          errorMessage="Can't reach the safety monitor. Check the network connection or restart the monitoring application."
        />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="page loading-view">
        <div className="loading-spinner" />
        <p className="loading-text">Connecting to PPE Safety Monitor…</p>
      </div>
    );
  }

  const camera = data.cameras[0];
  const friendlyCameraName = getCameraDisplayName(camera);
  const activeViolationsCount = data.active_violations?.length ?? data.violations ?? 0;
  const isAllSafe = activeViolationsCount === 0 && data.total_people > 0;

  return (
    <div className="page dashboard-page">
      {/* 1. PRIMARY SAFETY HERO STATUS BANNER */}
      <SafetyStatusBanner
        systemOk={data.system_ok}
        liveAvailable={data.live_available}
        totalPeople={data.total_people}
        violations={activeViolationsCount}
        mock={data.mock}
        cameraName={friendlyCameraName}
      />

      {/* 2. MAIN SPLIT: CAMERA VIEW & NEEDS ATTENTION NOW */}
      <div className="live-operator-layout">
        <div className="camera-column">
          <CameraView
            camera={camera}
            snapshotIntervalMs={snapshotIntervalMs}
            mock={data.mock}
          />
        </div>
        <div className="attention-column">
          <ViolationList
            violations={data.active_violations || []}
            events={data.recent_events || []}
            timestamp={data.timestamp}
          />
        </div>
      </div>

      {/* 3. SAFETY METRIC CARDS */}
      <div className="metric-strip" aria-label="Key Safety Metrics">
        <StatusCard
          label="Workers in view"
          value={data.total_people}
          tone="neutral"
          subtext="Active in monitored zone"
        />
        <StatusCard
          label="Wearing required PPE"
          value={data.compliant_people}
          tone="safe"
          subtext="Fully compliant with zone rules"
        />
        <StatusCard
          label="Needs attention"
          value={activeViolationsCount}
          tone={activeViolationsCount > 0 ? "attention" : "safe"}
          subtext={activeViolationsCount > 0 ? "Missing helmet or vest" : "Zero active violations"}
        />
        <StatusCard
          label="Cameras online"
          value={`${data.cameras_online}/${data.cameras_total}`}
          tone={data.cameras_online > 0 ? "safe" : "attention"}
          subtext={data.cameras_online > 0 ? "Feed operating normally" : "No camera stream"}
        />
      </div>

      {/* 4. RECENT SAFETY ALERTS (CONFIRMED) */}
      <RecentEvents events={data.recent_events || []} />

      {/* 5. WORKERS CURRENTLY IN VIEW */}
      <section className="people-overview-section">
        <div className="section-header">
          <div>
            <h2 className="section-title">WORKERS IN MONITORED ZONE</h2>
            <span className="section-subtitle">Individual PPE breakdown for currently tracked personnel</span>
          </div>
          <span className="people-count-badge">{data.people?.length || 0} visible</span>
        </div>

        {!data.people || !data.people.length ? (
          <div className="empty-people-card">
            <div className="empty-people-icon">👥</div>
            <h3>No workers in view</h3>
            <p>
              {data.live_available
                ? "The monitored zone is currently empty."
                : "Live view isn't available yet. Check that camera monitoring is running."}
            </p>
          </div>
        ) : (
          <div className="people-operator-grid">
            {data.people.map((person) => {
              const isCompliant = person.overall_status === "COMPLIANT";
              return (
                <article
                  className={`worker-operator-card ${isCompliant ? "safe" : "attention"}`}
                  key={`${person.camera_id}-${person.person_id}`}
                >
                  <div className="worker-card-header">
                    <span className="worker-id-title">Worker #{person.person_id}</span>
                    <span className={`worker-compliance-chip ${isCompliant ? "safe" : "attention"}`}>
                      {isCompliant ? "Safe ✓" : "Attention ✗"}
                    </span>
                  </div>
                  <PPEStatus person={person} compact />
                </article>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
