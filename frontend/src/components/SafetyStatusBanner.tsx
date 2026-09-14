type Props = {
  systemOk: boolean;
  liveAvailable: boolean;
  totalPeople: number;
  violations: number;
  mock: boolean;
  cameraName?: string;
  errorMessage?: string | null;
};

export function SafetyStatusBanner({
  systemOk,
  liveAvailable,
  totalPeople,
  violations,
  mock,
  cameraName = "Plant Entrance",
  errorMessage,
}: Props) {
  if (!systemOk) {
    return (
      <section className="safety-hero status-offline" aria-label="Safety status">
        <div className="safety-hero-badge">
          <span className="hero-dot bad" />
          SYSTEM ALERT
        </div>
        <div className="safety-hero-content">
          <h1 className="safety-hero-title">MONITORING UNAVAILABLE</h1>
          <p className="safety-hero-desc">
            {errorMessage || "Can't reach the safety monitor. Check the network connection or restart the monitoring application."}
          </p>
        </div>
        <div className="safety-hero-location">{cameraName}</div>
      </section>
    );
  }

  if (!liveAvailable) {
    return (
      <section className="safety-hero status-waiting" aria-label="Safety status">
        <div className="safety-hero-badge">
          <span className="hero-dot warn" />
          STANDBY
        </div>
        <div className="safety-hero-content">
          <h1 className="safety-hero-title">WAITING FOR CAMERA</h1>
          <p className="safety-hero-desc">
            Live view isn't available yet. Check that camera monitoring is running.
          </p>
        </div>
        <div className="safety-hero-location">{cameraName}</div>
      </section>
    );
  }

  if (violations > 0) {
    return (
      <section className="safety-hero status-attention" aria-label="Safety status">
        <div className="safety-hero-badge">
          <span className="hero-dot bad pulse" />
          ACTION REQUIRED
        </div>
        <div className="safety-hero-content">
          <h1 className="safety-hero-title">
            🔴 {violations} {violations === 1 ? "WORKER NEEDS" : "WORKERS NEED"} ATTENTION
          </h1>
          <p className="safety-hero-desc">
            Required PPE is missing. Immediate supervisor intervention recommended.
          </p>
        </div>
        <div className="safety-hero-location">
          {mock && <span className="demo-tag">DEMO MODE</span>}
          {cameraName}
        </div>
      </section>
    );
  }

  if (totalPeople === 0) {
    return (
      <section className="safety-hero status-safe" aria-label="Safety status">
        <div className="safety-hero-badge">
          <span className="hero-dot ok" />
          MONITORING
        </div>
        <div className="safety-hero-content">
          <h1 className="safety-hero-title">NO WORKERS IN VIEW</h1>
          <p className="safety-hero-desc">
            The camera is monitoring normally. No PPE activity in this zone.
          </p>
        </div>
        <div className="safety-hero-location">
          {mock && <span className="demo-tag">DEMO MODE</span>}
          {cameraName}
        </div>
      </section>
    );
  }

  return (
    <section className="safety-hero status-safe" aria-label="Safety status">
      <div className="safety-hero-badge">
        <span className="hero-dot ok" />
        COMPLIANT
      </div>
      <div className="safety-hero-content">
        <h1 className="safety-hero-title">✓ ALL WORKERS COMPLIANT</h1>
        <p className="safety-hero-desc">
          Everyone currently in view is wearing the required PPE.
        </p>
      </div>
      <div className="safety-hero-location">
        {mock && <span className="demo-tag">DEMO MODE</span>}
        {cameraName}
      </div>
    </section>
  );
}
