type Props = {
  label: string;
  value: number | string;
  tone?: "safe" | "attention" | "warn" | "neutral" | "ok" | "bad";
  subtext?: string;
};

export function StatusCard({ label, value, tone = "neutral", subtext }: Props) {
  const resolvedTone = tone === "bad" ? "attention" : tone === "ok" ? "safe" : tone;

  return (
    <div className={`metric-card tone-${resolvedTone}`}>
      <div className="metric-header">
        <span className="metric-label">{label}</span>
      </div>
      <div className="metric-value">{value}</div>
      {subtext && <div className="metric-subtext">{subtext}</div>}
    </div>
  );
}
