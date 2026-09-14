import type { PersonStatus } from "../types";

type Props = {
  person: PersonStatus | null | undefined;
  compact?: boolean;
};

export function PPEStatus({ person, compact = false }: Props) {
  if (!person) {
    return <div className="ppe-empty">PPE state not recorded for this event.</div>;
  }

  const isCompliant =
    person.overall_status === "COMPLIANT" ||
    (!person.missing_ppe || person.missing_ppe.length === 0);

  const getMark = (val?: string) => {
    if (val === "COMPLIANT" || val === "PRESENT") return { label: "✓", ok: true };
    if (val === "MISSING") return { label: "✗", ok: false };
    return { label: "—", ok: null };
  };

  const helmetMark = getMark(person.helmet);
  const vestMark = getMark(person.vest);
  const maskMark = getMark(person.mask);

  if (compact) {
    return (
      <div className="ppe-compact-row">
        <span className={`ppe-badge ${helmetMark.ok === true ? "ok" : helmetMark.ok === false ? "bad" : "neutral"}`}>
          Helmet {helmetMark.label}
        </span>
        <span className={`ppe-badge ${vestMark.ok === true ? "ok" : vestMark.ok === false ? "bad" : "neutral"}`}>
          Vest {vestMark.label}
        </span>
        {person.mask && person.mask !== "UNKNOWN" && (
          <span className={`ppe-badge ${maskMark.ok === true ? "ok" : maskMark.ok === false ? "bad" : "neutral"}`}>
            Mask {maskMark.label}
          </span>
        )}
      </div>
    );
  }

  return (
    <div className="ppe-status-card">
      <div className="ppe-status-overall">
        <span className="status-label-title">Status:</span>
        <span className={`status-badge-pill ${isCompliant ? "safe" : "attention"}`}>
          {isCompliant ? "Wearing required PPE ✓" : "Needs attention ✗"}
        </span>
      </div>

      <div className="ppe-item-grid">
        <div className={`ppe-item-box ${helmetMark.ok === true ? "ok" : helmetMark.ok === false ? "bad" : "neutral"}`}>
          <div className="ppe-item-name">Helmet</div>
          <div className="ppe-item-result">
            {helmetMark.ok === true ? "Worn ✓" : helmetMark.ok === false ? "Missing ✗" : "Not evaluated"}
          </div>
        </div>

        <div className={`ppe-item-box ${vestMark.ok === true ? "ok" : vestMark.ok === false ? "bad" : "neutral"}`}>
          <div className="ppe-item-name">Safety Vest</div>
          <div className="ppe-item-result">
            {vestMark.ok === true ? "Worn ✓" : vestMark.ok === false ? "Missing ✗" : "Not evaluated"}
          </div>
        </div>

        {person.mask && person.mask !== "UNKNOWN" && (
          <div className={`ppe-item-box ${maskMark.ok === true ? "ok" : maskMark.ok === false ? "bad" : "neutral"}`}>
            <div className="ppe-item-name">Mask</div>
            <div className="ppe-item-result">
              {maskMark.ok === true ? "Worn ✓" : maskMark.ok === false ? "Missing ✗" : "Not evaluated"}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
