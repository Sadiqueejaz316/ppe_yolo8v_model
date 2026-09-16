export type PpeMark = "COMPLIANT" | "MISSING" | "PRESENT" | "UNKNOWN";

export type PersonStatus = {
  person_id: number;
  camera_id?: string;
  helmet?: PpeMark | string;
  mask?: PpeMark | string;
  vest?: PpeMark | string;
  overall_status?: string;
  missing_ppe?: string[];
  present_ppe?: string[];
};

export type ActiveViolation = {
  person_id: number;
  camera_id?: string;
  missing_ppe: string[];
  overall_status?: string;
  helmet?: string;
  mask?: string;
  vest?: string;
};

export type CameraRow = {
  id: string;
  name: string;
  location?: string;
  zone?: string;
  enabled?: boolean;
  target_fps?: number;
  status: string;
  camera_fps?: number | null;
  inference_fps?: number | null;
  has_snapshot?: boolean;
};

export type EventRow = {
  event_id: string;
  timestamp: string;
  camera_id: string;
  camera_name?: string;
  location?: string | null;
  zone?: string;
  person_id: number;
  violation_type: string;
  violation: string;
  confidence?: number;
  status: string;
  ppe?: PersonStatus | null;
  has_evidence?: boolean;
  evidence_url?: string | null;
};

export type DashboardSummary = {
  timestamp: string;
  mock: boolean;
  live_available: boolean;
  system_ok: boolean;
  message: string | null;
  total_people: number;
  compliant_people: number;
  violations: number;
  cameras_online: number;
  cameras_total: number;
  active_violations: ActiveViolation[];
  people: PersonStatus[];
  recent_events: EventRow[];
  cameras: CameraRow[];
  poll_interval_ms: number;
  snapshot_poll_interval_ms?: number;
};

export type HealthResponse = {
  ok: boolean;
  mock: boolean;
  poll_interval_ms: number;
  snapshot_poll_interval_ms?: number;
  timestamp: string;
};
