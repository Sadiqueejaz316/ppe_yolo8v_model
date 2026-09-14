import type { CameraRow, DashboardSummary, EventRow, HealthResponse } from "../types";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(typeof detail.detail === "string" ? detail.detail : `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => getJson<HealthResponse>("/api/health"),
  summary: () => getJson<DashboardSummary>("/api/dashboard/summary"),
  cameras: () => getJson<CameraRow[]>("/api/cameras"),
  camera: (id: string) => getJson<CameraRow>(`/api/cameras/${encodeURIComponent(id)}`),
  events: (params?: { camera_id?: string; violation_type?: string }) => {
    const query = new URLSearchParams();
    if (params?.camera_id) query.set("camera_id", params.camera_id);
    if (params?.violation_type) query.set("violation_type", params.violation_type);
    const suffix = query.toString() ? `?${query}` : "";
    return getJson<EventRow[]>(`/api/events${suffix}`);
  },
  event: (id: string) => getJson<EventRow>(`/api/events/${encodeURIComponent(id)}`),
  snapshotUrl: (cameraId: string, bust?: number) =>
    `/api/cameras/${encodeURIComponent(cameraId)}/snapshot${bust ? `?t=${bust}` : ""}`,
  evidenceUrl: (eventId: string) => `/api/evidence/${encodeURIComponent(eventId)}`,
};

export function formatTime(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

export function formatDateTime(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export function missingLabel(item: string): string {
  const map: Record<string, string> = {
    helmet: "Helmet",
    hardhat: "Helmet",
    mask: "Mask",
    safety_vest: "Safety Vest",
    vest: "Safety Vest",
  };
  const clean = item.toLowerCase().replace(/^no[-_]/, "").replace(/[-_]missing$/, "");
  return map[clean] || item.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatMissingPpe(missing?: string[]): string {
  if (!missing || missing.length === 0) return "None";
  return missing.map(missingLabel).join(" + ");
}

export function getCameraDisplayName(
  camera?: { id?: string; name?: string; location?: string } | string | null,
): string {
  if (!camera) return "Camera 1";
  if (typeof camera === "string") {
    if (camera === "CAM-001") return "Plant Entrance";
    return camera;
  }
  if (camera.location && camera.location !== camera.id) return camera.location;
  if (camera.name && camera.name !== camera.id) return camera.name;
  if (camera.id === "CAM-001") return "Plant Entrance";
  return camera.id ? `Camera ${camera.id.replace(/\D+/g, "") || camera.id}` : "Camera 1";
}
