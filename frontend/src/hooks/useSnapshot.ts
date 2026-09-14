import { useEffect, useRef, useState } from "react";
import { api } from "../services/api";

/**
 * Polls the live snapshot endpoint independently of the summary poll.
 * Uses image preloading to prevent UI flicker.
 * Single in-flight guard: skips a tick if the previous image is still loading.
 */
export function useSnapshot(
  cameraId: string | undefined,
  intervalMs: number = 120,
  enabled: boolean = true
): string | null {
  const [snapshotUrl, setSnapshotUrl] = useState<string | null>(null);
  const inFlightRef = useRef(false);

  useEffect(() => {
    if (!cameraId || !enabled) {
      setSnapshotUrl(null);
      inFlightRef.current = false;
      return;
    }

    let isMounted = true;
    let preloader: HTMLImageElement | null = null;

    const fetchNext = () => {
      if (!isMounted || inFlightRef.current) {
        return;
      }
      inFlightRef.current = true;
      const targetUrl = api.snapshotUrl(cameraId, Date.now());

      preloader = new Image();
      preloader.onload = () => {
        if (!isMounted) return;
        setSnapshotUrl(targetUrl);
        inFlightRef.current = false;
      };
      preloader.onerror = () => {
        if (!isMounted) return;
        inFlightRef.current = false;
      };
      preloader.src = targetUrl;
    };

    // Initial fetch immediately
    fetchNext();

    const interval = Math.max(50, intervalMs);
    const timerId = window.setInterval(fetchNext, interval);

    return () => {
      isMounted = false;
      window.clearInterval(timerId);
      if (preloader) {
        preloader.onload = null;
        preloader.onerror = null;
        preloader.src = "";
        preloader = null;
      }
      inFlightRef.current = false;
    };
  }, [cameraId, intervalMs, enabled]);

  return snapshotUrl;
}
