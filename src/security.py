"""Helpers that keep credentials out of logs and console output."""

from __future__ import annotations

from urllib.parse import urlparse


def redact_rtsp_url(url: str | None) -> str:
    """Return a log-safe RTSP/HTTP URL with userinfo stripped.

    Never returns the original string if it appears to contain credentials.
    """
    if not url:
        return ""
    raw = url.strip()
    if "://" not in raw:
        return "<invalid-url>"
    try:
        parsed = urlparse(raw)
    except Exception:
        return "<redacted>"

    if parsed.username or parsed.password or "@" in raw.split("://", 1)[-1].split("/", 1)[0]:
        host = parsed.hostname or "unknown-host"
        port = f":{parsed.port}" if parsed.port else ""
        path = parsed.path or ""
        scheme = parsed.scheme or "rtsp"
        return f"{scheme}://***:***@{host}{port}{path}"

    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path or ""
    scheme = parsed.scheme or "rtsp"
    return f"{scheme}://{host}{port}{path}"
