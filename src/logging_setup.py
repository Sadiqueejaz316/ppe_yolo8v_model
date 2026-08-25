"""Process-wide logging configuration."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging. Safe to call more than once."""
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(numeric)
        for handler in root.handlers:
            handler.setLevel(numeric)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root.setLevel(numeric)
    root.addHandler(handler)

    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
