"""V1 must use the existing local checkpoint. This script does not download weights."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "best.pt"


def main() -> int:
    if CHECKPOINT.is_file():
        print(f"Local checkpoint OK: {CHECKPOINT}")
        return 0
    print(
        "ERROR: models/best.pt is missing.\n"
        "Copy the existing tested checkpoint to models/best.pt.\n"
        "Do not download a different model.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
