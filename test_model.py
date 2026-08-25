"""This file does not download or replace models/best.pt.

Use the local checkpoint verification CLI instead:

    python -m src.inference.test_model --image test_images/test.jpg
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "Do not download a different checkpoint.\n"
        "Verify the existing local model with:\n"
        "  python -m src.inference.test_model --image test_images/test.jpg",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
