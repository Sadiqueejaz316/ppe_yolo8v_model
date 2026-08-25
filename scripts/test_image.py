"""Local image check. Does not download a checkpoint.

Use:

    python -m src.inference.test_model --image test_images/test.jpg
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "This script does not download weights.\n"
        "Run:\n"
        "  python -m src.inference.test_model --image test_images/test.jpg",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
