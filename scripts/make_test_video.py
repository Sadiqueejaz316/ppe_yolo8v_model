"""Create a short looping clip from the existing test image."""

from pathlib import Path

import cv2


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "test_images" / "test.jpg"
    dest = root / "test_images" / "test.mp4"
    image = cv2.imread(str(source))
    if image is None:
        raise SystemExit(f"Could not read {source}")
    height, width = image.shape[:2]
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not open writer for {dest}")
    for _ in range(40):
        writer.write(image)
    writer.release()
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
