#!/bin/bash

IMAGE_DIR="test_images"

# Recursively find all supported image files
find "$IMAGE_DIR" -type f \( \
    -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o \
    -iname "*.bmp" -o -iname "*.webp" \
\) | while read img; do
    echo "Processing: $img"
    python -m src.main --mode image --source "$img" --no-display --save-output "outputs/$(basename "$img")"
done

