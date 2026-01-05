#!/bin/bash

SOURCE_DIR="."
DEST_DIR="configs/takealook"

# Create destination directory
mkdir -p "$DEST_DIR"

# Navigate to source so paths are relative
cd "$SOURCE_DIR" || exit

# Find files and copy them, preserving parents
# --parents flag ensures the directory tree is created in the destination
find . -type f -name "*.toml" -exec cp --parents -v {} "../$DEST_DIR" \;

echo "Done."