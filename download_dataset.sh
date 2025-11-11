#!/bin/bash

# Check if a URL was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <URL>"
    exit 1
fi

URL="$1"
TARGET_DIR="./data/raw/EDAIC"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Use wget to download files recursively
wget -r -np -nH --cut-dirs=1 -P "$TARGET_DIR" "$URL"

echo "Download complete. Files saved in $TARGET_DIR"
