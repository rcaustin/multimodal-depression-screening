#!/bin/bash

# Paths
SRC_DIR="data/raw/EDAIC/data"
DEST_DIR="data/processed/sessions"

# Ensure Destination Directory Exists
mkdir -p "$DEST_DIR"

# Copy the Metadata CSV File
cp data/raw/EDAIC/metadata_mapped.csv data/processed/
echo "Copied metadata_mapped.csv"

# Loop through All *_P.tar.gz Files
for archive in "$SRC_DIR"/*_P.tar.gz; do
    # Extract Session ID from Filename
    filename=$(basename "$archive")
    session_id="${filename%%_P.tar.gz}"

    # Destination Path for this Session
    session_dest="$DEST_DIR/$session_id"

    # Skip if Already Extracted
    if [ -d "$session_dest" ]; then
        echo "Skipping $session_id, already extracted."
        continue
    fi

    # Create the Directory for this Session
    mkdir -p "$session_dest/features"

    echo "Extracting selected files from $filename to $session_dest ..."

    # ✅ Correct argument order
    tar -xvzf "$archive" \
        -C "$session_dest" \
        --strip-components=1 \
        "${session_id}_P/${session_id}_Transcript.csv" \
        "${session_id}_P/features/${session_id}_OpenFace2.1.0_Pose_gaze_AUs.csv" \
        "${session_id}_P/features/${session_id}_OpenSMILE2.3.0_egemaps.csv"

    # Check if Extraction Succeeded
    if [ $? -ne 0 ]; then
        echo "Error extracting $filename" >&2
        continue
    fi

    echo "Successfully extracted $filename"
done

echo "Complete."
