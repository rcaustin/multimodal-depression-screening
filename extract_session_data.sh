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

    # Check if Session Already Exists
    if [ -d "$session_dest" ]; then
        echo "Skipping $session_id, already extracted."
        continue
    fi

    # Create the Directory for this Session
    mkdir -p "$session_dest"

    echo "Extracting $filename to $session_dest ..."

    # Extract the Archive
    tar -xvf "$archive" --strip-components=1 -C "$session_dest"

    # Check if Extraction Succeeded
    if [ $? -ne 0 ]; then
        echo "Error extracting $filename" >&2
        continue
    fi

    echo "Successfully extracted $filename"

    # Keep only the three files we need
    transcript_file="$session_dest/${session_id}_Transcript.csv"
    features_dir="$session_dest/features"

    # Delete everything except the three desired files
    find "$session_dest" -mindepth 1 ! -path "$transcript_file" ! -path "$features_dir/${session_id}_OpenFace2.1.0_Pose_gaze_AUs.csv" ! -path "$features_dir/${session_id}_OpenSMILE2.3.0_egemaps.csv" -exec rm -rf {} +

    echo "Cleaned up unnecessary files for session $session_id"
done

echo "Complete."
