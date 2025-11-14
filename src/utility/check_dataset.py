#!/usr/bin/env python3
import os
import pandas as pd

# Paths
DATA_DIR = "data/processed/sessions"
METADATA_PATH = "data/processed/metadata_mapped.csv"


def main():
    # Load metadata
    metadata = pd.read_csv(METADATA_PATH)
    participant_ids = set(metadata["Participant_ID"].astype(str))
    total_participants = len(metadata)
    total_sessions = len(metadata)
    print(f"Total participants in metadata: {total_participants}")
    print(f"Total sessions in metadata: {total_sessions}\n")

    usable_sessions = []
    omitted_sessions = []

    # Check which sessions exist in metadata
    for pid in participant_ids:
        session_dir = os.path.join(DATA_DIR, pid)
        if os.path.isdir(session_dir):
            usable_sessions.append(pid)
        else:
            omitted_sessions.append((pid, "Missing session folder"))

    print(f"Usable sessions: {len(usable_sessions)}\n")

    if omitted_sessions:
        print("Omitted sessions and reasons:")
        for pid, reason in omitted_sessions:
            print(f"  {pid}: {reason}")
    else:
        print("No sessions were omitted.")

    # Check for extra sessions in DATA_DIR not in metadata
    data_dir_sessions = set(os.listdir(DATA_DIR))
    extra_sessions = data_dir_sessions - participant_ids

    if extra_sessions:
        print("\nSessions in data directory not in metadata:")
        for pid in sorted(extra_sessions):
            print(f"  {pid}")
    else:
        print("\nNo extra sessions found in data directory.")


if __name__ == "__main__":
    main()
