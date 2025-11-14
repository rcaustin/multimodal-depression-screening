#!/usr/bin/env python3
import os

import pandas as pd

# Paths
DATA_DIR = "data/processed/sessions"
METADATA_PATH = "data/processed/metadata_mapped.csv"


def main():
    # Load metadata
    metadata = pd.read_csv(METADATA_PATH)
    total_participants = len(metadata)
    total_sessions = len(metadata)
    print(f"Total participants in metadata: {total_participants}")
    print(f"Total sessions in metadata: {total_sessions}\n")

    usable_sessions = []
    omitted_sessions = []

    for _, row in metadata.iterrows():
        participant_id = str(row["Participant_ID"])
        session_dir = os.path.join(DATA_DIR, participant_id)

        if os.path.isdir(session_dir):
            usable_sessions.append(participant_id)
        else:
            omitted_sessions.append((participant_id, "Missing session folder"))

    print(f"Usable sessions: {len(usable_sessions)}\n")

    if omitted_sessions:
        print("Omitted sessions and reasons:")
        for pid, reason in omitted_sessions:
            print(f"  {pid}: {reason}")
    else:
        print("No sessions were omitted.")


if __name__ == "__main__":
    main()
