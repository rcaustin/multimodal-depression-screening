"""
compute_max_seq_lengths.py

Utility script to compute recommended maximum sequence lengths for temporal
modeling. Scans raw data files per session for text, audio, and visual modalities,
and prints the 95th percentile sequence lengths for each. These values can be
hard-coded into the TemporalDataset class.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = "data/processed/sessions"
METADATA_PATH = "data/processed/metadata_mapped.csv"
PERCENTILE = 95
MODALITIES = ("text", "audio", "visual")


def count_text_rows(transcript_path):
    """Return the number of rows in the transcript CSV (excluding header)."""
    if not os.path.isfile(transcript_path):
        return 0
    df = pd.read_csv(transcript_path)
    return len(df)


def count_visual_rows(visual_path):
    """Return the number of rows in the OpenFace visual features CSV."""
    if not os.path.isfile(visual_path):
        return 0
    df = pd.read_csv(visual_path)
    return len(df)


def count_audio_rows(audio_path):
    """Return the number of rows in the OpenSMILE audio features CSV."""
    if not os.path.isfile(audio_path):
        return 0
    df = pd.read_csv(audio_path, sep=";")
    return len(df)


def main():
    metadata = pd.read_csv(METADATA_PATH)
    session_ids = [
        str(pid) for pid in metadata["Participant_ID"].tolist()
        if os.path.isdir(os.path.join(DATA_DIR, str(pid)))
    ]

    seq_lengths = {mod: [] for mod in MODALITIES}

    for session_id in session_ids:
        session_dir = os.path.join(DATA_DIR, session_id)
        transcript_path = os.path.join(session_dir, f"{session_id}_Transcript.csv")
        visual_path = os.path.join(
            session_dir, "features", f"{session_id}_OpenFace2.1.0_Pose_gaze_AUs.csv"
        )
        audio_path = os.path.join(
            session_dir, "features", f"{session_id}_OpenSMILE2.3.0_egemaps.csv"
        )

        seq_lengths["text"].append(count_text_rows(transcript_path))
        seq_lengths["visual"].append(count_visual_rows(visual_path))
        seq_lengths["audio"].append(count_audio_rows(audio_path))

    max_seq_len = {
        mod: int(np.percentile(lengths, PERCENTILE))
        for mod, lengths in seq_lengths.items()
    }

    print("Recommended max sequence lengths (95th percentile):")
    for mod, length in max_seq_len.items():
        print(f"{mod}: {length}")


if __name__ == "__main__":
    main()
