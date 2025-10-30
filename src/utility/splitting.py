import os
import random

from pandas import read_csv


def patient_level_split(
    test_fraction: float = 0.2,
    seed: int = 42,
    data_dir="data/processed/sessions",
    metadata_path="data/processed/metadata_mapped.csv",
):
    """
    Splits a sorted list of session IDs into train/test sets at the participant/session level.

    Args:
        session_ids (list): Sorted list of session IDs
        test_fraction (float): Fraction of sessions to include in the test set
        seed (int): Random seed for reproducibility

    Returns:
        train_sessions (list), test_sessions (list)
    """
    # Set Random Seed
    random.seed(seed)

    # List All Valid Session IDs (Based on Metadata and Existing Session Folders)
    metadata = read_csv(metadata_path)
    sessions = [
        str(participant_id)
        for participant_id in metadata["Participant_ID"].tolist()
        if os.path.isdir(os.path.join(data_dir, str(participant_id)))
    ]

    # Shuffle Session IDs
    random.shuffle(sessions)

    # Determine Test Size
    num_test = int(len(sessions) * test_fraction)

    # Split Into Train/Test
    test_sessions = sessions[:num_test]
    train_sessions = sessions[num_test:]

    return train_sessions, test_sessions
