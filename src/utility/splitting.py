import os
import pandas as pd
from loguru import logger
from collections import Counter
from sklearn.model_selection import train_test_split

TRAIN_SPLIT = 0.8
RANDOM_STATE = 42

def stratified_patient_split(
    test_fraction=0.2,
    seed=42,
    data_dir="data/processed/sessions",
    metadata_path="data/processed/metadata_mapped.csv",
):
    metadata = pd.read_csv(metadata_path)

    # Keep Only Participants with Processed Session Directories
    metadata["HasSession"] = metadata["Participant_ID"].apply(
        lambda pid: os.path.isdir(os.path.join(data_dir, str(pid)))
    )
    metadata = metadata[metadata["HasSession"]]

    # Combined Stratification Label (PHQ_Binary + Gender)
    metadata["Stratum"] = (
        metadata["PHQ_Binary"].astype(str) + "_" + metadata["Gender"].astype(str)
    )

    X = metadata["Participant_ID"].astype(str)
    y = metadata["Stratum"]

    # Count distribution across strata
    stratum_counts = y.value_counts()
    min_count = stratum_counts.min()

    # Decide whether to stratify
    can_stratify = min_count >= 2

    if not can_stratify:
        logger.warning(
            f"Stratification disabled: smallest stratum has only {min_count} sample(s). "
            f"Falling back to random split without stratify."
        )
        X_train, X_test = train_test_split(
            X,
            test_size=test_fraction,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
    else:
        X_train, X_test = train_test_split(
            X,
            test_size=test_fraction,
            random_state=seed,
            shuffle=True,
            stratify=y,
        )

    train_sessions = X_train.tolist()
    test_sessions = X_test.tolist()

    # Log Stratification Summary
    logger.info("Stratification Summary (PHQ_Binary + Gender)")

    def log_distribution(df, name):
        counts = df["Stratum"].value_counts().sort_index()
        total = len(df)

        logger.info(f"--- {name} Set ({total} participants) ---")
        for stratum, count in counts.items():
            pct = (count / total) * 100 if total > 0 else 0
            logger.info(f"  {stratum}: {count} ({pct:.1f}%)")
        return counts

    train_meta = metadata[metadata["Participant_ID"].astype(str).isin(train_sessions)]
    test_meta = metadata[metadata["Participant_ID"].astype(str).isin(test_sessions)]

    counts_train = log_distribution(train_meta, "Train")
    counts_test = log_distribution(test_meta, "Test")

    # Warn if a stratum is missing in a split
    all_strata = sorted(metadata["Stratum"].unique())
    for s in all_strata:
        if s not in counts_train:
            logger.warning(f"Stratum '{s}' missing from TRAIN split.")
        if s not in counts_test:
            logger.warning(f"Stratum '{s}' missing from TEST split.")

    return train_sessions, test_sessions