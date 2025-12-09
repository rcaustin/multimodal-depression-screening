import os
import pandas as pd
from loguru import logger
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold

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


def stratified_patient_kfold(
    n_splits=5,
    seed=42,
    data_dir="data/processed/sessions",
    metadata_path="data/processed/metadata_mapped.csv",
):
    """
    Generate K stratified folds for cross-validation.

    Args:
        n_splits: Number of folds (default: 5)
        seed: Random seed for reproducibility
        data_dir: Directory containing session data
        metadata_path: Path to metadata CSV

    Yields:
        Tuple of (fold_idx, train_sessions, val_sessions) for each fold
    """
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

    X = metadata["Participant_ID"].astype(str).values
    y = metadata["Stratum"].values

    # Count distribution across strata
    stratum_counts = metadata["Stratum"].value_counts()
    min_count = stratum_counts.min()

    # Check if smallest stratum can support n_splits
    if min_count < n_splits:
        logger.warning(
            f"Smallest stratum has only {min_count} sample(s), which is less than n_splits={n_splits}. "
            f"Reducing n_splits to {min_count} to ensure each fold has at least one sample per stratum."
        )
        n_splits = max(2, min_count)  # At least 2 folds

    logger.info(f"Generating {n_splits}-fold cross-validation splits")
    logger.info("Stratification Summary (PHQ_Binary + Gender)")

    # Create StratifiedKFold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Generate folds
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X, y)):
        train_sessions = X[train_indices].tolist()
        val_sessions = X[val_indices].tolist()

        # Create metadata subsets for logging
        train_meta = metadata[metadata["Participant_ID"].astype(str).isin(train_sessions)]
        val_meta = metadata[metadata["Participant_ID"].astype(str).isin(val_sessions)]

        # Log stratification summary for this fold
        def log_distribution(df, name):
            counts = df["Stratum"].value_counts().sort_index()
            total = len(df)

            logger.info(f"--- Fold {fold_idx} {name} Set ({total} participants) ---")
            for stratum, count in counts.items():
                pct = (count / total) * 100 if total > 0 else 0
                logger.info(f"  {stratum}: {count} ({pct:.1f}%)")
            return counts

        counts_train = log_distribution(train_meta, "Train")
        counts_val = log_distribution(val_meta, "Val")

        # Warn if a stratum is missing in a split
        all_strata = sorted(metadata["Stratum"].unique())
        for s in all_strata:
            if s not in counts_train:
                logger.warning(f"Fold {fold_idx}: Stratum '{s}' missing from TRAIN split.")
            if s not in counts_val:
                logger.warning(f"Fold {fold_idx}: Stratum '{s}' missing from VAL split.")

        yield fold_idx, train_sessions, val_sessions
