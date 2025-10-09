import os

import numpy as np
import pandas as pd
from loguru import logger


class VisualLoader:
    """
    Loader for visual features from precomputed CNN .mat files.

    Responsibilities:
    - Load session-level visual features (e.g., VGG, ResNet)
    - Handle missing or malformed files gracefully
    - Pool over frames to create a fixed-length vector
    - Optional caching of loaded features
    """

    def __init__(
            self,
            feature_file_template=(
                "{session_id}_OpenFace2.1.0_Pose_gaze_AUs.csv"
            ),
            feature_size=4096,
            cache=True
    ):
        """
        Args:
            feature_file_template: .mat filename template with session_id
            feature_size: Dimension of output feature vector
            cache: Whether to cache features in memory
        """
        self.feature_file_template = feature_file_template
        self.feature_size = feature_size
        self.cache = cache
        self._cache_dict = {}

    def load(self, session_dir: str) -> np.ndarray:
        """
        Load and process visual features for a single session.

        Args:
            session_dir: Path to the session directory

        Returns:
            features: np.ndarray of shape (feature_size,)
        """
        session_id = os.path.basename(session_dir)

        # Check Cache
        if self.cache and session_id in self._cache_dict:
            return self._cache_dict[session_id]

        # Construct File Path
        csv_path = os.path.join(
            session_dir, "features",
            self.feature_file_template.format(session_id=session_id)
        )

        if not os.path.exists(csv_path):
            logger.warning(
                f"[VisualLoader] Visual feature file not found for session "
                f"{session_id}"
            )
            features = np.zeros(self.feature_size, dtype=np.float32)
        else:
            try:
                # Load CSV and Extract Relevant Columns
                columns = [
                    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
                    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
                    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"
                ]

                # Load DataFrame and Handle Non-Numeric Data Gracefully
                df = pd.read_csv(csv_path, usecols=columns)
                df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

                # Pool Over Frames (Mean Pooling)
                features = df.values.astype(np.float32)
                features = np.mean(features, axis=0)

            except Exception as e:
                logger.warning(
                    f"[VisualLoader] Failed to load visual features for "
                    f"session {session_id}: {e}"
                )
                features = np.zeros(self.feature_size, dtype=np.float32)

        if self.cache:
            self._cache_dict[session_id] = features

        return features
