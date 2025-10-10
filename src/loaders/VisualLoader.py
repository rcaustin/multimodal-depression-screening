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
            fixed_dim=4096,
            cache=True
    ):
        """
        Args:
            feature_file_template: .mat filename template with session_id
            feature_size: Dimension of output feature vector
            cache: Whether to cache features in memory
        """
        self.feature_file_template = feature_file_template
        self.fixed_dim = fixed_dim
        self.cache = cache
        self._cache_dict = {}

        # Action Units to Extract
        self.action_units = [
            "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
            "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
            "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"
        ]

    def load(self, session_dir: str) -> np.ndarray:
        """
        Load and process visual features for a single session.

        Args:
            session_dir: path to the session directory

        Returns:
            features: np.ndarray of shape (fixed_dim,)
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

        # Load CSV
        visual_data = self._load_csv(csv_path)

        # Pool over Frames and Ensure Fixed Dimension
        embeddings = self._generate_embedding(visual_data)

        if self.cache:
            self._cache_dict[session_id] = embeddings

        return embeddings

    def _load_csv(self, csv_path: str) -> np.ndarray:
        """
        Load visual features from a CSV file.

        Args:
            csv_path: path to the CSV file

        Returns:
            np.ndarray of shape (n_samples, n_features)
        """
        try:
            # Load DataFrame and Handle Non-Numeric Data Gracefully
            df = (
                pd.read_csv(csv_path, usecols=self.action_units)
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0.0)
            )
            return df.values.astype(np.float32)

        except Exception as e:
            logger.warning(f"[VisualLoader] Failed to load CSV file {csv_path}: {e}")
            return np.array([], dtype=np.float32)

    def _generate_embedding(self, visual_data: np.ndarray) -> np.ndarray:
        """
        Generate fixed-dimension embedding from visual data.

        Args:
            visual_data: input visual data array

        Returns:
            embedding: np.ndarray of shape (fixed_dim,)
        """
        if visual_data.ndim == 2:
            embedding = np.mean(visual_data, axis=0)
        elif visual_data.ndim == 1:
            embedding = visual_data
        else:
            embedding = np.zeros(self.fixed_dim, dtype=np.float32)

        # Ensure Fixed Dimension
        if embedding.shape[0] < self.fixed_dim:
            # Pad to Fixed Dimension
            embedding = np.pad(embedding, (0, self.fixed_dim - embedding.shape[0]))
        elif embedding.shape[0] > self.fixed_dim:
            # Truncate to Fixed Dimension
            embedding = embedding[:self.fixed_dim]

        return embedding
