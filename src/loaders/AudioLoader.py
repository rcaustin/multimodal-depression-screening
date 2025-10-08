import os

import numpy as np
import pandas as pd
from loguru import logger


class AudioLoader:
    """
    Loader for audio-based features from precomputed OpenSMILE CSV files.

    Responsibilities:
    - Load session-level audio features
    - Handle missing files gracefully
    - Pool over time frames to create fixed-length feature vectors
    - Enforce consistent dimensionality
    - Optional caching of loaded features
    """

    def __init__(
        self,
        feature_file_template: str = (
            "{session_id}_BoAW_openSMILE_2.3.0_eGeMAPS.csv"
        ),
        fixed_dim: int = 88,
        cache: bool = True,
    ):
        """
        Args:
            feature_file_template: CSV filename template with session_id
            fixed_dim: Dimension of output feature vector
            cache: Whether to cache features in memory
        """
        self.feature_file_template = feature_file_template
        self.fixed_dim = fixed_dim
        self.cache = cache
        self._cache_dict = {}

    def load(self, session_dir: str) -> np.ndarray:
        """
        Load and process audio features for a single session.

        Args:
            session_dir: Path to the session directory

        Returns:
            features: np.ndarray of shape (fixed_dim,)
        """
        session_id = os.path.basename(session_dir)

        # Check cache
        if self.cache and session_id in self._cache_dict:
            return self._cache_dict[session_id]

        audio_path = os.path.join(
            session_dir,
            "features",
            self.feature_file_template.format(session_id=session_id),
        )

        if not os.path.exists(audio_path):
            logger.warning(
                f"[AudioLoader] Audio feature file not found for session "
                f"{session_id}"
            )
            features = np.zeros(self.fixed_dim, dtype=np.float32)
        else:
            try:
                df = pd.read_csv(audio_path)
                # Keep only numeric columns
                df = df.select_dtypes(include=[np.number]).apply(
                    pd.to_numeric, errors="coerce"
                ).fillna(0.0)
                data = df.values.astype(np.float32)

                # Pool over frames (mean pooling)
                if data.ndim == 2 and data.shape[0] > 0:
                    features = np.mean(data, axis=0)
                elif data.ndim == 1:
                    features = data
                else:
                    features = np.zeros(self.fixed_dim, dtype=np.float32)

                # Ensure fixed dimension
                if features.shape[0] < self.fixed_dim:
                    features = np.pad(
                        features,
                        (0, self.fixed_dim - features.shape[0])
                    )
                elif features.shape[0] > self.fixed_dim:
                    features = features[:self.fixed_dim]

            except Exception as e:
                logger.warning(
                    f"[AudioLoader] Failed to load audio for "
                    f"session {session_id}: {e}"
                )
                features = np.zeros(self.fixed_dim, dtype=np.float32)

        if self.cache:
            self._cache_dict[session_id] = features

        return features
