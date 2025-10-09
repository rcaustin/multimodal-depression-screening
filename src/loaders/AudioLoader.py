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
            embedding: np.ndarray of shape (fixed_dim,)
        """
        session_id = os.path.basename(session_dir)

        # Check Cache
        if self.cache and session_id in self._cache_dict:
            return self._cache_dict[session_id]

        audio_path = os.path.join(
            session_dir,
            "features",
            self.feature_file_template.format(session_id=session_id)
        )

        # Check File Existence; Return Zero Embedding if Missing
        if not os.path.exists(audio_path):
            logger.warning(f"[AudioLoader] Audio feature file not found for session {session_id}")
            embedding = np.zeros(self.fixed_dim, dtype=np.float32)
            if self.cache:
                self._cache_dict[session_id] = embedding
            return embedding

        # Load CSV
        audio_data = self._load_csv(audio_path)

        # Pool over Frames and Ensure Fixed Dimension
        embedding = self._generate_embedding(audio_data)

        if self.cache:
            self._cache_dict[session_id] = embedding

        return embedding

    def _load_csv(self, path: str) -> np.ndarray:
        """
        Load CSV file and return as numpy array.

        Args:
            path: Path to the CSV file

        Returns:
            np.ndarray of shape (n_samples, n_features)
        """
        try:
            df = pd.read_csv(path)
            df = (
                df.select_dtypes(include=[np.number])
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
            )
            return df.values.astype(np.float32)
        except Exception as e:
            logger.warning(f"[AudioLoader] Failed to load CSV file {path}: {e}")
            return np.array([], dtype=np.float32)

    def _generate_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Generate fixed-dimension embedding from audio data.

        Args:
            audio_data: np.ndarray of shape (n_samples, n_features)

        Returns:
            np.ndarray of shape (fixed_dim,)
        """
        # Pool over Frames (Mean Pooling)
        if audio_data.ndim == 2 and audio_data.shape[0] > 0:
            embedding = np.mean(audio_data, axis=0)
        elif audio_data.ndim == 1:
            embedding = audio_data
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
