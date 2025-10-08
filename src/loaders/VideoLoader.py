import os
import numpy as np
import scipy.io
from loguru import logger


class VideoLoader:
    """
    Loader for visual features from precomputed CNN .mat files.

    Responsibilities:
    - Load session-level visual features (e.g., VGG, ResNet)
    - Handle missing or malformed files gracefully
    - Pool over frames to create a fixed-length vector
    - Optional caching of loaded features
    """

    def __init__(self, feature_file_template="{session_id}_CNN_VGG.mat",
                 feature_size=4096, cache=True):
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

        # Check cache
        if self.cache and session_id in self._cache_dict:
            return self._cache_dict[session_id]

        mat_path = os.path.join(
            session_dir, "features",
            self.feature_file_template.format(session_id=session_id)
        )

        if not os.path.exists(mat_path):
            logger.warning(
                f"[VideoLoader] Visual feature file not found for session "
                f"{session_id}"
            )
            features = np.zeros(self.feature_size, dtype=np.float32)
        else:
            try:
                mat = scipy.io.loadmat(mat_path)
                # Choose first key that doesn't start with '__'
                key = next(
                    (k for k in mat.keys() if not k.startswith("__")), None
                )
                if key:
                    data = np.array(mat[key], dtype=np.float32)
                    # Mean pooling if multiple rows
                    if data.ndim > 1:
                        features = np.mean(data, axis=0)
                    else:
                        features = data
                else:
                    logger.warning(
                        f"[VideoLoader] No valid key found in .mat for "
                        f"session {session_id}"
                    )
                    features = np.zeros(self.feature_size, dtype=np.float32)

                # Ensure fixed size
                if features.shape[0] < self.feature_size:
                    features = np.pad(features, (0, self.feature_size -
                                                 features.shape[0]))
                elif features.shape[0] > self.feature_size:
                    features = features[:self.feature_size]

            except Exception as e:
                logger.warning(
                    f"[VideoLoader] Failed to load visual features for "
                    f"session {session_id}: {e}"
                )
                features = np.zeros(self.feature_size, dtype=np.float32)

        if self.cache:
            self._cache_dict[session_id] = features

        return features
