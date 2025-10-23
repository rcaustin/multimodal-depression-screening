# src/loaders/temporal/AudioLoader.py
import os

import numpy as np
import pandas as pd
import torch
from loguru import logger


class AudioLoader:
    """
    Loader for temporal audio features from OpenSMILE CSV files.

    Responsibilities:
    - Load frame-level audio features for a session
    - Enforce consistent per-frame feature dimension
    - Handle missing files gracefully
    - Cache sequences in memory
    """

    def __init__(
        self,
        feature_file_template: str = "{session_id}_BoAW_openSMILE_2.3.0_eGeMAPS.csv",
        feature_dim: int = 88,
        cache: bool = True,
    ):
        """
        Args:
            feature_file_template: CSV filename template with {session_id}
            feature_dim: Number of features per frame
            cache: Whether to cache sequences in memory
        """
        self.feature_file_template = feature_file_template
        self.feature_dim = feature_dim
        self.cache = cache
        self._cache_dict = {}

    def load(self, session_dir: str) -> torch.Tensor:
        """
        Load the temporal audio sequence for a single session.

        Returns:
            Tensor of shape [seq_len, feature_dim]
        """
        session_id = os.path.basename(session_dir)

        # Return cached sequence if available
        if self.cache and session_id in self._cache_dict:
            return self._cache_dict[session_id]

        # Construct CSV path
        csv_path = os.path.join(
            session_dir,
            "features",
            self.feature_file_template.format(session_id=session_id),
        )

        # Load CSV
        frame_data = self._load_csv(csv_path)

        # Handle missing/empty data
        if frame_data is None or frame_data.shape[0] == 0:
            logger.warning(
                f"[AudioLoader] No audio data for session {session_id}, returning zero frame."
            )
            frame_data = np.zeros((1, self.feature_dim), dtype=np.float32)

        # Ensure correct per-frame feature dimension
        frame_data = self._fix_feature_dim(frame_data)

        # Convert to torch.Tensor
        frame_tensor = torch.from_numpy(frame_data).float()

        # Cache
        if self.cache:
            self._cache_dict[session_id] = frame_tensor

        return frame_tensor

    def _load_csv(self, csv_path: str) -> np.ndarray | None:
        """Load CSV as numpy array of shape [num_frames, num_features]."""
        try:
            df = pd.read_csv(csv_path)
            # Keep only numeric columns
            df = (
                df.select_dtypes(include=[np.number])
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
            )
            return df.to_numpy(dtype=np.float32)
        except Exception as e:
            logger.warning(f"[AudioLoader] Failed to load CSV {csv_path}: {e}")
            return None

    def _fix_feature_dim(self, data: np.ndarray) -> np.ndarray:
        """
        Pad or truncate frames to match feature_dim.

        Args:
            data: [seq_len, n_features]

        Returns:
            np.ndarray: [seq_len, feature_dim]
        """
        seq_len, n_features = data.shape

        if n_features < self.feature_dim:
            pad_width = self.feature_dim - n_features
            data = np.pad(data, ((0, 0), (0, pad_width)), mode="constant")
        elif n_features > self.feature_dim:
            data = data[:, : self.feature_dim]

        return data.astype(np.float32)
