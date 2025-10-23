# src/loaders/temporal/AudioLoader.py
import os
import torch
import numpy as np
import pandas as pd
from loguru import logger


class AudioLoader:
    """
    Loader for temporal audio data (frame-level features per session).

    Responsibilities:
        - Load audio features for a given session directory
        - Return tensor of shape [seq_len, feature_dim]
        - Optionally implement caching to avoid repeated disk reads
        - Optionally normalize features
    """

    def __init__(
        self,
        feature_file_template: str = "{session_id}_BoAW_openSMILE_2.3.0_eGeMAPS.csv",
        cache: bool = True,
        normalize: bool = True,
    ):
        self.feature_file_template = feature_file_template
        self.cache = cache
        self._cache_store = {}
        self.normalize = normalize

    def load(self, session_dir: str) -> torch.Tensor:
        """
        Load the audio sequence for the given session.

        Args:
            session_dir (str): Path to session folder.

        Returns:
            torch.Tensor: [seq_len, feature_dim]
        """
        # derive session_id from folder name; remove trailing "_P" if present
        dirname = os.path.basename(session_dir)
        session_id = dirname[:-2] if dirname.endswith("_P") else dirname

        # Check cache
        if self.cache and session_id in self._cache_store:
            return self._cache_store[session_id]

        # Construct file path
        audio_path = os.path.join(
            session_dir,
            "features",
            self.feature_file_template.format(session_id=session_id),
        )

        # Load CSV
        frame_features = self._load_csv(audio_path)

        # Normalize features (optional)
        if self.normalize:
            frame_features = self._normalize(frame_features)

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(frame_features).float()

        logger.info(
            f"[TemporalAudioLoader] Returned tensor shape for {session_id}: {tuple(audio_tensor.shape)}"
        )  # temporary

        if self.cache:
            self._cache_store[session_id] = audio_tensor

        return audio_tensor

    def _load_csv(self, csv_path: str) -> np.ndarray:
        """
        Load CSV file and return as numpy array.

        Args:
            csv_path: path to the CSV file

        Returns:
            np.ndarray of shape (n_samples, n_features)
        """
        try:
            df = pd.read_csv(csv_path)

            # Drop the first two columns: unknown + timestamp
            df = df.iloc[:, 2:]  # keep only feature columns

            # Keep numeric columns and clean them
            df = (
                df.select_dtypes(include=[np.number])
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
            )

            return df.values.astype(np.float32)

        except Exception as e:
            logger.warning(f"[AudioLoader] Failed to load CSV file {csv_path}: {e}")
            return np.zeros((0, 0), dtype=np.float32)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Z-score normalize each feature column (frame-level).

        Args:
            data: np.ndarray of shape [seq_len, feature_dim]

        Returns:
            np.ndarray: normalized data [seq_len, feature_dim]
        """
        if data.size == 0:
            return data

        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        std[std == 0] = 1.0  # avoid division by zero
        normalized = (data - mean) / std
        return normalized
