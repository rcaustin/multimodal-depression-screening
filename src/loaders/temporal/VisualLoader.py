# src/loaders/temporal/VisualLoader.py
import os

import loguru as logger
import numpy as np
import pandas as pd
import torch


class VisualLoader:
    """
    Loader for temporal visual data (frame-level facial features per session).

    Responsibilities:
        - Load visual features for a given session directory
        - Return tensor of shape [seq_len, feature_dim]
        - Optionally implement caching to avoid repeated disk reads
    """

    def __init__(
        self,
        cache=True,
        feature_file_template=("{session_id}_OpenFace2.1.0_Pose_gaze_AUs.csv"),
    ):
        self.cache = cache
        self._cache_store = {}
        self.feature_file_template = feature_file_template

        # Action Units to Extract
        self.action_units = [
            "AU01_r",
            "AU02_r",
            "AU04_r",
            "AU05_r",
            "AU06_r",
            "AU07_r",
            "AU09_r",
            "AU10_r",
            "AU12_r",
            "AU14_r",
            "AU15_r",
            "AU17_r",
            "AU20_r",
            "AU23_r",
            "AU25_r",
            "AU26_r",
            "AU45_r",
        ]

        self.F = len(self.action_units)

    def load(self, session_dir: str) -> torch.Tensor:
        """
        Load the visual sequence for the given session.

        Args:
            session_dir (str): Path to session folder.

        Returns:
            frame_data: torch.Tensor of shape [seq_len, feature_dim]
        """
        session_id = os.path.basename(session_dir)

        # Check Cache
        if self.cache and session_id in self._cache_store:
            return self._cache_store[session_id]

        # Construct File Path
        csv_path = os.path.join(
            session_dir,
            "features",
            self.feature_file_template.format(session_id=session_id),
        )

        # Load CSV
        frame_data = self._load_csv(csv_path)

        # Handle missing/empty: return a single zero frame [1, F]
        if frame_data is None or frame_data.numel() == 0:
            logger.warning(
                f"[VisualLoader] No visual data for session {session_id}, returning zero tensor."
            )
            frame_data = torch.zeros((1, self.F), dtype=torch.float32)

        if self.cache:
            self._cache_store[session_id] = frame_data

        return frame_data

    def _load_csv(self, csv_path: str) -> torch.Tensor | None:
        """
        Load visual features from a CSV file.

        Args:
            csv_path: path to the CSV file

        Returns:
            torch
        """
        try:
            # Load DataFrame and Handle Non-Numeric Data Gracefully
            df = (
                pd.read_csv(csv_path, usecols=self.action_units)
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
            )
            arr = df.to_numpy(dtype=np.float32, copy=False)
            return torch.from_numpy(arr)
        except Exception as e:
            logger.warning(f"[VisualLoader] Failed to load CSV file {csv_path}: {e}")
            return None
