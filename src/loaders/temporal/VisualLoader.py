import os

import pandas as pd
import torch
from loguru import logger


class VisualLoader:
    """
    Temporal loader for visual features (OpenFace) with timestamps.
    """

    def __init__(
        self,
        cache=True,
        feature_file_template="{session_id}_OpenFace2.1.0_Pose_gaze_AUs.csv",
    ):
        self.cache = cache
        self._cache_dict = {}
        self.feature_file_template = feature_file_template
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

    def load(self, session_dir: str):
        session_id = os.path.basename(session_dir)
        if self.cache and session_id in self._cache_dict:
            return self._cache_dict[session_id]

        csv_path = os.path.join(
            session_dir,
            "features",
            self.feature_file_template.format(session_id=session_id),
        )
        try:
            df = pd.read_csv(csv_path, usecols=["timestamp"] + self.action_units)
            df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        except Exception as e:
            logger.warning(f"[VisualLoader] Failed to load {csv_path}: {e}")
            seq_feats = torch.zeros((0, self.F))
            seq_times = torch.zeros(0)
            return seq_feats, seq_times

        seq_feats = torch.tensor(df[self.action_units].values, dtype=torch.float32)
        seq_times = torch.tensor(df["timestamp"].values, dtype=torch.float32)

        if self.cache:
            self._cache_dict[session_id] = (seq_feats, seq_times)

        return seq_feats, seq_times
