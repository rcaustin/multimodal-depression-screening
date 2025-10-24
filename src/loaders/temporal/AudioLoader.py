import os
import pandas as pd
import torch
from loguru import logger


class AudioLoader:
    """
    Temporal loader for audio features (OpenSMILE) with timestamps.
    """

    def __init__(
        self,
        feature_file_template="{session_id}_OpenSMILE2.3.0_egemaps.csv",
        cache=True,
    ):
        self.feature_file_template = feature_file_template
        self.cache = cache
        self._cache_dict = {}

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
            df = pd.read_csv(csv_path, sep=";")

            # Extract timestamps first
            if "frameTime" in df.columns:
                seq_times = torch.tensor(df["frameTime"].values, dtype=torch.float32)
            else:
                logger.warning(
                    f"[AudioLoader] 'frameTime' column missing in {csv_path}"
                )
                seq_times = torch.zeros(len(df), dtype=torch.float32)

            # Keep only numeric columns as features
            numeric_df = df.select_dtypes(include=["number"]).drop(
                columns=["frameTime"], errors="ignore"
            )
            seq_feats = torch.tensor(numeric_df.values, dtype=torch.float32)

        except Exception as e:
            logger.warning(f"[AudioLoader] Failed to load {csv_path}: {e}")
            seq_feats = torch.zeros((0, 0), dtype=torch.float32)
            seq_times = torch.zeros(0, dtype=torch.float32)

        if self.cache:
            self._cache_dict[session_id] = (seq_feats, seq_times)

        return seq_feats, seq_times
