import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.loaders.temporal.AudioLoader import AudioLoader
from src.loaders.temporal.TextLoader import TextLoader
from src.loaders.temporal.VisualLoader import VisualLoader


class TemporalDataset(Dataset):
    """
    PyTorch Dataset for temporal multimodal depression classification.

    Each modality returns a sequence of embeddings/features:
        - Text: sequence of sentence embeddings per session
        - Audio: frame-level audio features per session
        - Visual: frame-level facial features per session

    Responsibilities:
        - Load session metadata
        - Coordinate temporal loaders for each modality
        - Return tensors of shape [seq_len, feature_dim] per modality
        - Return label as scalar tensor

    Max sequence lengths (hardcoded per modality based on 95th percentile):
        - Text: 152
        - Audio: 150104
        - Visual: 45038

    Sequences longer than these lengths are truncated; shorter sequences are zero-padded.
    """

    def __init__(
        self,
        data_dir="data/processed/sessions",
        metadata_path="data/processed/metadata_mapped.csv",
        modalities=("text", "audio", "visual"),
        transform=None,
        cache=True,
    ):
        self.data_dir = data_dir
        self.modalities = modalities
        self.transform = transform

        # Hardcoded per-modality max sequence lengths
        self.max_seq_len = {"text": 152, "audio": 150104, "visual": 45038}

        self.metadata = pd.read_csv(metadata_path)

        # List of Session IDs (Folder Names, Corresponding to Participant_ID in metadata)
        self.session_ids = [
            str(pid)
            for pid in self.metadata["Participant_ID"].tolist()
            if os.path.isdir(os.path.join(data_dir, str(pid)))
        ]

        # Initialize temporal loaders for requested modalities
        self.loaders = {}
        if "text" in modalities:
            self.loaders["text"] = TextLoader(cache=cache)
        if "audio" in modalities:
            self.loaders["audio"] = AudioLoader(cache=cache)
        if "visual" in modalities:
            self.loaders["visual"] = VisualLoader(cache=cache)

    def __len__(self):
        return len(self.session_ids)

    def __getitem__(self, idx):
        session_id = self.session_ids[idx]
        session_dir = os.path.join(self.data_dir, session_id)

        features = {}
        for mod, loader in self.loaders.items():
            seq = loader.load(session_dir)  # should return [seq_len, feature_dim]

            # Truncation/padding per modality
            if mod in self.max_seq_len:
                max_len = self.max_seq_len[mod]
                seq_len, feat_dim = seq.shape
                if seq_len > max_len:
                    seq = seq[:max_len]
                elif seq_len < max_len:
                    pad = torch.zeros((max_len - seq_len, feat_dim))
                    seq = torch.cat([seq, pad], dim=0)

            features[mod] = torch.tensor(seq, dtype=torch.float32)

        # Load label
        row = self.metadata.loc[
            self.metadata["Participant_ID"].astype(str) == session_id
        ]
        if len(row) == 0:
            raise ValueError(f"No metadata found for session {session_id}")
        label_tensor = torch.tensor(
            float(row.iloc[0]["PHQ_Binary"]), dtype=torch.float32
        )

        # Optional transform
        if self.transform:
            features = self.transform(features)

        return {**features, "label": label_tensor}
