import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.loaders.temporal.AudioLoader import AudioLoader
from src.loaders.temporal.TextLoader import TextLoader
from src.loaders.temporal.VisualLoader import VisualLoader
from src.utility.alignment import align_to_grid


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
        - Align all modalities to a common temporal grid (e.g., 30Hz)
        - Return tensors of shape [seq_len, feature_dim] per modality
        - Return label as scalar tensor
    """

    def __init__(
        self,
        data_dir="data/processed/sessions",
        metadata_path="data/processed/metadata_mapped.csv",
        modalities=("text", "audio", "visual"),
        step_hz=30.0,
        transform=None,
        cache=True,
    ):
        self.data_dir = data_dir
        self.modalities = modalities
        self.transform = transform
        self.step_hz = step_hz

        # Load metadata
        self.metadata = pd.read_csv(metadata_path)

        # List of Session IDs (folder names)
        self.session_ids = [
            str(pid)
            for pid in self.metadata["Participant_ID"].tolist()
            if os.path.isdir(os.path.join(data_dir, str(pid)))
        ]
        self.session_ids = ["300", "301"]

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

        # Load features and timestamps for each modality
        features_with_ts = {}
        for mod, loader in self.loaders.items():
            seq, ts = loader.load(session_dir)  # returns (features, timestamps)
            features_with_ts[mod] = (seq, ts)

        # Separate sequences and timestamps for alignment
        seq_list = [feat for feat, ts in features_with_ts.values()]
        ts_list = [ts for feat, ts in features_with_ts.values()]

        # Align all modalities to the common temporal grid
        aligned_features = align_to_grid(
            modality_list=seq_list, timestamp_list=ts_list, step_hz=self.step_hz
        )
        # `aligned_features` is a list in the same order as self.loaders.keys()

        features = {
            mod: aligned_features[i].detach().clone()
            for i, mod in enumerate(self.loaders.keys())
        }

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
