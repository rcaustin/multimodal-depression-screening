import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.loaders.AudioLoader import AudioLoader
from src.loaders.TextLoader import TextLoader
from src.loaders.VisualLoader import VisualLoader


class DepressionDataset(Dataset):
    """
    PyTorch Dataset for multimodal depression classification.

    Responsibilities:
    - Load session metadata
    - Coordinate modality loaders (text, audio, visual)
    - Return feature tensors and labels
    """

    def __init__(
        self,
        data_dir="data/processed/sessions",
        metadata_path="data/processed/metadata_mapped.csv",
        modalities=("text", "audio", "visual"),
        transform=None,
        cache=True,
    ):
        """
        Args:
            data_dir: Directory containing session subfolders
            metadata_path: Path to metadata CSV file
            modalities: Tuple of modalities to load
            transform: Optional transform applied to features
            cache: Whether to cache features in memory
        """
        self.data_dir = data_dir
        self.modalities = modalities
        self.transform = transform

        self.metadata = pd.read_csv(metadata_path)

        # List of Session IDs (Folder Names, Corresponding to Participant_ID in metadata)
        self.session_ids = [
            name for name in self.metadata["Participant_ID"].astype(str).tolist()
            if os.path.isdir(os.path.join(data_dir, name))
        ]

        # Initialize modality loaders
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

        # Load features for requested modalities
        features = {}
        for mod, loader in self.loaders.items():
            features[mod] = loader.load(session_dir)

        # Load label
        matching_rows = self.metadata.loc[self.metadata["Participant_ID"].astype(str) == session_id]
        if len(matching_rows) == 0:
            raise ValueError(f"No metadata found for session {session_id}")
        row = matching_rows.iloc[0]
        label = float(row["PHQ_Binary"])

        # Apply optional transform
        if self.transform:
            features = self.transform(features)

        # Convert features and label to tensors
        for mod in features:
            features[mod] = torch.tensor(features[mod], dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return {**features, "label": label_tensor}
