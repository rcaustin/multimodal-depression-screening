import os

import numpy as np
import pandas as pd
import scipy.io
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset


class DepressionDataset(Dataset):
    def __init__(
            self,
            data_dir="data/processed/sessions",
            metadata_path="data/processed/metadata_mapped.csv",
            modalities=('text', 'audio', 'visual'),
            transform=None
    ):
        self.data_dir = data_dir
        self.modalities = modalities
        self.transform = transform

        # Load Metadata
        self.metadata = pd.read_csv(metadata_path)
        self.session_ids = self.metadata['Participant_ID'].astype(str).tolist()

        # Initialize Text Model
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')

    def __len__(self):
        return len(self.session_ids)

    def _load_text(self, session_id):
        """Load and embed transcript text."""
        # Load transcript text
        transcript_path = os.path.join(
            self.data_dir, session_id, f"{session_id}_transcript.txt"
        )

        # Handle missing transcript files
        if not os.path.exists(transcript_path):
            return np.zeros(384, dtype=np.float32)

        # Combine all text entries into a single string
        df = pd.read_csv(transcript_path)
        text = " ".join(df.get("Text", "").astype(str).tolist())
        embedding = self.text_model.encode(text)

        # If embedding is multi-dimensional, average to get a single vector
        if embedding.ndim > 1:
            embedding = np.mean(embedding, axis=0)
        return embedding

    def _load_audio(self, session_id, fixed_dim=88):
        """Load precomputed audio features."""
        # Load audio features
        audio_path = os.path.join(
            self.data_dir,
            session_id,
            f"features/{session_id}_BoAW_openSMILE_2.3.0_eGeMAPS.csv"
        )

        # Handle missing audio files
        if not os.path.exists(audio_path):
            return np.zeros(88, dtype=np.float32)

        # Load features, ignore non-numeric columns,
        # coerce errors to NaN, fill NaN with 0
        df = pd.read_csv(audio_path)
        df = df.select_dtypes(include=[np.number])  # Keep only numeric columns
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        features = df.values.astype(np.float32)

        # Pool over frames (rows) if necessary
        if features.ndim == 2 and features.shape[0] > 0:
            features = np.mean(features, axis=0)
        elif features.ndim == 1:
            features = features
        else:
            features = np.zeros(fixed_dim, dtype=np.float32)

        # Ensure fixed dimension
        if features.shape[0] < fixed_dim:
            features = np.pad(features, (0, fixed_dim - features.shape[0]))
        elif features.shape[0] > fixed_dim:
            features = features[:fixed_dim]

        return features

    def _load_visual(self, session_id):
        """Load precomputed visual features."""
        # Load visual features
        visual_path = os.path.join(
            self.data_dir,
            session_id,
            f"features/{session_id}_CNN_VGG.mat"
        )

        # Handle missing visual files
        if not os.path.exists(visual_path):
            return np.zeros(4096, dtype=np.float32)  # Typical VGG feature size

        # Load visual features from .mat file
        mat = scipy.io.loadmat(visual_path)
        key = next((k for k in mat.keys() if not k.startswith("__")), None)
        if key is not None:
            features = np.array(mat[key]).astype(np.float32)
            # If multiple rows, average to get a single vector
            if features.ndim > 1:
                features = np.mean(features, axis=0)
            return features

        return np.zeros(4096, dtype=np.float32)  # Typical VGG feature size

    def __getitem__(self, idx):
        session_id = self.session_ids[idx]

        # Load features
        text_feat = (
            self._load_text(session_id)
            if "text" in self.modalities
            else np.array([])
        )
        audio_feat = (
            self._load_audio(session_id)
            if "audio" in self.modalities
            else np.array([])
        )
        visual_feat = (
            self._load_visual(session_id)
            if "visual" in self.modalities
            else np.array([])
        )

        # Load label (PHQ_Binary)
        row = self.metadata.loc[
            self.metadata["Participant_ID"].astype(str) == str(session_id)
        ].iloc[0]
        label = 1.0 if row["PHQ_Binary"] == 1 else 0.0

        # Apply transforms (if any)
        if self.transform:
            text_feat, audio_feat, visual_feat = self.transform(
                (text_feat, audio_feat, visual_feat)
            )

        assert text_feat.ndim == 1, f"text_feat shape: {text_feat.shape}"
        assert audio_feat.ndim == 1, f"audio_feat shape: {audio_feat.shape}"
        assert visual_feat.ndim == 1, f"visual_feat shape: {visual_feat.shape}"

        return (
            torch.tensor(text_feat, dtype=torch.float32).flatten(),
            torch.tensor(audio_feat, dtype=torch.float32).flatten(),
            torch.tensor(visual_feat, dtype=torch.float32).flatten(),
            torch.tensor(label, dtype=torch.float32),
        )
