import os

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset

from src.loaders.temporal.AudioLoader import AudioLoader
from src.loaders.temporal.TextLoader import TextLoader
from src.loaders.temporal.VisualLoader import VisualLoader
from src.utility.alignment import align_to_grid


class TemporalDataset(Dataset):
    """
    PyTorch Dataset for temporal multimodal depression classification with caching.

    Each modality returns a sequence of embeddings/features:
        - Text: sequence of sentence embeddings per session
        - Audio: frame-level audio features per session
        - Visual: frame-level facial features per session

    Responsibilities:
        - Load session metadata
        - Coordinate temporal loaders for each modality
        - Align all modalities to a common temporal grid (e.g., 30Hz)
        - Cache aligned tensors per session for faster reuse
        - Return tensors of shape [seq_len, feature_dim] per modality
        - Return label as scalar tensor
    """

    def __init__(
        self,
        sessions,
        data_dir="data/processed/sessions",
        metadata_path="data/processed/metadata_mapped.csv",
        modalities=("text", "audio", "visual"),
        step_hz=30.0,
        transform=None,
        cache=True,
        chunk_len = None,
        chunk_hop = None
    ):
        self.session_ids = sessions
        self.data_dir = data_dir
        self.modalities = modalities
        self.transform = transform
        self.step_hz = step_hz
        self.cache = cache
        self.metadata = pd.read_csv(metadata_path)

        # Initialize Temporal Loaders
        self.loaders = {}
        if "text" in modalities:
            self.loaders["text"] = TextLoader(cache=cache)
        if "audio" in modalities:
            self.loaders["audio"] = AudioLoader(cache=cache)
        if "visual" in modalities:
            self.loaders["visual"] = VisualLoader(cache=cache)

        # Chunking configuration
        self.chunk_len = chunk_len
        if self.chunk_len is not None:
            self.chunk_hop = chunk_hop or chunk_len
        else:
            self.chunk_hop = None

        # Build index over sessions. chunks: list of (session_id, start)
        self.index = self._build_index()

    # Internal helpers for chunking
    def _load_aligned_features(self, session_id):
        """
        Load aligned features for a single session, using caching when available.

        Returns:
            dict: {modality_name: tensor [T, D]}
        """
        session_dir = os.path.join(self.data_dir, session_id)
        cache_dir = os.path.join(session_dir, "aligned_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Try loading cached aligned tensors
        cache_paths = {
            mod: os.path.join(cache_dir, f"{mod}.pt") for mod in self.loaders.keys()
        }
        cache_exists = all(os.path.exists(path) for path in cache_paths.values())

        if self.cache and cache_exists:
            features = {
                mod: torch.load(path) for mod, path in cache_paths.items()
            }
        else:
            # Load raw features and timestamps
            features_with_ts = {}
            for mod, loader in self.loaders.items():
                seq, ts = loader.load(session_dir)
                features_with_ts[mod] = (seq, ts)

            seq_list = [feat for feat, ts in features_with_ts.values()]
            ts_list = [ts for feat, ts in features_with_ts.values()]

            # Align to common temporal grid
            aligned_features = align_to_grid(
                modality_list=seq_list, timestamp_list=ts_list, step_hz=self.step_hz
            )

            # Store in dict keyed by modality
            features = {
                mod: aligned_features[i].detach().clone()
                for i, mod in enumerate(self.loaders.keys())
            }

            # Save aligned tensors for reuse
            logger.info(f"Caching aligned features for session {session_id}")
            if self.cache:
                for mod, seq in features.items():
                    torch.save(seq, cache_paths[mod])
        
        return features


    def _build_index(self):
        """
        Build an index of sessions (session_id, start) pairs.
        
        In full session mode (chunk_len = None), there is exactly one entry per session with start = None.
        In chunked mode, sliding windows of length chunk_len and hop chunk_hop are created.
        """

        index = []

        if self.chunk_len is None:
            # One item per session, original behaviour
            for sid in self.session_ids:
                index.append((sid, None))
            return index
        
        # Chunked mode: build chunks for each session
        for sid in self.session_ids:
            features = self._load_aligned_features(sid)
            # Use first modality's length as T (all modalities aligned)
            first_mod = next(iter(features.keys()))
            T = features[first_mod].shape[0]

            start = 0
            while start + self.chunk_len <= T:
                index.append((sid, start))
                start += self.chunk_hop # Final tail chunk is dropped to eliminate padding requirement

        return index
    
    # Dataset interface
    def __len__(self):
        return len(self.index) # In full session mode, len == number of sessions

    def __getitem__(self, idx):
        # Resolve session / chunk
        session_id, start = self.index[idx]

        # Load aligned features for session
        features = self._load_aligned_features(session_id)

        # In in chunking mode, slice the features to [chunk_len, D]
        if self.chunk_len is not None and start is not None:
            end = start + self.chunk_len
            features = {
                mod: seq[start:end]
                for mod, seq in features.items()
            }

        # Optional transform
        if self.transform:
            features = self.transform(features)

        # Load label + gender from metadata using session_id
        row = self.metadata.loc[
            self.metadata["Participant_ID"].astype(str) == session_id
        ]
        if len(row) == 0:
            raise ValueError(f"No metadata found for session {session_id}")
        
        label_tensor = torch.tensor(row.iloc[0]["PHQ_Binary"], dtype=torch.float32)

        g_raw = str(row.iloc[0]["Gender"]).strip().lower() # expects 'male' or 'female'
        if g_raw == "male":
            gender = 0.0
        elif g_raw == "female":
            gender = 1.0
        else:
            logger.warning(f"Unknown gender '{g_raw}' for session {session_id}, defaulting to 0.0")
            gender = 0.0

        gender_tensor = torch.tensor(gender, dtype=torch.float32)

        sample = {
            **features,
            "label": label_tensor,
            "gender": gender_tensor,
            "session": session_id,
        }

        # For testing aggregation, also return start index
        if self.chunk_len is not None:
            sample["start"] = start

        return sample        
