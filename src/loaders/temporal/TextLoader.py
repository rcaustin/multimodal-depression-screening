import os
import re
import string
from typing import List, Tuple

import torch
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer


class TextLoader:
    """
    Temporal loader for session transcripts.

    Responsibilities:
    - Load transcript CSV
    - Split transcript into sentences/utterances
    - Encode each sentence into embeddings with SentenceTransformer
    - Return torch.Tensor [seq_len, feature_dim] and timestamps [seq_len]
    """

    def __init__(
        self,
        model_name: str = "all-MPNet-base-v2",
        normalize: bool = False,
        frame_hop: int = 1,
        device: str = "cpu",
        cache: bool = True,
    ):
        self.normalize = normalize
        self.frame_hop = max(1, int(frame_hop))
        self.device = device
        self.cache = cache
        self._cache_dict = {}

        # Load sentence transformer
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def load(self, session_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and embed transcript for a session.

        Returns:
            features: [seq_len, embedding_dim]
            timestamps: [seq_len]
        """
        session_id = os.path.basename(session_dir)

        # Return cached if available
        if self.cache and session_id in self._cache_dict:
            return self._cache_dict[session_id]

        transcript_path = os.path.join(session_dir, f"{session_id}_Transcript.csv")
        if not os.path.exists(transcript_path):
            logger.warning(
                f"[TemporalTextLoader] Transcript missing for session {session_id}"
            )
            seq_embeddings = torch.zeros((0, self.embedding_dim), dtype=torch.float32)
            timestamps = torch.zeros(0, dtype=torch.float32)
            if self.cache:
                self._cache_dict[session_id] = (seq_embeddings, timestamps)
            return seq_embeddings, timestamps

        # Load sentences and timestamps
        sentences, timestamps = self._load_sentences(transcript_path)
        seq_embeddings = self._embed_sentences(sentences)

        # Optional normalization
        if self.normalize and seq_embeddings.numel() > 0:
            mu = seq_embeddings.mean(dim=0, keepdim=True)
            sigma = seq_embeddings.std(dim=0, keepdim=True) + 1e-8
            seq_embeddings = (seq_embeddings - mu) / sigma

        # Downsample by frame_hop
        if self.frame_hop > 1 and seq_embeddings.size(0) > 0:
            seq_embeddings = seq_embeddings[:: self.frame_hop]
            timestamps = timestamps[:: self.frame_hop]

        if self.cache:
            self._cache_dict[session_id] = (seq_embeddings, timestamps)

        return seq_embeddings, timestamps

    def _load_sentences(self, transcript_path: str) -> Tuple[List[str], torch.Tensor]:
        """Load text and timestamps, split into non-empty utterances."""
        try:
            df = pd.read_csv(transcript_path)
            text_col = next((c for c in df.columns if "text" in c.lower()), None)
            start_col = next((c for c in df.columns if "start" in c.lower()), None)
            if text_col and start_col:
                lines = df[text_col].astype(str).tolist()
                starts = df[start_col].astype(float).tolist()
            else:
                # fallback: flatten all columns
                lines = df.astype(str).values.ravel().tolist()
                starts = list(range(len(lines)))
        except Exception as e:
            logger.warning(
                f"[TemporalTextLoader] Failed to read {transcript_path}: {e}"
            )
            return [], torch.zeros(0, dtype=torch.float32)

        # Clean and filter
        sentences = []
        timestamps = []
        for s, t in zip(lines, starts):
            s_clean = self._clean_text(s)
            if s_clean.strip():
                sentences.append(s_clean)
                timestamps.append(t)

        return sentences, torch.tensor(
            timestamps, dtype=torch.float32, device=self.device
        )

    def _embed_sentences(self, sentences: List[str]) -> torch.Tensor:
        """Convert sentences to embeddings using SentenceTransformer."""
        if not sentences:
            return torch.zeros(
                (0, self.embedding_dim), dtype=torch.float32, device=self.device
            )

        embeddings = self.model.encode(
            sentences,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False,
        )
        # Ensure shape [seq_len, embedding_dim]
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        elif embeddings.ndim > 2:
            embeddings = embeddings.mean(dim=0)

        return embeddings.float()

    def _clean_text(self, text: str) -> str:
        """Lowercase, remove filler words/punctuation, collapse whitespace."""
        text = text.lower()
        text = re.sub(r"\b(um|uh|like|you know|i mean)\b", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text
