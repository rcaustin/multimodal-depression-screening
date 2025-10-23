import os
import re
import string
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer


class TextLoader:
    """
    Temporal Text Loader for session transcripts.

    Responsibilities:
    - Load transcript CSV for a session
    - Split transcript into sentences / utterances
    - Encode each sentence into embeddings using SentenceTransformer
    - Optionally normalize or downsample embeddings
    - Cache per-session embeddings in memory
    """

    def __init__(
        self,
        model_name: str = "all-MPNet-base-v2",
        embedding_dim: int = 768,
        cache: bool = True,
        normalize: bool = False,
        frame_hop: int = 1,
    ):
        """
        Args:
            model_name: name of the SentenceTransformer model
            embedding_dim: output embedding dimension
            cache: whether to cache session embeddings
            normalize: whether to z-normalize per feature
            frame_hop: downsample sequence by this stride
        """
        self.embedding_dim = embedding_dim
        self.cache = cache
        self._cache_dict: Dict[str, np.ndarray] = {}
        self.normalize = normalize
        self.frame_hop = max(1, int(frame_hop))
        self.model = SentenceTransformer(model_name, device="cpu")

    def load(self, session_dir: str) -> torch.Tensor:
        """
        Load and embed transcript for a single session.

        Returns:
            torch.Tensor: sequence of embeddings [seq_len, embedding_dim]
        """
        session_id = os.path.basename(session_dir)

        # Return cached if available
        if self.cache and session_id in self._cache_dict:
            return torch.from_numpy(self._cache_dict[session_id])

        transcript_path_csv = os.path.join(session_dir, f"{session_id}_Transcript.csv")
        if not os.path.exists(transcript_path_csv):
            logger.warning(f"[TextLoader] Transcript missing for session {session_id}")
            seq_embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)
            if self.cache:
                self._cache_dict[session_id] = seq_embeddings
            return torch.from_numpy(seq_embeddings)

        sentences = self._load_sentences(transcript_path_csv)
        seq_embeddings = self._embed_sentences(sentences)

        # Optional normalization
        if self.normalize and seq_embeddings.size > 0:
            mu = seq_embeddings.mean(axis=0, keepdims=True)
            sigma = seq_embeddings.std(axis=0, keepdims=True) + 1e-8
            seq_embeddings = (seq_embeddings - mu) / sigma

        # Downsample by frame_hop
        if self.frame_hop > 1 and seq_embeddings.shape[0] > 0:
            seq_embeddings = seq_embeddings[:: self.frame_hop]

        if self.cache:
            self._cache_dict[session_id] = seq_embeddings

        return torch.from_numpy(seq_embeddings.astype(np.float32))

    def _load_sentences(self, transcript_path_csv: str) -> List[str]:
        """Load text and split into non-empty sentences/utterances."""
        try:
            df = pd.read_csv(transcript_path_csv)
            text_col = next((c for c in df.columns if "text" in c.lower()), None)
            if text_col:
                lines = df[text_col].astype(str).tolist()
            else:
                lines = df.astype(str).values.ravel().tolist()
        except Exception as e:
            logger.warning(f"[TextLoader] Failed to read {transcript_path_csv}: {e}")
            return []

        # Clean and filter
        sentences = [self._clean_text(s) for s in lines if s.strip()]
        return sentences

    def _embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """Encode each sentence with SentenceTransformer."""
        if not sentences:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        # Ensure shape [num_sentences, embedding_dim]
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, self.embedding_dim)
        elif embeddings.ndim > 2:
            embeddings = embeddings.mean(axis=0)
        return embeddings.astype(np.float32)

    def _clean_text(self, text: str) -> str:
        """Lowercase, remove filler words/punctuation, collapse whitespace."""
        text = text.lower()
        text = re.sub(r"\b(um|uh|like|you know|i mean)\b", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text
