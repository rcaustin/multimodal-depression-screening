import os

import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer


class TextLoader:
    """
    Loader for text-based features from transcripts.

    Responsibilities:
    - Load transcript CSV for a session
    - Encode text into embeddings using SentenceTransformer
    - Cache embeddings to avoid recomputation
    - Handle missing files or malformed data gracefully
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        cache: bool = True,
    ):
        """
        Args:
            model_name: Name of the SentenceTransformer model to use
            embedding_dim: Dimension of output embedding
            cache: Whether to cache embeddings in memory
        """
        self.embedding_dim = embedding_dim
        self.cache = cache
        self._cache_dict = {}

        # Load SentenceTransformer model
        self.model = SentenceTransformer(model_name)

    def load(self, session_dir: str) -> np.ndarray:
        """
        Load and embed transcript for a single session.

        Args:
            session_dir: Path to the session directory

        Returns:
            embedding: np.ndarray of shape (embedding_dim,)
        """
        session_id = os.path.basename(session_dir)

        # Check cache
        if self.cache and session_id in self._cache_dict:
            return self._cache_dict[session_id]

        transcript_path_csv = os.path.join(
            session_dir, f"{session_id}_Transcript.csv"
        )
        transcript_path_txt = os.path.join(
            session_dir, f"{session_id}_transcript.txt"
        )

        # Load text
        text = ""
        if os.path.exists(transcript_path_csv):
            try:
                df = pd.read_csv(transcript_path_csv)
                # Find column with "Text" (case-insensitive)
                text_col = next(
                    (c for c in df.columns if "Text" in c.lower()), None
                )
                if text_col:
                    text = " ".join(df[text_col].astype(str).tolist())
                else:
                    # Fallback: concatenate all columns
                    text = " ".join(df.astype(str).sum(axis=1).tolist())
            except Exception as e:
                logger.warning(
                    f"[TextLoader] Failed to read CSV transcript for "
                    f"{session_id}: {e}"
                )
        elif os.path.exists(transcript_path_txt):
            try:
                with open(transcript_path_txt, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                logger.warning(
                    f"[TextLoader] Failed to read TXT transcript for "
                    f" {session_id}: {e}"
                )
        else:
            # Missing transcript
            logger.warning(
                f"[TextLoader] Transcript not found for "
                f"session {session_id}"
            )

        # Generate embedding
        if text.strip():
            embedding = self.model.encode(text)
            # If embedding is multi-dimensional (e.g., sentence-level), average
            if embedding.ndim > 1:
                embedding = np.mean(embedding, axis=0)
        else:
            embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        embedding = embedding.astype(np.float32)

        # Cache result
        if self.cache:
            self._cache_dict[session_id] = embedding

        return embedding
