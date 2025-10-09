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
        model_name: str = "all-MPNet-base-v2",
        embedding_dim: int = 384,
        cache: bool = True,
    ):
        """
        Args:
            model_name: name of the SentenceTransformer model to use
            embedding_dim: dimension of output embedding
            cache: whether to cache embeddings in memory
        """
        self.embedding_dim = embedding_dim
        self.cache = cache
        self._cache_dict = {}

        self.model = SentenceTransformer(model_name)

    def load(self, session_dir: str) -> np.ndarray:
        """
        Load and embed transcript for a single session.

        Args:
            session_dir: path to the session directory

        Returns:
            embedding: np.ndarray of shape (embedding_dim,)
        """
        session_id = os.path.basename(session_dir)

        # Check Cache
        if self.cache and session_id in self._cache_dict:
            return self._cache_dict[session_id]

        transcript_path_csv = os.path.join(session_dir, f"{session_id}_Transcript.csv")

        # Check File Existence; Return Zero Embedding if Missing
        if not os.path.exists(transcript_path_csv):
            logger.warning(f"[TextLoader] Transcript file missing for session {session_id}")
            embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            if self.cache:
                self._cache_dict[session_id] = embedding
            return embedding

        # Load Text from CSV to Single Concatenated String
        text = self._load_text(transcript_path_csv)

        # Generate Embedding for Text
        embedding = self._generate_embedding(text)

        # Cache result
        if self.cache:
            self._cache_dict[session_id] = embedding

        return embedding

    def _load_text(self, transcript_path_csv: str) -> str:
        """
        Load raw text from a transcript CSV file.

        Args:
            transcript_path_csv: path to the transcript CSV file

        Returns:
            text: concatenated text from the transcript
        """
        try:
            df = pd.read_csv(transcript_path_csv)
            text_column = next((c for c in df.columns if "text" in c.lower()), None)
            if text_column:
                return " ".join(df[text_column].astype(str).tolist())
            else:
                return ""
        except Exception as e:
            logger.warning(f"[TextLoader] Failed to read {transcript_path_csv}: {e}")
            return ""

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for given text.

        Args:
            text: input text string

        Returns:
            embedding: np.ndarray of shape (embedding_dim,)
        """
        if text.strip():
            embedding = self.model.encode(text, device="cpu")
            # If embedding is multi-dimensional (e.g., sentence-level), average
            if embedding.ndim > 1:
                embedding = np.mean(embedding, axis=0)
        else:
            # Return Zero Embedding for Empty Text
            embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        return embedding.astype(np.float32)
