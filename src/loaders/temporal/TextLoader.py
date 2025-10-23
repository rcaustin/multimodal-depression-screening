from __future__ import annotations
import os
import io
from typing import Optional, Dict
import numpy as np
import torch

try:
    # Optional dependency; only used when transcript->embeddings is requested.
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore


class TextLoader:
    """
    Temporal Text Loader

    Responsibilities implemented:
    - Single-session load via .load(session_dir) -> torch.FloatTensor [seq_len, feature_dim]
    - Support precomputed embeddings (*.npy or *.csv) OR on-the-fly transcript embedding (if model_name provided)
    - Optional per-feature z-normalization and temporal downsampling (frame_hop)
    - No padding/truncation here (leave to dataset/collate)
    - Optional in-memory caching keyed by absolute session_dir

    Typical per-session files:
      - "text_embeddings.npy" (preferred)    # shape [T, D]
      - "text_embeddings.csv"                # shape [T, D], numeric only
      - "transcript.txt" or "transcript.csv" # embedded if model_name is set
    """

    def __init__(
        self,
        embedding_glob: str = "text_embeddings.npy",
        transcript_glob: str = "transcript.txt",
        model_name: Optional[str] = None,
        cache: bool = True,
        normalize: bool = False,
        frame_hop: int = 1,
    ) -> None:
        self.embedding_glob = embedding_glob
        self.transcript_glob = transcript_glob
        self.model_name = model_name
        self.cache = cache
        self.normalize = normalize
        self.frame_hop = max(1, int(frame_hop))
        self._cache: Dict[str, torch.Tensor] = {}
        self._st_model = None

    # ---------- helpers ----------
    def _maybe_load_st_model(self):
        if self.model_name and self._st_model is None:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers is not available, cannot embed transcripts. "
                    "Install it or supply precomputed embeddings."
                )
            self._st_model = SentenceTransformer(self.model_name)

    def _glob_first(self, session_dir: str, pattern: str) -> Optional[str]:
        import glob

        matches = sorted(glob.glob(os.path.join(session_dir, pattern)))
        return matches[0] if matches else None

    def _load_array(self, path: str) -> np.ndarray:
        if path.endswith(".npy"):
            arr = np.load(path)
        elif path.endswith(".csv"):
            import pandas as pd

            arr = (
                pd.read_csv(path, header=None).select_dtypes(include=[np.number]).values
            )
        else:
            raise ValueError(f"Unsupported text feature file: {path}")
        if arr.ndim != 2:
            raise ValueError(
                f"Text features must be 2D [T, D], got shape {arr.shape} at {path}"
            )
        return arr.astype(np.float32, copy=False)

    def _embed_transcript(self, path: str) -> np.ndarray:
        self._maybe_load_st_model()
        # Read raw text, split lines -> sentences; ignore empties.
        if path.endswith(".csv"):
            import pandas as pd

            df = pd.read_csv(path)
            cand = None
            for c in df.columns:
                if c.lower() in ("text", "transcript", "utterance", "sentence"):
                    cand = c
                    break
            if cand is not None:
                lines = [str(t).strip() for t in df[cand].tolist() if str(t).strip()]
            else:
                lines = [
                    str(s).strip()
                    for s in df.astype(str).values.ravel()
                    if str(s).strip()
                ]
        else:
            with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

        if not lines:
            return np.zeros((0, 0), dtype=np.float32)

        embs = self._st_model.encode(
            lines, convert_to_numpy=True, normalize_embeddings=False
        )
        if embs.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings from SentenceTransformer, got {embs.shape}"
            )
        return embs.astype(np.float32, copy=False)

    def _postprocess(self, arr: np.ndarray) -> np.ndarray:
        # Downsample
        if self.frame_hop > 1 and arr.shape[0] > 0:
            arr = arr[:: self.frame_hop]
        # Normalize per feature
        if self.normalize and arr.size:
            mu = arr.mean(axis=0, keepdims=True)
            std = arr.std(axis=0, keepdims=True)
            arr = (arr - mu) / (std + 1e-8)
        return arr

    # ---------- public API ----------
    def load(self, session_dir: str) -> torch.Tensor:
        """
        Load a single session's text features.

        Returns:
            torch.FloatTensor with shape [seq_len, feature_dim]
        """
        session_dir = os.path.abspath(session_dir)
        if self.cache and session_dir in self._cache:
            return self._cache[session_dir]

        # Prefer precomputed embeddings
        emb_path = self._glob_first(session_dir, self.embedding_glob)
        if emb_path:
            arr = self._load_array(emb_path)
        else:
            # Fallback to transcript embedding ONLY if a model is given
            trans_path = self._glob_first(session_dir, self.transcript_glob)
            if trans_path and self.model_name:
                arr = self._embed_transcript(trans_path)
            else:
                raise FileNotFoundError(
                    f"No text feature file found in {session_dir}. "
                    f"Tried '{self.embedding_glob}' and '{self.transcript_glob}'. "
                    "If using a transcript, pass a Sentence-Transformer `model_name`."
                )

        arr = self._postprocess(arr)
        tensor = torch.tensor(arr, dtype=torch.float32)
        if self.cache:
            self._cache[session_dir] = tensor
        return tensor
