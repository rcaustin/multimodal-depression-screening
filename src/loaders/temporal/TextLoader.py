# src/loaders/temporal/TextLoader.py
import os

import torch


class TextLoader:
    """
    Loader for temporal text data (sentence embeddings per session).

    Responsibilities:
        - Load text embeddings for a given session directory
        - Return tensor of shape [seq_len, feature_dim]
        - Optionally implement caching to avoid repeated disk reads
    """

    def __init__(self, cache=True):
        self.cache = cache
        self._cache_store = {}

    def load(self, session_dir):
        """
        Load the text sequence for the given session.

        Args:
            session_dir (str): Path to session folder.

        Returns:
            torch.Tensor: [seq_len, feature_dim]
        """
        # TODO: Implement loading logic
        raise NotImplementedError
