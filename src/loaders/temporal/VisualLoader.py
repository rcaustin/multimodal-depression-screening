# src/loaders/temporal/VisualLoader.py
import os

import torch


class VisualLoader:
    """
    Loader for temporal visual data (frame-level facial features per session).

    Responsibilities:
        - Load visual features for a given session directory
        - Return tensor of shape [seq_len, feature_dim]
        - Optionally implement caching to avoid repeated disk reads
    """

    def __init__(self, cache=True):
        self.cache = cache
        self._cache_store = {}

    def load(self, session_dir):
        """
        Load the visual sequence for the given session.

        Args:
            session_dir (str): Path to session folder.

        Returns:
            torch.Tensor: [seq_len, feature_dim]
        """
        # TODO: Implement loading logic
        raise NotImplementedError
