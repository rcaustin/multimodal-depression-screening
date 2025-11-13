# src/components/DomainAdversary.py

import torch.nn as nn

"""
Domain adversary head for DANN

Takes a shared feature representation (e.g., from TemporalModel)
and predicts a binary domain label  (gender encoded as 0/1).

Gradient reversal layer forces the shared feature extractor to remove
gender-specific information from the learned representation.
"""

class DANN(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128):
        """
        Inputs:
            input_dim: dimension of the input feature representation
            hidden_dim: dimension of the hidden layer
        """
        super().__init__()
        self.adversary = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        """
        Forward pass of the domain adversary.
        
        Inputs:
            features: input feature tensor of shape [B, input_dim]

        Outputs:
            [B, 1] tensor of domain logits
        """
        return self.adversary(features)