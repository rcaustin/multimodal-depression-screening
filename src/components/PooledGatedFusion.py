import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class PooledGatedFusion(nn.Module):
    """
    Gated fusion module, performs vector-level gated fusion to learn the importance of each modality.

    Learns scalar gates per modality.

    Args:
        input_dims (List[int]): List of input dimensions for each modality.
        hidden_dim (int): Output dimension after fusion.
        dropout (float): Dropout probability.
    """

    def __init__(self, input_dims: List[int], hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        total_in = sum(input_dims)

        self.gate_network = nn.Sequential(
            nn.Linear(total_in, 32),
            nn.ReLU(),
            nn.Linear(32, len(input_dims)),
            nn.Sigmoid()
        )

        self.fusion_layer = nn.Linear(total_in, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *modality_embeddings: torch.Tensor, return_gates: bool = False):
        """
        Forward pass for gated fusion module.

        Args:
            modality_embeddings: Variable number of tensors, each of shape [batch_size, dim_i]

        Returns:
            Tensor: Fused representation of shape [batch_size, hidden_dim]
        """

        concatenated = torch.cat(modality_embeddings, dim=1)

        gated=[]

        g = self.gate_network(concatenated)
        for i, x in enumerate(modality_embeddings):
            gated.append(x * g[:, i:i+1])

        fused = torch.cat(gated, dim=1)
        h = F.relu(self.fusion_layer(fused))
        h = self.layer_norm(h)
        h = self.dropout(h)
        return (h, g) if return_gates else h # Optionally return gate values for analysis