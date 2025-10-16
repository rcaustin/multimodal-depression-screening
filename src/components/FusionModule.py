import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModule(nn.Module):
    """
    Module for fusing multiple modality embeddings into a single representation.

    Uses concatenation of embeddings followed by a fully connected layer,
    activation, and optional dropout.

    Args:
        input_dims (list[int]): List of input dimensions for each modality.
        hidden_dim (int): Output dimension after fusion.
        dropout (float): Dropout probability.
    """

    def __init__(self, input_dims: list[int], hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        # Linear layer to project concatenated embeddings to hidden_dim
        total_input_dim = sum(input_dims)
        self.fusion_layer = nn.Linear(total_input_dim, hidden_dim)

        # Normalization, activation, dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *modality_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for fusion module.

        Args:
            modality_embeddings: Variable number of tensors, each of shape [batch_size, dim_i]

        Returns:
            Tensor: Fused representation of shape [batch_size, hidden_dim]
        """
        # Concatenate embeddings along the feature dimension
        concatenated = torch.cat(modality_embeddings, dim=1)

        # Linear projection + ReLU
        fused = F.relu(self.fusion_layer(concatenated))

        # Layer normalization and dropout
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        return fused
