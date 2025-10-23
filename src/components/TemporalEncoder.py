import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """
    Generic temporal encoder module for sequence modeling.

    This class supports two model types:
      - LSTM (default)
      - Transformer encoder

    It takes a sequence of per-timestep embeddings and produces a single
    sequence-level representation using one of several pooling strategies
    ("mean", "max", or "last").

    Args:
        input_dim (int): dimensionality of each timestep input embedding.
        hidden_dim (int): hidden size of the temporal model.
        model_type (str): either "lstm" or "transformer".
        num_layers (int): number of recurrent or transformer layers.
        dropout (float): dropout probability.
        pooling (str): pooling strategy for collapsing the time dimension.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        model_type: str = "lstm",
        num_layers: int = 1,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()

        self.model_type = model_type.lower()
        self.pooling = pooling.lower()
        self.hidden_dim = hidden_dim

        if self.model_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=False,
            )

        elif self.model_type == "transformer":
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                transformer_layer, num_layers=num_layers
            )

            # Optional projection if Transformer output dimension differs
            self.output_projection = (
                nn.Linear(input_dim, hidden_dim)
                if input_dim != hidden_dim
                else nn.Identity()
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the temporal encoder.

        Args:
            input_sequence (Tensor): input tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            Tensor: encoded sequence representation of shape [batch_size, hidden_dim].
        """

        # Ensure input has 3 dimensions
        if input_sequence.ndim == 2:
            input_sequence = input_sequence.unsqueeze(1)  # Add seq_len=1 if missing

        if self.model_type == "lstm":
            sequence_outputs, (final_hidden_state, _) = self.encoder(input_sequence)

            if self.pooling == "last":
                pooled_output = final_hidden_state[-1]
            elif self.pooling == "mean":
                pooled_output = sequence_outputs.mean(dim=1)
            elif self.pooling == "max":
                pooled_output, _ = sequence_outputs.max(dim=1)
            else:
                raise ValueError(f"Unsupported pooling type: {self.pooling}")

        elif self.model_type == "transformer":
            sequence_outputs = self.encoder(input_sequence)
            projected_outputs = self.output_projection(sequence_outputs)

            if self.pooling == "mean":
                pooled_output = projected_outputs.mean(dim=1)
            elif self.pooling == "max":
                pooled_output, _ = projected_outputs.max(dim=1)
            elif self.pooling == "last":
                pooled_output = projected_outputs[:, -1, :]
            else:
                raise ValueError(f"Unsupported pooling type: {self.pooling}")

        normalized_output = self.layer_norm(pooled_output)
        return normalized_output
