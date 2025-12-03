from typing import Optional
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TemporalEncoder(nn.Module):
    """
    Generic temporal encoder for sequence modeling.

    Supports:
      - LSTM
      - Transformer

    Returns either a pooled sequence representation or the full sequence
    when pooling=None (used for timestep-wise fusion).

    Args:
        input_dim (int): Dimensionality of each timestep input embedding.
        hidden_dim (int): Hidden size of the temporal model.
        model_type (str): "lstm" or "transformer".
        num_layers (int): Number of recurrent or transformer layers.
        dropout (float): Dropout probability.
        pooling (str or None): "mean", "max", "last", or None for no pooling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        model_type: str = "lstm",
        num_layers: int = 1,
        dropout: float = 0.1,
        pooling: Optional[str] = "mean",
    ) -> None:
        super().__init__()

        self.model_type = model_type.lower()
        self.pooling = pooling.lower() if pooling is not None else None
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
            # Project to hidden_dim if needed
            self.output_projection = (
                nn.Linear(input_dim, hidden_dim)
                if input_dim != hidden_dim
                else nn.Identity()
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_sequence: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass for the temporal encoder.

        Args:
            input_sequence (Tensor): [batch_size, seq_len, input_dim]

        Returns:
            Tensor: either pooled [B, H] or full sequence [B, T, H] if pooling=None
        """
        # Ensure input has 3 dimensions
        if input_sequence.ndim == 2:
            input_sequence = input_sequence.unsqueeze(1)  # Add seq_len=1 if missing

        B, T, _ = input_sequence.shape

        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=input_sequence.device)

        if self.model_type == "lstm":
            self.encoder.flatten_parameters()

            sequence_outputs, (final_hidden_state, _) = self.encoder(input_sequence)
        
        elif self.model_type == "transformer":
            sequence_outputs = self.encoder(input_sequence)
            sequence_outputs = self.output_projection(sequence_outputs)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        if self.pooling is None:
            # Return full sequence for timestep-wise fusion
            return self.layer_norm(sequence_outputs)

        # Pooling options
        mask_valid = ( # True for real data, False for padding
            torch.arange(T, device=sequence_outputs.device)[None, :].expand(B, T) < lengths[:, None]
        )
        mask_valid_f = mask_valid.unsqueeze(-1).float()

        if self.pooling == "last":
            if self.model_type == "lstm":
                pooled_output = final_hidden_state[-1]  # [B, H]
            else:
                idx = (lengths - 1).clamp(min=0)  # [B]
                pooled_output = sequence_outputs[torch.arange(B), idx, :]  # [B, H]
        
        elif self.pooling == "mean":
            summed = (sequence_outputs * mask_valid_f).sum(dim=1)  # [B, H]
            denom = lengths.clamp(min=1).unsqueeze(-1)
            pooled_output = summed / denom
        
        elif self.pooling == "max":
            huge_neg = torch.finfo(sequence_outputs.dtype).min
            masked_seq = sequence_outputs.clone()
            masked_seq[~mask_valid] = huge_neg
            pooled_output, _ = masked_seq.max(dim=1)

        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

        return self.layer_norm(pooled_output)
