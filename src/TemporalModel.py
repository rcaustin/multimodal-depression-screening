import torch.nn as nn

from src.components.FusionModule import FusionModule
from src.components.TemporalEncoder import TemporalEncoder


class TemporalModel(nn.Module):
    """
    Multimodal model with temporal encoders for each modality and a fusion module.

    Args:
        text_dim (int): Dimension of per-timestep text embeddings.
        audio_dim (int): Dimension of per-timestep audio features.
        visual_dim (int): Dimension of per-timestep visual features.
        hidden_dim (int): Hidden dimension for temporal encoders and fusion.
        encoder_type (str): 'lstm' or 'transformer'.
        dropout (float): Dropout probability for fusion module.
        pooling (str): Pooling strategy for temporal encoder ('mean', 'max', 'last').
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 88,
        visual_dim: int = 17,
        hidden_dim: int = 128,
        encoder_type: str = "lstm",
        dropout: float = 0.3,
        pooling: str = "mean",
    ):
        super().__init__()

        # Temporal encoders for each modality
        self.text_encoder = TemporalEncoder(
            input_dim=text_dim, hidden_dim=hidden_dim, model_type=encoder_type, pooling=pooling
        )
        self.audio_encoder = TemporalEncoder(
            input_dim=audio_dim, hidden_dim=hidden_dim, model_type=encoder_type, pooling=pooling
        )
        self.visual_encoder = TemporalEncoder(
            input_dim=visual_dim, hidden_dim=hidden_dim, model_type=encoder_type, pooling=pooling
        )

        # Fusion module
        self.fusion_module = FusionModule(
            input_dims=[hidden_dim, hidden_dim, hidden_dim],
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Final output layer for binary classification
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, text_seq, audio_seq, visual_seq):
        """
        Forward pass.

        Args:
            text_seq (Tensor): [batch_size, seq_len, text_dim]
            audio_seq (Tensor): [batch_size, seq_len, audio_dim]
            visual_seq (Tensor): [batch_size, seq_len, visual_dim]

        Returns:
            Tensor: Output logits [batch_size, 1]
        """

        # Encode each modality temporally
        text_emb = self.text_encoder(text_seq)
        audio_emb = self.audio_encoder(audio_seq)
        visual_emb = self.visual_encoder(visual_seq)

        # Fuse embeddings
        fused_emb = self.fusion_module(text_emb, audio_emb, visual_emb)

        # Compute final logit
        output_logit = self.output_layer(fused_emb)

        return output_logit
