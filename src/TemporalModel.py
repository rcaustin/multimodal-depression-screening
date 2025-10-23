import torch
import torch.nn as nn

from src.components.TemporalEncoder import TemporalEncoder


class TemporalModel(nn.Module):
    """
    Multimodal model with timestep-wise fusion and joint temporal encoding.

    Each modality is first encoded separately without pooling, then concatenated
    per timestep. The fused sequence is fed to a joint temporal encoder for final
    classification.
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
    ) -> None:
        """
        Initialize the temporal fusion model.

        Args:
            text_dim: Dimension of per-timestep text embeddings.
            audio_dim: Dimension of per-timestep audio features.
            visual_dim: Dimension of per-timestep visual features.
            hidden_dim: Hidden dimension for temporal encoders.
            encoder_type: 'lstm' or 'transformer'.
            dropout: Dropout probability before output layer.
            pooling: Pooling strategy for the joint temporal encoder.
        """
        super().__init__()

        # Per-modality temporal encoders (no pooling)
        self.text_encoder = TemporalEncoder(
            input_dim=text_dim,
            hidden_dim=hidden_dim,
            model_type=encoder_type,
            pooling=None,
        )
        self.audio_encoder = TemporalEncoder(
            input_dim=audio_dim,
            hidden_dim=hidden_dim,
            model_type=encoder_type,
            pooling=None,
        )
        self.visual_encoder = TemporalEncoder(
            input_dim=visual_dim,
            hidden_dim=hidden_dim,
            model_type=encoder_type,
            pooling=None,
        )

        # Learnable gates per modality
        self.text_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.audio_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.visual_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
        )

        # Joint temporal encoder after gated, timestep-wise fusion
        self.joint_encoder = TemporalEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            model_type=encoder_type,
            pooling=pooling,
        )

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(
        self, text_seq: torch.Tensor, audio_seq: torch.Tensor, visual_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through timestep-aligned temporal fusion.

        Args:
            text_seq: [batch_size, seq_len, text_dim]
            audio_seq: [batch_size, seq_len, audio_dim]
            visual_seq: [batch_size, seq_len, visual_dim]

        Returns:
            Tensor: Output logits [batch_size, 1]
        """
        # Encode each modality (no pooling)
        text_emb = self.text_encoder(text_seq)  # [B, T, H]
        audio_emb = self.audio_encoder(audio_seq)  # [B, T, H]
        visual_emb = self.visual_encoder(visual_seq)  # [B, T, H]

        # Compute gates
        gT = self.text_gate(text_emb)  # [B, T, H]
        gA = self.audio_gate(audio_emb)  # [B, T, H]
        gV = self.visual_gate(visual_emb)  # [B, T, H]

        # Gated fusion per timestep
        fused_seq = gT * text_emb + gA * audio_emb + gV * visual_emb  # [B, T, H]

        # Joint temporal modeling
        fused_emb = self.joint_encoder(fused_seq)  # [B, H] after pooling

        fused_emb = self.dropout(fused_emb)
        output_logit = self.output_layer(fused_emb)  # [B, 1]

        return output_logit
