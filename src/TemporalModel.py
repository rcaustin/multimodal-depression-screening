import torch
import torch.nn as nn

from src.components.TemporalEncoder import TemporalEncoder


class TemporalModel(nn.Module):
    """
    Multimodal model with timestep-wise fusion and joint temporal encoding.

    Each modality is first encoded separately without pooling, then fused per
    timestep using learnable gates. The fused sequence is passed to a joint
    temporal encoder, which produces a pooled representation for final classification.
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 23,
        visual_dim: int = 17,
        hidden_dim: int = 128,
        encoder_type: str = "lstm",
        dropout: float = 0.3,
        pooling: str = "mean",
    ) -> None:
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

        # Joint temporal encoder after gated fusion
        self.joint_encoder = TemporalEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            model_type=encoder_type,
            pooling=pooling,
        )

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        text_seq: torch.Tensor,
        audio_seq: torch.Tensor,
        visual_seq: torch.Tensor,
        return_features: bool = False,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with aligned temporal sequences.

        Args:
            text_seq: [B, T, text_dim]
            audio_seq: [B, T, audio_dim]
            visual_seq: [B, T, visual_dim]

        Returns:
            [B, 1] logits
        """
        # Encode each modality (sequence-to-sequence)
        text_emb = self.text_encoder(text_seq, lengths=lengths)  # [B, T, H]
        audio_emb = self.audio_encoder(audio_seq, lengths=lengths)  # [B, T, H]
        visual_emb = self.visual_encoder(visual_seq, lengths=lengths)  # [B, T, H]

        # Apply learnable gates
        gT = self.text_gate(text_emb)
        gA = self.audio_gate(audio_emb)
        gV = self.visual_gate(visual_emb)

        # Gated timestep-wise fusion
        fused_seq = gT * text_emb + gA * audio_emb + gV * visual_emb  # [B, T, H]

        # Joint temporal modeling (pooled)
        fused_emb = self.joint_encoder(fused_seq)  # [B, H] after pooling

        fused_emb = self.dropout(fused_emb)
        output_logit = self.output_layer(fused_emb)  # [B, 1]

        # If requested, return feature representations as well
        if return_features:
            return output_logit, fused_emb

        return output_logit
