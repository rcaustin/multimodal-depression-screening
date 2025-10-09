import torch
import torch.nn as nn


class SimpleMultimodalModel(nn.Module):
    """
    Simple multimodal neural network for depression classification.
    Processes text, audio, and visual features, then fuses them.
    """
    def __init__(
        self,
        text_input_dim=768,     # SentenceTransformer embedding size
        audio_input_dim=88,     # OpenSMILE audio feature size
        visual_input_dim=17,    # Facial AU feature size
        hidden_dim=256,         # Hidden layer dimension
        dropout_prob=0.3        # Dropout probability
    ):
        super().__init__()
        # Encode Each Modality into a Shared Hidden Space
        self.text_encoder = nn.Linear(text_input_dim, hidden_dim)
        self.audio_encoder = nn.Linear(audio_input_dim, hidden_dim)
        self.visual_encoder = nn.Linear(visual_input_dim, hidden_dim)

        # Layer Normalization for Each Modality
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.audio_norm = nn.LayerNorm(hidden_dim)
        self.visual_norm = nn.LayerNorm(hidden_dim)

        # Fuse All Modalities into a Combined Representation
        self.fusion_layer = nn.Linear(hidden_dim * 3, hidden_dim)

        # Layer Normalization after Fusion
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # Final Output Layer (Single Logit for Binary Classification)
        self.output_layer = nn.Linear(hidden_dim, 1)

        # Dropout for Regularization
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, text_features, audio_features, visual_features):
        """
        Perform a forward pass through the multimodal network.

        Args:
            text_features (Tensor): Tensor of shape
                (batch_size, text_input_dim) containing text embeddings.
            audio_features (Tensor): Tensor of shape
                (batch_size, audio_input_dim) containing audio features.
            visual_features (Tensor): Tensor of shape
                (batch_size, visual_input_dim) containing visual features.

        Returns:
            Tensor: Output logits of shape (batch_size, 1) for
            binary classification.
        """
        # Encode Each Modality
        text_emb = torch.relu(
            self.text_norm(self.text_encoder(text_features))
        )
        audio_emb = torch.relu(
            self.audio_norm(self.audio_encoder(audio_features))
        )
        visual_emb = torch.relu(
            self.visual_norm(self.visual_encoder(visual_features))
        )

        # Concatenate Modality Embeddings
        fused_features = torch.cat([text_emb, audio_emb, visual_emb], dim=1)

        # Fuse Modalities, Apply Batch Normalization, ReLU, and Dropout
        fused_hidden = torch.relu(
            self.fusion_norm(self.fusion_layer(fused_features))
        )
        fused_hidden = self.dropout(fused_hidden)

        # Compute Output Logit
        output_logit = self.output_layer(fused_hidden)

        return output_logit
