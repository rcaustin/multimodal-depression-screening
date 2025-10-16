import pytest
import torch
from src.components.FusionModule import FusionModule


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("hidden_dim", [8, 16])
def test_fusion_output_shape(batch_size, hidden_dim):
    """Check that fusion module returns correct output shape."""
    text_dim, audio_dim, visual_dim = 10, 12, 8
    module = FusionModule(
        input_dims=[text_dim, audio_dim, visual_dim],
        hidden_dim=hidden_dim,
        dropout=0.0
    )

    # Dummy embeddings
    text_emb = torch.randn(batch_size, text_dim)
    audio_emb = torch.randn(batch_size, audio_dim)
    visual_emb = torch.randn(batch_size, visual_dim)

    fused = module(text_emb, audio_emb, visual_emb)
    assert fused.shape == (batch_size, hidden_dim)
    assert not torch.isnan(fused).any()


def test_gradients_flow():
    """Ensure gradients propagate through the fusion module."""
    module = FusionModule(input_dims=[4, 5, 6], hidden_dim=8, dropout=0.0)
    text_emb = torch.randn(3, 4, requires_grad=True)
    audio_emb = torch.randn(3, 5, requires_grad=True)
    visual_emb = torch.randn(3, 6, requires_grad=True)

    output = module(text_emb, audio_emb, visual_emb)
    loss = output.sum()
    loss.backward()

    assert text_emb.grad is not None
    assert audio_emb.grad is not None
    assert visual_emb.grad is not None
