import pytest
import torch

from src.TemporalModel import TemporalModel


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len", [1, 5, 10])
@pytest.mark.parametrize("encoder_type", ["lstm", "transformer"])
@pytest.mark.parametrize("pooling", ["mean", "max", "last"])
def test_output_shape(batch_size, seq_len, encoder_type, pooling):
    """Check that the model outputs correct shape logits."""
    text_dim, audio_dim, visual_dim = 128, 128, 128
    hidden_dim = 64

    model = TemporalModel(
        text_dim=text_dim,
        audio_dim=audio_dim,
        visual_dim=visual_dim,
        hidden_dim=hidden_dim,
        encoder_type=encoder_type,
        pooling=pooling,
        dropout=0.0,
    )

    # Create dummy inputs
    text_seq = torch.randn(batch_size, seq_len, text_dim)
    audio_seq = torch.randn(batch_size, seq_len, audio_dim)
    visual_seq = torch.randn(batch_size, seq_len, visual_dim)

    logits = model(text_seq, audio_seq, visual_seq)

    # Expect shape [batch_size, 1]
    assert logits.shape == (batch_size, 1)
    assert not torch.isnan(logits).any(), "Output contains NaNs"


@pytest.mark.parametrize("encoder_type", ["lstm", "transformer"])
def test_gradients_flow(encoder_type):
    """Ensure gradients propagate through all encoders and fusion."""
    batch_size, seq_len = 3, 5
    text_dim, audio_dim, visual_dim = 8, 8, 8
    hidden_dim = 32

    model = TemporalModel(
        text_dim=text_dim,
        audio_dim=audio_dim,
        visual_dim=visual_dim,
        hidden_dim=hidden_dim,
        encoder_type=encoder_type,
        pooling="mean",
        dropout=0.0,
    )

    text_seq = torch.randn(batch_size, seq_len, text_dim, requires_grad=True)
    audio_seq = torch.randn(batch_size, seq_len, audio_dim, requires_grad=True)
    visual_seq = torch.randn(batch_size, seq_len, visual_dim, requires_grad=True)

    output = model(text_seq, audio_seq, visual_seq)
    loss = output.sum()
    loss.backward()

    assert text_seq.grad is not None, "No gradients for text input"
    assert audio_seq.grad is not None, "No gradients for audio input"
    assert visual_seq.grad is not None, "No gradients for visual input"


def test_single_timestep():
    """Test that the model can handle sequence length = 1."""
    model = TemporalModel(
        text_dim=16,
        audio_dim=12,
        visual_dim=8,
        hidden_dim=32,
        encoder_type="lstm",
        pooling="last",
    )

    text_seq = torch.randn(2, 1, 16)
    audio_seq = torch.randn(2, 1, 12)
    visual_seq = torch.randn(2, 1, 8)

    output = model(text_seq, audio_seq, visual_seq)
    assert output.shape == (2, 1)


def test_variable_sequence_length():
    """Test that different sequence lengths produce correct output shapes."""
    # Use small dims divisible by 4 to satisfy Transformer attention head requirement
    text_dim, audio_dim, visual_dim = 8, 8, 8
    hidden_dim = 16

    model = TemporalModel(
        text_dim=text_dim,
        audio_dim=audio_dim,
        visual_dim=visual_dim,
        hidden_dim=hidden_dim,
        encoder_type="transformer",
        pooling="mean",
        dropout=0.0,
    )

    for seq_len in [1, 3, 7]:
        text_seq = torch.randn(2, seq_len, text_dim)
        audio_seq = torch.randn(2, seq_len, audio_dim)
        visual_seq = torch.randn(2, seq_len, visual_dim)
        output = model(text_seq, audio_seq, visual_seq)
        assert output.shape == (2, 1)
