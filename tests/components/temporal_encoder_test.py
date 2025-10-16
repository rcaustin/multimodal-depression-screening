import pytest
import torch

from src.components.TemporalEncoder import TemporalEncoder


@pytest.mark.parametrize("model_type", ["lstm", "transformer"])
@pytest.mark.parametrize("pooling", ["mean", "max", "last"])
def test_output_shape(model_type, pooling):
    """Ensure the encoder returns the expected output shape."""
    batch_size = 4
    seq_len = 10
    input_dim = 16
    hidden_dim = 32

    encoder = TemporalEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        model_type=model_type,
        pooling=pooling,
    )

    input_sequence = torch.randn(batch_size, seq_len, input_dim)
    output = encoder(input_sequence)

    # Expect shape [batch_size, hidden_dim]
    assert output.shape == (batch_size, hidden_dim)
    assert not torch.isnan(output).any(), "Output contains NaNs"


def test_invalid_model_type():
    """Verify that unsupported model types raise ValueError."""
    with pytest.raises(ValueError):
        TemporalEncoder(input_dim=8, model_type="invalid_model")


def test_invalid_pooling_type():
    """Verify that unsupported pooling strategies raise ValueError."""
    encoder = TemporalEncoder(input_dim=8, model_type="lstm", pooling="invalid")
    input_sequence = torch.randn(2, 5, 8)
    with pytest.raises(ValueError):
        encoder(input_sequence)


def test_consistent_output_across_calls():
    """Check deterministic behavior with same inputs and no dropout."""
    torch.manual_seed(42)
    encoder = TemporalEncoder(input_dim=8, hidden_dim=16, dropout=0.0)
    input_sequence = torch.randn(2, 5, 8)

    first_run = encoder(input_sequence)
    second_run = encoder(input_sequence)

    assert torch.allclose(first_run, second_run, atol=1e-6), \
        "Outputs differ across identical forward passes"


@pytest.mark.parametrize("model_type", ["lstm", "transformer"])
def test_gradients_flow(model_type):
    """Ensure gradients propagate back through the model."""
    encoder = TemporalEncoder(input_dim=8, hidden_dim=16, model_type=model_type)
    input_sequence = torch.randn(2, 5, 8, requires_grad=True)

    output = encoder(input_sequence)
    loss = output.sum()
    loss.backward()

    assert input_sequence.grad is not None, "Gradients did not flow to input"
    assert not torch.isnan(input_sequence.grad).any(), "Gradient contains NaNs"
