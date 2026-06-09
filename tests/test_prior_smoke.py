import torch
import pytest

from ddsp.prior.prior import Prior


def test_prior_forward_shapes_and_loss():
    batch_size = 2
    seq_len = 16
    num_controls = 8

    model = Prior(num_controls=num_controls, embedding_dim=16, quantization_channels=32, nhead=8, num_layers=2, dropout=0.0, max_len=64, lr=1e-3, normalization_dict=None, device='cpu')

    # random continuous latents in [-1, 1]
    x = torch.randn(batch_size, seq_len, num_controls).tanh()

    # logits shape
    logits = model(x)
    assert logits.shape == (batch_size, seq_len, num_controls, model._quantization_channels)

    # step computes loss dict
    loss = model._step(x)
    assert 'loss' in loss and 'mse' in loss and 'acc' in loss
    # ensure finite
    assert torch.isfinite(loss['loss'])
    assert torch.isfinite(loss['mse'])
    assert torch.isfinite(loss['acc'])


def test_prior_generate_length():
    num_controls = 8
    model = Prior(num_controls=num_controls, embedding_dim=16, quantization_channels=32, nhead=8, num_layers=2, dropout=0.0, max_len=32, lr=1e-3, normalization_dict=None, device='cpu')

    prime = torch.zeros(4, num_controls)
    out = model.generate(prime=prime, seq_len=20, temperature=0.5)
    assert out.shape == (20, num_controls)
