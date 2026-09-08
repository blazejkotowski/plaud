"""The reconstruction loss must be oriented as (prediction, target).

auraloss' spectral convergence is ||target - input||_F / ||target||_F: it divides by its SECOND
argument. Feeding it (target, prediction) normalises by the prediction instead, so a prediction
that is twice too loud costs |1-2|/2 = 0.5 while one that is half as loud costs |1-0.5|/0.5 = 1.0.
That asymmetry rewards excessive output level. These tests pin the orientation.
"""
import cached_conv as cc
import pytest
import torch

from ddsp import DDSP
from ddsp.interfaces import build_control_space


@pytest.fixture(scope='module')
def model():
  cc.use_cached_conv(False)
  control_space = build_control_space([
    dict(name='loudness', dim=1, source='feature', extractor='LibrosaFeatureExtractor', params={}),
    dict(name='latents', dim=2, source='latent'),
  ])
  return DDSP(
    control_space=control_space,
    synth_configs=[{'class': 'NoiseBandSynth', 'params': {'n_filters': 16}}],
    fs=48000, n_channels=1, resampling_factor=128, capacity=8, device='cpu',
    losses=[{'type': 'MRSTFT', 'weight': 1.0, 'params': {}}], adversarial_loss=False,
  ).to('cpu')


@pytest.fixture(scope='module')
def target():
  torch.manual_seed(0)
  return torch.randn(2, 1, 16384) * 0.1


def test_perfect_reconstruction_is_zero(model, target):
  assert model._reconstruction_loss(target, target).item() == pytest.approx(0.0, abs=1e-5)


def test_too_loud_is_penalised_more_than_too_quiet(model, target):
  """The whole point of the fix: 2x is a bigger error than 0.5x, because |1-2| > |1-0.5|."""
  too_loud = model._reconstruction_loss(2.0 * target, target).item()
  too_quiet = model._reconstruction_loss(0.5 * target, target).item()
  assert too_loud > too_quiet, f"loud {too_loud:.4f} should cost more than quiet {too_quiet:.4f}"


def test_matches_the_logged_component(model, target):
  """val_loss and the logged per-component value must be the same number for a single loss."""
  torch.manual_seed(1)
  pred = target + 0.05 * torch.randn_like(target)
  total = model._reconstruction_loss(pred, target).item()
  comps = model._loss_component_values(pred, target)
  assert total == pytest.approx(comps['MultiResolutionSTFTLoss'].item(), rel=1e-6)
