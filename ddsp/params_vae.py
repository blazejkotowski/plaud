import torch
import numpy as np
from torch import nn

from typing import List

from ddsp.blocks import Decoder, _make_sequential, _make_mlp


class Encoder(nn.Module):
  def __init__(self, n_params: int, latent_size: int = 4, layer_sizes: List[int] = [128, 64, 32]):
    super(Encoder, self).__init__()

    self.gru = nn.GRU(n_params, layer_sizes[0], num_layers=2, batch_first=True)
    self.bottleneck = _make_sequential(layer_sizes)
    self.out = nn.Linear(layer_sizes[-1], latent_size)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x, _ = self.gru(x)
    x = self.bottleneck(x)
    z = self.out(x)
    return z


class Decoder(nn.Module):
  def __init__(self, n_params: int, latent_size: int = 4, layer_sizes: List[int] = [32, 64, 128]):
    super(Decoder, self).__init__()

    self.reverse_bottleneck = _make_sequential([latent_size] + layer_sizes)
    self.gru = nn.GRU(layer_sizes[-1], layer_sizes[-1], num_layers=2, batch_first=True)

    self.inter_mlp = _make_mlp(layer_sizes[-1], 3, layer_sizes[-1])

    self.output_params = nn.Linear(layer_sizes[-1], n_params)

  def forward(self, z: torch.Tensor) -> torch.Tensor:
    x = self.reverse_bottleneck(z)
    x, _ = self.gru(x)
    x = self.inter_mlp(x)
    out = self.output_params(x)
    return out


class ParamsVAE(nn.Module):
  def __init__(self, n_params: int, latent_size: int = 4):
    super(ParamsVAE, self).__init__()
    self.n_params = n_params
    self.latent_size = latent_size

    self.encoder = Encoder(n_params=n_params, latent_size=latent_size)
    self.decoder = Decoder(n_params=n_params, latent_size=latent_size)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the VAE.
    Arguments:
      - x: torch.Tensor, the input parameters
    Returns:
      - out: torch.Tensor, the reconstructed parameters
    """
    z = self.encoder(x)
    out = self.decoder(z)
    return out
