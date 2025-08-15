import torch
from kymatio.torch import Scattering1D

class ScatteringLoss(torch.nn.Module):
  def __init__(self, n_signal, j, q):
    super().__init__()
    self.scattering = Scattering1D(J=j, shape=n_signal, Q=q).to(device='cuda')

  def forward(self, x, y):
    x_sc = self.scattering(x)
    y_sc = self.scattering(y)
    return torch.nn.functional.l1_loss(x_sc, y_sc)
