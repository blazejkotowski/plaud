import torch
import math
import scipy.signal
import torch.nn.functional as func

from typing import List

class SlicedWassersteinLoss(torch.nn.Module):
  """
  Sliced Wasserstein Loss for comparing  two audio signals.
  This loss computes the Sliced Wasserstein solution to the Optimal Tranposrt problem,
  as described in the following paper paper.
  Fabiani, L., Schlecht, S. J., & Elvander, F. (2024).
  Time-Frequency Audio Similarity Using Optimal Transport.
  2024 58th Asilomar Conference on Signals, Systems, and Computers, 1414–1417.
  https://doi.org/10.1109/IEEECONF60004.2024.10943074

  Args:
    - window: torch.Tensor, the window to use for the STFT
    - n_projections: int, the number of projections to use for the sliced Wasserstein distance
    - sampling_rate: int, the sampling rate for the STFT
    - p: int, the p in the p-Wasserstein distance
    - magnitude: str, the type of magnitude to use ('lin' for linear, 'log' for logarithmic)
    - device: str, the device to use for the computation (e.g., 'cuda' or 'cpu')
  """

  eps = 1e-8
  # k = 1e1 # regulatization constant from the paper

  def __init__(self, window: torch.Tensor, n_projections: int = 10, sampling_rate: int = 44100, p: int = 2, magnitude: str = 'lin', device: str = 'cuda'):
    super(SlicedWassersteinLoss, self).__init__()

    self.stft_window = window
    self.win_size = self.stft_window.shape[0]
    self.hop_size = self.win_size // 2
    self.n_projections = n_projections
    self.sampling_rate = sampling_rate
    self.p = p
    self.device = device
    self.magnitude = magnitude

    self._projections_precomputed = False
    self._projections = None

    self.proj_directions = torch.nn.Parameter(
      torch.randn(self.n_projections, 2)  # [proj_direction_i: [cos fi_i, sin fi_i]]
    )
    self.proj_directions.data = func.normalize(self.proj_directions.data, dim=1)

    # self.sinkloss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)


  def stft(self, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Short-Time Fourier Transform (STFT) of the input signal.

    Args:
      - x: torch.Tensor, the input audio signal

    Returns:
      - X: torch.Tensor, the magnitude spectrum of the STFT
    """
    # window = torch.hann_window(self.win_size)
    x_stft = torch.stft(x, n_fft=self.win_size, hop_length=self.hop_size, window=self.stft_window, return_complex=True)
    x_mag = torch.sqrt(torch.clamp(x_stft.real**2 + x_stft.imag**2, min=self.eps))
    if self.magnitude == 'pow':
      x_mag = x_mag * x_mag
    if self.magnitude == 'log':
      x_mag = torch.log1p(x_mag)
    return x_mag

  def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Sliced Wasserstein loss between two audio signals.

    Args:
      - x: torch.Tensor[B, N], the first audio signal
      - y: torch.Tensor[B, N], the second audio signal

    Returns:
      - loss: torch.Tensor, the computed Sliced Wasserstein loss
    """
    X_mag = self.stft(x).permute(0, 2, 1)  # shape: [B, T, F]
    Y_mag = self.stft(y).permute(0, 2, 1)  # shape: [B, T, F]

    B, T, F = X_mag.shape

    # energy_loss = self._compute_energy_loss(X_mag, Y_mag)

    X_mag = self._convert_to_prob(X_mag)  # shape: [B, T, F]
    Y_mag = self._convert_to_prob(Y_mag)  # shape: [B, T, F]

    # return self.sinkloss(X_mag, Y_mag)

    if not self._projections_precomputed:
      self._projections = self._compute_sorted_projections(T, F, self.n_projections)
      self._projections_precomputed = True

    projections = self._projections

    X_flat = X_mag.flatten(1) # shape: [B, T*F]
    Y_flat = Y_mag.flatten(1) # shape: [B, T*F]

    # Option 1: POT
    # diffs = []
    # for projection in self._projections:
    #   # print(X_flat.shape, Y_flat.shape, projection.shape)
    #   diff = wasserstein_1d(projection, projection, X_flat.squeeze(), Y_flat.squeeze(), p=2, require_sort=True)
    #   diffs.append(diff)
    # return torch.stack(diffs).mean()

    # Option 2: manual cdf looping
    # diffs = []
    # for projection in self._projections:
      # X_mag_sorted = X_flat[:, projection]  # shape: [B, T*F]
      # Y_mag_sorted = Y_flat[:, projection]  # shape: [B, T*F]

    #   # cumulative density functions
    #   x_cdf = torch.cumsum(X_mag_sorted, dim=1)
    #   y_cdf = torch.cumsum(Y_mag_sorted, dim=1)

    #   diff = torch.pow(torch.abs(x_cdf - y_cdf), self.p).sum() / (F) # normalize by the number of bins
    #   diffs.append(diff)
    # loss = torch.stack(diffs).mean() / math.sqrt(self.n_projections)  # normalize by the number of projections

    # Option: Learnable projections
    # x_bins = torch.linspace(0, 1, T, device=self.device)
    # y_bins = torch.linspace(0, 1, F, device=self.device)
    # grid = torch.stack(torch.meshgrid(x_bins, y_bins, indexing='ij'), dim=-1).reshape(-1, 2)  # shape: [T*F, 2]
    # projected = torch.matmul(self.proj_directions, grid.T)  # [n_proj, T*F]
    # sorted_idx = torch.argsort(projected, dim=1)  # [n_proj, T*F]
    # projections = sorted_idx


    # Option 3: Gather instead of a loop for speed
    X_projected = torch.gather(X_flat.unsqueeze(1).expand(-1, self.n_projections, -1), 2,
                          projections.unsqueeze(0).expand(B, -1, -1)) # [B, n_projections , T*f]
    Y_projected = torch.gather(Y_flat.unsqueeze(1).expand(-1, self.n_projections, -1), 2,
                          projections.unsqueeze(0).expand(B, -1, -1)) # [B, n_projections , T*f]

    x_cdf = torch.cumsum(X_projected, dim=2)  # shape: [B, n_projections, T*F]
    y_cdf = torch.cumsum(Y_projected, dim=2)  # shape: [B, n_projections, T*F]

    if self.p == 1:
      # L1 Wasserstein distance
      loss = torch.abs(x_cdf - y_cdf).sum(dim=2) / F
    elif self.p == 2:
      # L2 Wasserstein distance
      diff = (x_cdf - y_cdf)
      loss = (diff*diff).sum(dim=2) / F
    else:
      # General p-Wasserstein distance
      loss = torch.pow(torch.abs(x_cdf - y_cdf), self.p).sum(dim=2) / F

    loss = loss.mean()  # shape: [1]

    # loss += 1e-6 * energy_loss  # add energy loss to the final loss
    return loss


  def _convert_to_prob(self, X: torch.Tensor) -> torch.Tensor:
    """
    Converts the input tensor to a probability distribution by normalizing it.

    Arguments:
      - X: torch.Tensor[B, T, F], the input tensor

    Returns:
      - X_prob: torch.Tensor[B, T, F], the normalized tensor
    """

    # Option 2: Log normalization
    # Turn log-spectrograms into probability distributions
    # Compress amplitudes for the softmax
    if self.magnitude == 'log':
      B, T, F = X.shape
      return torch.softmax(X.flatten(1), dim=1).reshape(B, T, F)

    # Option 1: Linear normalization
    else:
      return X / (X.sum(dim=(1, 2), keepdim=True) + self.eps)  # shape: [B, T, F]


  def _compute_energy_loss(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Computes the energy loss between two audio signals.

    Arguments:
      - X: torch.Tensor[B, T, F], the first audio signal
      - Y: torch.Tensor[B, T, F], the second audio signal

    Returns:
      - energy_loss: torch.Tensor, the computed energy loss
    """
    # Option 1: the total mass instead of bin by bin
    # energy_X = X.pow(2).sum(dim=-1).sqrt()
    # energy_Y = Y.pow(2).sum(dim=-1).sqrt()
    # energy_loss = torch.abs(energy_X - energy_Y).mean()
    # return energy_loss

    # Option 2: the bin by bin mse
    energy_X = X.pow(2).sum(dim=(1, 2), keepdim=True).sqrt()
    energy_Y = Y.pow(2).sum(dim=(1, 2), keepdim=True).sqrt()
    energy_loss = torch.abs(energy_X - energy_Y) / (energy_X + energy_Y + self.eps)
    return energy_loss.mean()


  @staticmethod
  def _compute_sorted_projections(X: int, Y: int, angles: List[int]) -> torch.Tensor:
    """
    Computes the indices of normalised Radon projections given the
    2d matrix size and the list of angles.

    Arguments:
      - X: int, the horizontal dimension
      - Y: int, the vertical dimension
      - angles: List[int], the list of angles
      - sort: bool, whether to sort the coordinates
    Returns:
      - indices: torch.Tensor[n_angles, X*Y], the indices of the sorted projections
    """
    X_bins = torch.linspace(0, 1, X)
    Y_bins = torch.linspace(0, 1, Y)

    X_coords, Y_coords = torch.meshgrid([X_bins, Y_bins], indexing='ij')  # shape: [X, Y]

    fis = torch.linspace(0, 2*math.pi, angles+1)[:-1]

    projections = []

    for fi in fis:
      coords = torch.cos(fi) * X_coords  + torch.sin(fi) * Y_coords
      coords = coords.flatten()
      # projections.append(coords)
      projections.append(torch.argsort(coords))

    projections = torch.stack(projections)  # shape: [n_angles, X*Y]
    return projections
