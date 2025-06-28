import torch
import math

class SlicedWassersteinLoss(torch.nn.Module):
  """
  Sliced Wasserstein Loss for comparing two audio signals.
  This loss computes the Sliced Wasserstein solution to the Optimal Tranposrt problem,
  as described in the following paper paper.
  Fabiani, L., Schlecht, S. J., & Elvander, F. (2024).
  Time-Frequency Audio Similarity Using Optimal Transport.
  2024 58th Asilomar Conference on Signals, Systems, and Computers, 1414–1417.
  https://doi.org/10.1109/IEEECONF60004.2024.10943074

  Args:
    - win_size: int, the size of the STFT window
    - hop_size: int, the hop size of the STFT
    - n_projections: int, the number of projections to use for the sliced Wasserstein distance
    - device: str, the device to use for the computation (e.g., 'cuda' or 'cpu')
  """

  eps = 1e-8
  k = 1e6 # regulatization constant from the paper

  def __init__(self, win_size: int = 2048, hop_size: int = 512, n_projections: int = 10, sampling_rate: int = 44100, device: str = 'cuda'):
    super(SlicedWassersteinLoss, self).__init__()
    self.win_size = win_size
    self.hop_size = hop_size
    self.n_projections = n_projections
    self.sampling_rate = sampling_rate
    self.device = device


  def stft(self, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Short-Time Fourier Transform (STFT) of the input signal.

    Args:
      - x: torch.Tensor, the input audio signal

    Returns:
      - X: torch.Tensor, the magnitude spectrum of the STFT
    """
    x_stft = torch.stft(x, n_fft=self.win_size, hop_length=self.hop_size, return_complex=True)
    x_mag = torch.sqrt(torch.clamp(x_stft.real**2 + x_stft.imag**2, min=self.eps))
    # x_mag = (x_stft.real**2+x_stft.imag**2
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
    # Ensure the input tensors are the same shape
    if x.shape != y.shape:
      raise ValueError("Input tensors must have the same shape.")

    X_mag = self.stft(x).permute(0, 2, 1)  # shape: [B, T, F]
    Y_mag = self.stft(y).permute(0, 2, 1)  # shape: [B, T, F]

    # TODO: Make sure that this is necessary & correct
    X_mag = X_mag / (X_mag.sum(dim=(1, 2), keepdim=True) + self.eps)
    Y_mag = Y_mag / (X_mag.sum(dim=(1, 2), keepdim=True) + self.eps)

    B, T, F = X_mag.shape

    time_bins = torch.linspace(0, x.shape[-1]/self.sampling_rate, T)
    freq_bins = torch.linspace(0, self.sampling_rate//2, F)
    t_coords, f_coords = torch.meshgrid([time_bins, freq_bins], indexing='ij')

    thetas = torch.arange(0, self.n_projections) * 2 * torch.pi / self.n_projections
    diffs = []
    for theta in thetas:
      projected_coords = t_coords*torch.cos(theta)*math.sqrt(self.k) + f_coords*torch.sin(theta)
      sorted_idx = torch.argsort(projected_coords.flatten())
      X_vec = X_mag.reshape(B, -1)
      Y_vec = Y_mag.reshape(B, -1)
      x_cdf = torch.cumsum(X_vec[:, sorted_idx], -1)
      # x_cdf /= x_cdf.sum()
      y_cdf = torch.cumsum(Y_vec[:, sorted_idx], -1)
      # y_cdf /= y_cdf.sum()

      diffs.append(((x_cdf - y_cdf)**2).sum())

    return torch.vstack(diffs).mean()
