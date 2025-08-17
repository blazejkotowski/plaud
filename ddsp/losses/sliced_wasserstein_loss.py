import torch
import math
import scipy.signal

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
    - sampling_rate: int, the sampling rate for the STFT
    - p: int, the p in the p-Wasserstein distance
    - device: str, the device to use for the computation (e.g., 'cuda' or 'cpu')
  """

  eps = 1e-8
  k = 1e1 # regulatization constant from the paper

  def __init__(self, win_size: int = 512, hop_size: int = 256, n_projections: int = 10, sampling_rate: int = 44100, p: int = 2, device: str = 'cuda'):
    super(SlicedWassersteinLoss, self).__init__()
    self.win_size = win_size
    self.hop_size = hop_size
    self.n_projections = n_projections
    self.sampling_rate = sampling_rate
    self.p = p
    self.device = device

    scipy_window = scipy.signal.windows.flattop(self.win_size, sym=False)  # sym=False for FFT use
    self.stft_window = torch.from_numpy(scipy_window).to(dtype=torch.float32, device=self.device)


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
    x_mag = torch.log(x_mag)
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

    # X_mag = X_mag / (X_mag.sum() + eps)
    # Y_mag = Y_mag / (Y_mag.sum() + eps)
    X_mag = torch.softmax(X_mag.flatten(1), dim=1).reshape(B, T, F)
    Y_mag = torch.softmax(Y_mag.flatten(1), dim=1).reshape(B, T, F)

    freq_bins = torch.linspace(0, F-1, F)
    time_bins = torch.linspace(0, T-1, T)

    t_coords, f_coords = torch.meshgrid([time_bins, freq_bins], indexing='ij') # shape: [T, F]

    thetas = torch.linspace(0, 2*math.pi, self.n_projections+1)[:-1]

    diffs = []
    for theta in thetas:
      coords = torch.cos(theta) * math.sqrt(self.k) * t_coords  + torch.sin(theta) * f_coords

      # Flatten everything
      coords = coords.flatten() # shape: [T*F]
      X_mag = X_mag.flatten(1) # shape: [B, T*F]
      Y_mag = Y_mag.flatten(1) # shape: [B, T*F]

      # Sort
      sorted_idx = torch.argsort(coords)

      # coords_sorted = coords[sorted_idx]
      X_mag_sorted = X_mag[:, sorted_idx]
      Y_mag_sorted = Y_mag[:, sorted_idx]

      # probability density functions
      x_cdf = torch.cumsum(X_mag_sorted, dim=1)
      y_cdf = torch.cumsum(Y_mag_sorted, dim=1)

      # diff = torch.trapz(torch.abs(x_pdf - y_pdf), coords_sorted).sum()
      diff = torch.pow(x_cdf - y_cdf, self.p).sum() / (F) # normalize by the number of bins
      # diff = torch.trapz(cdf_diff, coords_sorted.unsqueeze(0).expand(B, -1), dim=1)
      diffs.append(diff)

    return torch.stack(diffs).mean() / math.sqrt(self.n_projections)  # normalize by the number of projections
