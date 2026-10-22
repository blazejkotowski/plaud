import math

import torch


class EnvelopeLoss(torch.nn.Module):
  """
  Time-aligned band-envelope error of the reconstruction against the target, in dB.

  Per log-spaced band, the power envelope over time is smoothed (20 ms and 100 ms by default) and compared frame
  by frame, in both directions: a smeared or held decay costs as much as a cut one, a soft attack as much as a
  raised floor. Unlike a bin-wise spectral loss it doesn't ask for the random fine structure of noise (which no
  decoder can predict, so it pulls toward the average), only for the envelope the controls can explain.
  Bands are weighted toward the ones the target uses (log of its energy share), and levels carry a floor 80 dB
  below the target's loudest band frame, added in power so quiet errors fade out smoothly. Mirrors env20/env100
  of experiments/metrics/aligned.py.

  Arguments:
    - fs: int, sampling rate
    - n_fft, hop: the STFT
    - n_bands, f_min: the bands (log-spaced from f_min to fs / 2)
    - smooth_ms: list of envelope smoothing times, one term each (averaged)
    - range_db: float, floor below the target's maximum (silent targets get an absolute floor)
    - cap_db: float, per-frame differences are softly capped here, so a silent chunk can't dominate
  """
  def __init__(self, fs: int = 48000, n_fft: int = 2048, hop: int = 128, n_bands: int = 48, f_min: float = 40.0,
               smooth_ms=(20.0, 100.0), range_db: float = 80.0, cap_db: float = 40.0):
    super().__init__()
    self.n_fft, self.hop, self.range_db, self.cap_db = n_fft, hop, range_db, cap_db
    f = torch.linspace(0, fs / 2, n_fft // 2 + 1)
    edges = torch.logspace(math.log10(f_min), math.log10(fs / 2), n_bands + 1)
    edges[-1] = fs / 2 + 1
    self.register_buffer("bands", ((f[None] >= edges[:-1, None]) & (f[None] < edges[1:, None])).float(), persistent=False)
    self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
    self.kernels = [max(1, int(round(ms / 1000 * fs / hop))) for ms in smooth_ms]

  def _band_power(self, s):
    S = torch.stft(s, self.n_fft, self.hop, window=self.window.to(s.device), return_complex=True).abs() ** 2
    return torch.einsum("kf,bft->bkt", self.bands.to(s.device), S)              # [B, bands, frames]

  def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x: prediction, y: target, [B, C, T] or [B, T]."""
    x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
    Px, Py = self._band_power(x), self._band_power(y)
    with torch.no_grad():
      floor = (Py.amax(dim=(1, 2), keepdim=True) * 10 ** (-self.range_db / 10)).clamp_min(1e-10)   # silent chunks
      share = Py.mean(-1) / Py.mean(-1).sum(-1, keepdim=True).clamp_min(1e-20)
      w = torch.log1p(share * share.shape[-1])
      w = (w + 1e-6) / (w + 1e-6).sum(-1, keepdim=True)                          # [B, bands]; flat if silent
    loss = 0.0
    for k in self.kernels:
      sm = lambda P: torch.nn.functional.avg_pool1d(P, k, stride=1, padding=k // 2, count_include_pad=False) if k > 1 else P
      d = (10 * torch.log10(sm(Px) + floor) - 10 * torch.log10(sm(Py) + floor)).abs()
      d = (self.cap_db * torch.tanh(d / self.cap_db)).mean(-1)                  # soft cap: silence vs sound
      loss = loss + (d * w).sum(-1).mean()
    return loss / len(self.kernels)
