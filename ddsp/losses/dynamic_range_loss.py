import torch
import torchaudio


class DynamicRangeLoss(torch.nn.Module):
  """
  Per-band dynamic range and noise floor of the reconstruction against the target, in dB.

  The MRSTFT averages over every time-frequency bin, so a spectrum that is right on average but
  has a raised floor and flattened peaks (the "mush" of a decoder averaging many sounds) costs it
  little. Here, per mel band and per item, over time:
    range  |(q_hi - q_lo)(pred) - (q_hi - q_lo)(target)|, weighted by the target's band power
    floor  relu(q_lo(pred) - q_lo(target)), weighted by log band power (quiet bands count too)
  Quantiles go through a sort, so they are differentiable. Mirrors the evaluation's range and
  floor scores (experiments/datasize).

  Arguments:
    - fs: int, sampling rate
    - n_mels, n_fft, hop: the mel analysis
    - q_lo, q_hi: float, the quantiles taken as floor and peak level
  """
  def __init__(self, fs: int = 48000, n_mels: int = 64, n_fft: int = 2048, hop: int = 512,
               q_lo: float = 0.1, q_hi: float = 0.9, floor_weight: float = 1.0):
    super().__init__()
    self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=fs, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
                                                    f_min=30.0, power=2.0)
    self.q_lo, self.q_hi, self.floor_weight = q_lo, q_hi, floor_weight

  def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x: prediction, y: target, [B, 1, T] or [B, T]."""
    x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
    self.mel = self.mel.to(x.device)
    db = lambda s: 10 * torch.log10(self.mel(s) + 1e-10)                # [B, n_mels, frames]
    X, Y = db(x), db(y)
    q = torch.tensor([self.q_lo, 0.5, self.q_hi], device=x.device)
    qx, qy = torch.quantile(X, q, dim=-1), torch.quantile(Y, q, dim=-1)   # [3, B, n_mels]
    with torch.no_grad():
      w = torch.softmax(qy[1] * (torch.log(torch.tensor(10.0)) / 10), dim=-1)     # band power share
      wl = torch.log1p(w * w.shape[-1])
      wl = wl / wl.sum(-1, keepdim=True)
    rng = ((qx[2] - qx[0]) - (qy[2] - qy[0])).abs()
    floor = torch.relu(qx[0] - qy[0])
    return ((rng * w).sum(-1) + self.floor_weight * (floor * wl).sum(-1)).mean()
