from tqdm import tqdm

import torch
import torchcrepe
import torch.nn.functional as F
import numpy as np

from .base_extractor import BaseExtractor
from .utils import normalize_feature, smoothen_feature

class CrepePitchExtractor(BaseExtractor):
  """
  Extracts the pitch from an audio signal
  using CREPE algorithm
  """
  def __init__(self,
               postprocess: bool = True,
               smoothing_kernel_size: int = 257,
               fmin: float = 30.0,
               fmax: float = 2000.0,
               model: str = 'full',
               periodicity_threshold: float = 0.21,
               chunk_duration_s: int = 10,
               *args, **kwargs):
    """
    Args:
      - postprocess: bool, whether to apply postprocessing:
        > clamping between .05 and .95 percentiles,
        > normalization between [0, 1]
        > and smoothing to the extracted features
      - smoothing_kernel_size: in case of processing, the kernel size of the gaussian filtere
    """
    super(CrepePitchExtractor, self).__init__(*args, **kwargs)
    self._smoothing_kernel_size = smoothing_kernel_size
    self._postprocess = postprocess
    self._fmin = float(fmin)
    self._fmax = float(fmax)
    self._model = model
    self._periodicity_threshold = float(periodicity_threshold)
    self._chunk_duration_s = int(chunk_duration_s)


  @staticmethod
  def _interpolate_invalid_1d(x: np.ndarray) -> np.ndarray:
    """Fill NaN/inf values by linear interpolation in 1D."""
    valid = np.isfinite(x)
    if valid.sum() == 0:
      return np.zeros_like(x, dtype=np.float32)
    if valid.sum() == 1:
      out = np.full_like(x, x[valid][0], dtype=np.float32)
      return out

    idx = np.arange(x.shape[0], dtype=np.float32)
    out = np.interp(idx, idx[valid], x[valid]).astype(np.float32)
    return out

  def _calculate(self, audio: torch.Tensor, fs: int) -> torch.Tensor:
    """
    Args:
      - audio: torch.Tensor [T_audio] or [B, T_audio], the input audio tensor
      - fs: int, sampling rate
    Returns:
      - features: torch.Tensor [T_audio, C] or [B, T_audio, C], the extracted features at audio rate
    """

    squeeze_back = False
    if audio.ndim == 1:
      audio = audio.unsqueeze(0)
      squeeze_back = True

    x = audio.float()
    crepe_device = x.device

    chunk_samples = self._chunk_duration_s * fs
    pitches = []
    periodicities = []

    with torch.no_grad():
      # process in chunks to avoid memory overload
      for i in tqdm(range(0, x.shape[-1], chunk_samples), 'pitch extraction'):
        chunk = x[..., i:i+chunk_samples].to(crepe_device)
        pitch_chunk, periodicity_chunk = torchcrepe.predict(
          chunk,
          fs,
          fmin=self._fmin,
          fmax=self._fmax,
          model=self._model,
          device=crepe_device,
          return_periodicity=True,
        )
        pitches.append(pitch_chunk)
        periodicities.append(periodicity_chunk)

    pitch = torch.cat(pitches, dim=-1)
    periodicity = torch.cat(periodicities, dim=-1)

    # Mask unvoiced / low-confidence estimates before interpolation.
    voiced = (periodicity >= self._periodicity_threshold) & torch.isfinite(pitch) & (pitch > 0)
    pitch = torch.where(voiced, pitch, torch.full_like(pitch, float('nan')))

    # Fill invalid frame-level values using linear interpolation per batch item.
    pitch_np = pitch.detach().cpu().numpy()
    for b in range(pitch_np.shape[0]):
      pitch_np[b] = self._interpolate_invalid_1d(pitch_np[b])
    pitch = torch.from_numpy(pitch_np).to(crepe_device)

    # Interpolate from frame-rate CREPE output to audio-rate features.
    pitch = F.interpolate(
      pitch.unsqueeze(1),
      size=x.shape[-1],
      mode='linear',
      align_corners=False,
    ).squeeze(1)

    if self._postprocess:
      pitch = normalize_feature(pitch, low=5.0, high=95.0, dim=-1)
      if self._smoothing_kernel_size > 1:
        pitch = smoothen_feature(pitch, window_size=self._smoothing_kernel_size)

    if squeeze_back:
      pitch = pitch.squeeze(0)

    return pitch

