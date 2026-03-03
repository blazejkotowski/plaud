import numpy as np
import torch
import librosa
from tqdm import tqdm

from .base_extractor import BaseExtractor
from .utils import smoothen_feature


class PYINPitchExtractor(BaseExtractor):
  """Extract pitch with librosa pYIN and return audio-rate features."""

  def __init__(self,
               postprocess: bool = True,
               smoothing_kernel_size: int = 257,
               fmin: float = 30.0,
               fmax: float = 2000.0,
               frame_length: int = 2048,
               hop_length: int = 256,
               center: bool = True,
               fill_strategy: str = 'interpolate',
               chunk_duration_s: int = 10,
               *args, **kwargs):
    """
    Args:
      - postprocess: optional smoothing in Hz domain (no normalization).
      - smoothing_kernel_size: moving-average kernel size used when postprocess is True.
      - fmin, fmax: search range for pYIN in Hz.
      - frame_length, hop_length: pYIN analysis window / hop.
      - center: forwarded to librosa.pyin.
      - fill_strategy: how to fill unvoiced/invalid values before upsampling.
        Supported: 'interpolate', 'zero'.
      - chunk_duration_s: duration of each processing chunk in seconds.
    """
    super(PYINPitchExtractor, self).__init__(*args, **kwargs)
    self._postprocess = postprocess
    self._smoothing_kernel_size = int(smoothing_kernel_size)
    self._fmin = float(fmin)
    self._fmax = float(fmax)
    self._frame_length = int(frame_length)
    self._hop_length = int(hop_length)
    self._center = bool(center)
    self._fill_strategy = str(fill_strategy)
    self._chunk_duration_s = int(chunk_duration_s)

    if self._fmin <= 0 or self._fmax <= self._fmin:
      raise ValueError(f"Invalid pitch range: fmin={self._fmin}, fmax={self._fmax}")


  def _fill_invalid(self, f0: np.ndarray) -> np.ndarray:
    valid = np.isfinite(f0) & (f0 > 0.0)
    if valid.sum() == 0:
      return np.zeros_like(f0, dtype=np.float32)

    if self._fill_strategy == 'zero':
      out = np.where(valid, f0, 0.0).astype(np.float32)
      return out

    if self._fill_strategy != 'interpolate':
      raise ValueError(f"Unsupported fill_strategy: {self._fill_strategy}")

    idx = np.arange(f0.shape[0], dtype=np.float32)
    out = np.interp(idx, idx[valid], f0[valid]).astype(np.float32)
    return out


  def _extract_chunk_to_audio_rate(self, chunk_np: np.ndarray, fs: int) -> np.ndarray:
    f0, _, _ = librosa.pyin(
      chunk_np,
      fmin=self._fmin,
      fmax=self._fmax,
      sr=fs,
      frame_length=self._frame_length,
      hop_length=self._hop_length,
      center=self._center,
      fill_na=np.nan,
    )

    if f0 is None or len(f0) == 0:
      return np.zeros_like(chunk_np, dtype=np.float32)

    f0 = np.asarray(f0, dtype=np.float32)
    f0 = self._fill_invalid(f0)

    t_frames = librosa.times_like(f0, sr=fs, hop_length=self._hop_length).astype(np.float32)
    t_chunk = (np.arange(chunk_np.shape[0], dtype=np.float32) / float(fs)).astype(np.float32)
    return np.interp(t_chunk, t_frames, f0).astype(np.float32)


  def _calculate(self, audio: torch.Tensor, fs: int) -> torch.Tensor:
    x = audio.float()
    return_with_batch_dim = False

    if x.ndim == 2:
      if x.shape[0] != 1:
        raise ValueError("PYINPitchExtractor expects [T_audio] or [1, T_audio] input")
      x = x.squeeze(0)
      return_with_batch_dim = True
    elif x.ndim != 1:
      raise ValueError("PYINPitchExtractor expects [T_audio] or [1, T_audio] input")

    x_np = x.detach().cpu().numpy().astype(np.float32)
    chunk_samples = max(1, self._chunk_duration_s * fs)
    chunk_f0_audio = []
    for start in tqdm(range(0, x_np.shape[0], chunk_samples), 'pyin pitch extraction'):
      chunk_np = x_np[start:start + chunk_samples]
      chunk_f0_audio.append(self._extract_chunk_to_audio_rate(chunk_np, fs))

    if len(chunk_f0_audio) == 0:
      pitch = torch.zeros_like(x)
    else:
      f0_audio = np.concatenate(chunk_f0_audio, axis=0)
      f0_audio = f0_audio[:x_np.shape[0]]
      pitch = torch.from_numpy(f0_audio).to(x.device)

    if self._postprocess:
      if self._smoothing_kernel_size > 1:
        pitch = smoothen_feature(pitch.unsqueeze(0), window_size=self._smoothing_kernel_size).squeeze(0)

    if return_with_batch_dim:
      pitch = pitch.unsqueeze(0)

    return pitch
